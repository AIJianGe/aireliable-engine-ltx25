"""LTX-2.5 模型包装器（四分支量化加载版）。

负责 LTX-2.5 模型权重的实际加载、卸载与推理调用。

四分支量化加载（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §5 + M4-T4.2）：
    - F16 档（quantization.type=none）：
        LTX2Pipeline.from_pretrained(diffusers_dir, torch_dtype=float16)
        BF16 权重自动 cast 为 FP16（Turing SM 7.5 不支持 BF16，数值转换无损）
    - I8 档（quantization.type=torchao_int8）：
        transformer 用 TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
        运行时量化，其余组件 FP16 常驻（共享组件走 diffusers_format/）
    - G8 档（quantization.type=gguf_q8）：
        ltxv 原生推理（LTXVNativeRunner，见 ltxv_native.py）。
        diffusers from_single_file 路径已废弃：GGUF 权重是 LTXV 原生架构
        （ltx_core 方言），与 diffusers LTX2VideoTransformer3DModel 键维度
        不匹配，无法通过 from_single_file 加载。改走 ltx_pipelines blocks
        两阶段蒸馏管线（ANALYSIS-TASKS/g8_prototype.py 已验证输出 mp4），
        模型分时加载（Gemma → transformer → VAE），无 diffusers pipeline
    - G4 档（quantization.type=gguf_q4，M4-T4.2 新增）：
        与 G8 档同管线（LTXVNativeRunner），仅 GGUF 权重换为
        Q4_K_M 量化（*Q4_K_M*.gguf，15.7GB），低显存/低磁盘场景用

CPU offload 分级（development-guide.md §37.2.7）：
    - L0: 全 GPU（所有模型常驻 GPU）
    - L1: enable_model_cpu_offload（组件级 offload，TE 编码完后回 CPU）
    - L2: enable_sequential_cpu_offload（层级别 offload，慢 3-5×）
    - L3: 全 CPU（device=cpu + VAE tiling）

禁止兜底下载（project-rules.md 第二部分 §2）：
    load() 只做加载，不下载。文件缺失时直接抛 RuntimeError，
    指引用户到模型管理界面下载。

distilled 版推理约束：
    - num_inference_steps 固定 8（蒸馏版）
    - guidance_scale 固定 1.0（CFG=1，负向提示词无效）
    - 分辨率需被 32 整除（VAE 空间压缩 32×）
    - 帧数需满足 8k+1（时间压缩 8×，如 121 帧 ≈ 5s@24fps）
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Optional

from .scheme_manager import SchemeConfig

logger = logging.getLogger(__name__)


# 帧数对齐常量：LTX-2.5 时间压缩 8×，帧数需满足 8k+1
_FRAME_ALIGNMENT = 8  # 8k + 1（k ≥ 0，首帧 + 8 的倍数）

# 时长上限（秒）：两阶段蒸馏管线约束（48/50fps 高帧率同此上限）
_MAX_DURATION_S = 10


def _clamp_duration(duration_s: int, logger: Optional[logging.Logger] = None) -> int:
    """时长 >10s 自动降级到 10s（两阶段管线约束）。

    Args:
        duration_s: 请求时长（秒）
        logger: 可选日志器（记录降级事件）

    Returns:
        对齐后的时长（秒，≤ _MAX_DURATION_S，至少 1s）
    """
    if duration_s > _MAX_DURATION_S:
        if logger:
            logger.warning(
                "时长 %ds 超上限，自动降级到 %ds（两阶段蒸馏管线约束）",
                duration_s, _MAX_DURATION_S,
            )
        return _MAX_DURATION_S
    return max(1, duration_s)


def _align_num_frames(duration_s: int, fps: int) -> int:
    """把 duration×fps 的目标帧数对齐到 8k+1 格式。

    LTX-2.5 VAE 时间压缩 8×，帧数必须满足 8k+1（如 1, 9, 17, ..., 121）。

    Args:
        duration_s: 时长（秒）
        fps: 帧率

    Returns:
        对齐后的帧数（8k+1）
    """
    target = duration_s * fps
    # 8k+1 格式：k = round((target - 1) / 8)，至少 1 帧
    k = max(0, round((target - 1) / _FRAME_ALIGNMENT))
    return k * _FRAME_ALIGNMENT + 1


class LTX25Wrapper:
    """LTX-2.5 模型包装器。

    封装 LTX2Pipeline 的加载/卸载/推理，管理三分支量化与 CPU offload。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化包装器。

        Args:
            models_root: 模型根目录（models/）
            model_dir: LTX-2.5 模型目录（models/video/ltx25/，
                含 diffusers_format/ 与 gguf_format/ 子目录）
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self._logger = logger or logging.getLogger(__name__)

        # ===== 已加载状态 =====
        self._pipeline = None          # LTX2Pipeline 实例（F16/I8 档）
        self._native_runner = None    # LTXVNativeRunner 实例（G8 档，blocks 分时加载，无常驻权重）
        self._loaded_variant: Optional[str] = None
        self._loaded_scheme_id: Optional[str] = None
        self._device: str = "cpu"
        self._offload_level: str = "L1"
        self._load_time: float = 0.0

    # ===== 状态查询 =====

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载（diffusers pipeline 或 G8 原生 runner 任一就绪）。

        G8 档加载后 _pipeline 保持 None（blocks 分时加载架构无常驻 pipeline），
        以 _native_runner 就绪作为加载态判据。
        """
        return self._pipeline is not None or self._native_runner is not None

    @property
    def loaded_variant(self) -> Optional[str]:
        """已加载的 variant 名。"""
        return self._loaded_variant

    @property
    def loaded_scheme_id(self) -> Optional[str]:
        """已加载的方案 ID（F16 / I8 / G8）。"""
        return self._loaded_scheme_id

    @property
    def device(self) -> str:
        """当前推理设备。"""
        return self._device

    @property
    def offload_level(self) -> str:
        """当前 CPU offload 级别。"""
        return self._offload_level

    def get_vram_used(self) -> float:
        """查询当前进程 GPU 显存占用（MB）。"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    # ===== 加载 =====

    def load(
        self,
        variant: str,
        variant_path: str,
        scheme: SchemeConfig,
        device: str = "auto",
    ) -> None:
        """加载模型权重（三分支量化 + offload 分级）。

        遵循 project-rules.md 第二部分 §2：只做加载，不下载。
        文件缺失时抛 RuntimeError 指引下载。

        Args:
            variant: variant 名（LTX/LTX-2.5-Diffusers / LTX/LTX-2.5-GGUF-Q8 / LTX/LTX-2.5-GGUF-Q4）
            variant_path: variant 目录路径（model_dir/subdir）
            scheme: 方案配置（F16 / I8 / G8 / G4）
            device: 推理设备（auto / cpu / cuda:0）

        Raises:
            RuntimeError: 模型文件缺失或加载失败
        """
        load_start = time.time()
        self._logger.info(
            "LTX-2.5 加载开始: variant=%s, scheme=%s, quant=%s, offload=%s, device=%s",
            variant, scheme.scheme, scheme.quantization.type,
            scheme.offload_level, device,
        )

        # ===== 1. 解析设备 =====
        resolved_device = self._resolve_device(device)
        self._device = resolved_device

        # ===== 2. 文件完整性预校验（不下载，只校验）=====
        self._check_files_integrity(scheme)

        # ===== 3. 四分支量化加载 =====
        quant_type = scheme.quantization.type
        if quant_type in ("gguf_q8", "gguf_q4"):
            self._load_gguf_branch(variant, scheme, resolved_device)
        elif quant_type == "torchao_int8":
            self._load_int8_branch(variant, scheme, resolved_device)
        elif quant_type == "none":
            self._load_fp16_branch(variant, scheme, resolved_device)
        else:
            raise RuntimeError(
                f"未知量化类型: {quant_type}（支持: none / torchao_int8 / gguf_q8 / gguf_q4）"
            )

        # ===== 4. 应用 CPU offload 策略 =====
        self._offload_level = scheme.offload_level
        self._apply_offload_strategy(scheme, resolved_device)

        # ===== 5. 记录状态 =====
        self._loaded_variant = variant
        self._loaded_scheme_id = scheme.scheme
        self._load_time = time.time() - load_start

        vram_used = self.get_vram_used()
        self._logger.info(
            "LTX-2.5 加载完成: variant=%s, scheme=%s, device=%s, offload=%s, "
            "load_time=%.1fs, vram_used=%.1fMB",
            variant, scheme.scheme, self._device, self._offload_level,
            self._load_time, vram_used,
        )

    def _resolve_device(self, device: str) -> str:
        """解析设备字符串为具体值。

        Args:
            device: auto / cpu / cuda:0

        Returns:
            具体设备字符串（cuda:0 或 cpu）
        """
        if device == "cpu":
            return "cpu"
        if device.startswith("cuda"):
            return device
        # auto：检测 CUDA 可用性
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def _check_files_integrity(self, scheme: SchemeConfig) -> None:
        """校验模型文件完整性（不下载，只校验，遵循 project-rules.md §2）。

        按方案分支检查关键文件（标志文件 + 大权重文件存在性），
        缺失时抛 RuntimeError 指引用户到模型管理界面下载。

        Args:
            scheme: 方案配置

        Raises:
            RuntimeError: 文件缺失
        """
        missing: list[str] = []

        if scheme.quantization.type in ("gguf_q8", "gguf_q4"):
            # G8 / G4 档（ltxv 原生管线）：只用 GGUF + comfyui_format 组件，
            # 不依赖 diffusers_format/ 共享组件（校验由
            # _load_gguf_branch → LTXVNativeRunner.check_files 完成）
            gguf_dir = self.model_dir / "gguf_format"
            gguf_candidates = sorted(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
            if not gguf_candidates:
                missing.append(f"{gguf_dir}/（GGUF 目录下无任何 .gguf 文件）")
        else:
            # F16 / I8 档（diffusers 管线）：检查 diffusers_format/ 标志文件
            diffusers_dir = self.model_dir / "diffusers_format"
            shared_markers = [
                diffusers_dir / "model_index.json",
                diffusers_dir / "transformer" / "config.json",
                diffusers_dir / "text_encoder" / "config.json",
                diffusers_dir / "vae" / "config.json",
            ]
            for p in shared_markers:
                if not p.exists():
                    missing.append(str(p))

        if missing:
            raise RuntimeError(
                f"模型文件不完整，缺失 {len(missing)} 个文件：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                f"选择镜像源后下载。\n"
                f"本地路径：{self.model_dir}\n"
                f"HuggingFace 仓库：{scheme.model_repo}"
            )

    # ===== 三分支量化加载实现 =====

    def _load_fp16_branch(
        self,
        variant: str,
        scheme: SchemeConfig,
        device: str,
    ) -> None:
        """F16 档：BF16 权重 cast 为 FP16 加载完整 pipeline。

        Turing SM 7.5 不支持 BF16，from_pretrained 的 torch_dtype=float16
        会把 BF16 权重自动转换为 FP16（数值转换无损）。
        """
        import torch
        from diffusers import LTX2Pipeline

        diffusers_dir = self.model_dir / "diffusers_format"
        self._logger.info(
            "F16 档加载（BF16→FP16 cast）: %s", diffusers_dir,
        )

        self._pipeline = LTX2Pipeline.from_pretrained(
            str(diffusers_dir),
            torch_dtype=torch.float16,
        )
        self._logger.info("F16 档 pipeline 加载完成（dtype=float16）")

    def _load_int8_branch(
        self,
        variant: str,
        scheme: SchemeConfig,
        device: str,
    ) -> None:
        """I8 档：torchao INT8 运行时量化加载。

        transformer 用 TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
        量化（INT8 Tensor Core，Turing SM 7.5 支持），
        其余组件（TE/VAE）保持 FP16。
        """
        import torch
        from diffusers import LTX2Pipeline, TorchAoConfig
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig

        diffusers_dir = self.model_dir / "diffusers_format"
        self._logger.info(
            "I8 档加载（torchao INT8 运行时量化）: %s", diffusers_dir,
        )

        # torchao>=0.13 起 TorchAoConfig 只接受 AOBaseConfig 实例
        # （字符串 "int8_weight_only" 已废弃）；Int8DynamicActivationInt8WeightConfig
        # 已在 T1.2 于 sm75 上前向验证通过
        quant_config = TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
        self._pipeline = LTX2Pipeline.from_pretrained(
            str(diffusers_dir),
            torch_dtype=torch.float16,
            quantization_config=quant_config,
        )
        self._logger.info("I8 档 pipeline 加载完成（transformer INT8 量化）")

    def _load_gguf_branch(
        self,
        variant: str,
        scheme: SchemeConfig,
        device: str,
    ) -> None:
        """G8/G4 档：GGUF ltxv 原生推理（blocks 分时加载架构）。

        diffusers from_single_file 路径已废弃：GGUF 权重是 LTXV 原生架构
        （ltx_core 方言），与 diffusers LTX2VideoTransformer3DModel 键维度
        不匹配，无法通过 from_single_file 加载。改用 ltxv 原生两阶段蒸馏
        管线（ANALYSIS-TASKS/g8_prototype.py 已验证输出 mp4 成功）。

        G8 档（gguf_q8）匹配 *Q8_0*.gguf，G4 档（gguf_q4）匹配
        *Q4_K_M*.gguf，反量化均走 gguf.dequantize 通用接口。

        blocks 架构下 load() 阶段只校验文件、不预加载模型；
        generate() 时分时加载（Gemma → transformer → VAE），
        每个阶段用完释放显存（48GB Turing 卡约束）。
        文件校验失败抛 RuntimeError 指引用户到模型管理界面下载
        （遵循 project-rules.md 第二部分 §2：只加载，不下载）。
        """
        import torch

        from .ltxv_native import LTXVNativeRunner

        # 按方案量化类型选择 GGUF 权重匹配模式（G8 → Q8_0 / G4 → Q4_K_M）
        if scheme.quantization.type == "gguf_q4":
            gguf_glob, gguf_fallback = "*Q4_K_M*.gguf", "transformer-Q4_K_M.gguf"
        else:
            gguf_glob, gguf_fallback = "*Q8_0*.gguf", "transformer-Q8_0.gguf"

        # Turing SM 7.5 不支持 BF16 → FP16（与 F16/I8 档一致）
        runner = LTXVNativeRunner(
            model_dir=self.model_dir,
            device=device,
            dtype=torch.float16,
            logger=self._logger,
            gguf_glob=gguf_glob,
            gguf_fallback_name=gguf_fallback,
        )

        missing = runner.check_files()
        if missing:
            raise RuntimeError(
                f"模型文件不完整，缺失 {len(missing)} 个文件：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                f"选择镜像源后下载。\n"
                f"本地路径：{self.model_dir}\n"
                f"HuggingFace 仓库：{scheme.model_repo}"
            )

        self._native_runner = runner
        # 保持 self._pipeline 为 None（blocks 分时加载架构，无 diffusers pipeline）
        self._logger.info(
            "%s 档 ltxv 原生 runner 就绪（文件校验通过，模型按需分时加载）: gguf=%s",
            scheme.scheme, runner.gguf_path,
        )

    # ===== CPU offload 策略 =====

    def _apply_offload_strategy(self, scheme: SchemeConfig, device: str) -> None:
        """应用 CPU offload 策略。

        L0: 全 GPU（不启用任何 offload，pipeline.to(device)）
        L1: enable_model_cpu_offload（组件级 offload）
        L2: enable_sequential_cpu_offload（层级别 offload，慢）
        L3: 全 CPU（device=cpu + VAE tiling）

        另外按 scheme.vae_tiling 配置启用 VAE tiling。

        Args:
            scheme: 方案配置
            device: 解析后的设备
        """
        if self._pipeline is None:
            return

        level = scheme.offload_level

        if device == "cpu":
            # L3 / 强制 CPU：pipeline 移到 CPU
            self._pipeline.to("cpu")
            self._logger.info("offload 策略: 全 CPU（device=cpu）")
        elif level == "L0":
            self._pipeline.to(device)
            self._logger.info("offload 策略: L0 全 GPU（%s）", device)
        elif level == "L1":
            self._pipeline.enable_model_cpu_offload()
            self._logger.info("offload 策略: L1 组件级 offload（enable_model_cpu_offload）")
        elif level in ("L2", "L3"):
            self._pipeline.enable_sequential_cpu_offload()
            self._logger.info("offload 策略: %s 层级别 offload（enable_sequential_cpu_offload）", level)
        else:
            # 未知级别按 L1 处理
            self._pipeline.enable_model_cpu_offload()
            self._logger.warning("未知 offload 级别 %s，按 L1 处理", level)

        # VAE tiling（低显存优化）
        if scheme.vae_tiling.enabled:
            try:
                self._pipeline.enable_vae_tiling()
                self._logger.info(
                    "VAE tiling 已启用（tile_size=%d）", scheme.vae_tiling.tile_size,
                )
            except Exception as e:
                self._logger.warning("VAE tiling 启用失败（不影响运行）: %s", e)

    # ===== 推理 =====

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        duration: int = 5,
        fps: int = 24,
        width: int = 960,
        height: int = 544,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = -1,
        image=None,
        images=None,
        num_frames: Optional[int] = None,
        scheme: Optional[SchemeConfig] = None,
        auto_duration: bool = False,
        temporal_upscalings: int = 0,
        workflow: Optional[str] = None,
    ) -> dict:
        """执行视频生成推理。

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词（distilled CFG=1 时无效）
            duration: 时长（秒）
            fps: 帧率
            width: 宽度（32 的倍数）
            height: 高度（32 的倍数）
            num_inference_steps: 推理步数（None=用方案默认，distilled 固定 8）
            guidance_scale: CFG 引导强度（None=用方案默认，distilled 固定 1.0）
            seed: 随机种子（-1=随机）
            image: 【向后兼容】单张首帧图（PIL.Image，None=t2v 模式）
            images: 多图条件列表（新接口，优先于 image）：
                每项可为 schemas.ImageInput 实例或 dict，
                需包含 path + frame_idx + strength + 可选 pil
            num_frames: 显式帧数（None=由 duration×fps 对齐 8k+1）
            scheme: 当前方案配置（取推理默认值）
            auto_duration: 自动时长开关（T3.3，G8 原生链路 DurationHead
                按提示词预测帧数；仅 G8 档生效，文件缺失时降级手动时长）

        Returns:
            dict 含：
                - video_frames: 视频帧序列 (T, H, W, C) uint8
                - audio_data: 音频数据（G8 原生链路返回 vocoder 解码的
                  waveform；diffusers 分支恒为 None）
                - metadata: 元信息（实际步数 / 帧数 / 种子 / mode /
                  num_conditioning_images / conditioning_frame_idxs 等）

        Raises:
            RuntimeError: 模型未加载或推理失败
        """
        # 时长上限降级（两分支共用，>10s 自动降到 10s）
        duration = _clamp_duration(duration, self._logger)

        # === 向后兼容：image（单图旧签名）→ images（多图新签名）===
        if images is None and image is not None:
            images = [{"pil": image, "path": None, "role": "reference", "frame_idx": 0, "strength": 1.0}]

        # ===== 原生推理分支（G8/G4 档 ltxv blocks 架构，支持多图！）=====
        if self._native_runner is not None and self._pipeline is None:
            return self._generate_native(
                prompt=prompt,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                seed=seed,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image=image,        # 仍传旧签名保持单图兼容
                images=images,      # 新多图签名
                auto_duration=auto_duration,
                temporal_upscalings=temporal_upscalings,
                workflow=workflow,
            )

        if self._pipeline is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        # ===== diffusers 分支（F16/I8 档）=====
        # diffusers LTX2ImageToVideoPipeline 仅支持单图 I2V
        # 多图请求报错引导切换 G8/G4 原生档
        if images and len(images) > 1:
            raise RuntimeError(
                "F16/I8 档暂不支持多图生视频（首尾帧/多关键帧）。\n"
                "原因：diffusers LTX2ImageToVideoPipeline 仅暴露单图 I2V 接口。\n"
                "解决方案：切换到 G8/G4 原生档（Settings → Schemes → 选择 G8）。"
            )

        import torch

        # ===== 1. 参数解析（inputs > scheme 默认 > distilled 固定值）=====
        if num_inference_steps is None:
            num_inference_steps = (
                scheme.inference.num_inference_steps if scheme else 8
            )
        if guidance_scale is None:
            guidance_scale = (
                scheme.inference.guidance_scale if scheme else 1.0
            )

        # 分辨率对齐 32 的倍数（VAE 空间压缩 32×）
        width = max(256, (width // 32) * 32)
        height = max(256, (height // 32) * 32)

        # 帧数对齐 8k+1（时间压缩 8×）
        if num_frames is None:
            num_frames = _align_num_frames(duration, fps)

        # 种子
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ===== 2. 构造调用参数 =====
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "generator": generator,
        }
        # diffusers 分支：单图兼容（image 或 images[0].pil）
        single_image = image
        if single_image is None and images and len(images) == 1:
            single_image = images[0].get("pil") if isinstance(images[0], dict) else getattr(images[0], "pil", None)
        if single_image is not None:
            call_kwargs["image"] = single_image

        mode_label = (
            "t2v" if single_image is None else "i2v"
        )
        self._logger.info(
            "LTX-2.5 diffusers 推理开始: steps=%d, cfg=%.2f, size=%dx%d, frames=%d, seed=%d, mode=%s",
            num_inference_steps, guidance_scale, width, height, num_frames,
            seed, mode_label,
        )

        # ===== 3. 执行推理 =====
        infer_start = time.time()
        try:
            output = self._pipeline(**call_kwargs)
        except Exception as e:
            self._logger.exception("LTX-2.5 推理失败")
            raise RuntimeError(f"LTX-2.5 推理失败: {e}") from e
        infer_time = time.time() - infer_start

        # ===== 4. 提取帧序列 =====
        # LTX2PipelineOutput.frames: Tensor (T, C, H, W) 或 list[PIL.Image]
        frames_tensor = getattr(output, "frames", None)
        if frames_tensor is None:
            raise RuntimeError(f"pipeline 输出缺少 frames 字段: {type(output)}")

        video_frames = self._frames_to_uint8(frames_tensor)

        self._logger.info(
            "LTX-2.5 推理完成: infer_time=%.1fs, frames=%d, size=%s",
            infer_time, video_frames.shape[0], video_frames.shape[1:3],
        )

        return {
            "video_frames": video_frames,
            "audio_data": None,  # v1 未启用音频链路（M3 扩展项）
            "metadata": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "seed": seed,
                "infer_time": infer_time,
                "is_placeholder": False,
            },
        }

    def _frames_to_uint8(self, frames):
        """把 pipeline 输出帧转换为 (T, H, W, C) uint8 numpy 数组。

        Args:
            frames: Tensor (T, C, H, W) float [0,1] 或 list[PIL.Image]

        Returns:
            numpy 数组 (T, H, W, C) uint8
        """
        import numpy as np

        # Tensor (T, C, H, W) → (T, H, W, C) uint8
        if hasattr(frames, "cpu"):
            arr = frames.detach().cpu().float().numpy()
            if arr.ndim == 4 and arr.shape[1] in (1, 3):
                # (T, C, H, W) → (T, H, W, C)
                arr = arr.transpose(0, 2, 3, 1)
            arr = np.clip(arr, 0.0, 1.0)
            return (arr * 255).astype(np.uint8)

        # list[PIL.Image] → (T, H, W, C)
        if isinstance(frames, (list, tuple)):
            arrays = [np.asarray(f.convert("RGB")) for f in frames]
            return np.stack(arrays).astype(np.uint8)

        raise RuntimeError(f"无法识别的帧格式: {type(frames)}")

    # ===== G8 档原生推理（ltxv blocks 架构转发）=====

    def _generate_native(
        self,
        prompt: str,
        duration: int,
        fps: int,
        width: int,
        height: int,
        seed: int,
        num_frames: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        image=None,
        images=None,
        auto_duration: bool = False,
        temporal_upscalings: int = 0,
        workflow: Optional[str] = None,
    ) -> dict:
        """原生推理分支：转发给 LTXVNativeRunner.generate()（t2v / i2v / FL2V / 多关键帧）。

        原生两阶段蒸馏管线的步数/CFG 由固定 sigma 表决定（DISTILLED_SIGMAS /
        STAGE_2_DISTILLED_SIGMAS），num_inference_steps / guidance_scale /
        negative_prompt 对该分支无效，仅回填到 metadata 保持返回结构与
        diffusers 分支一致。

        多图条件（核心！T3.1 + T4）：
        - images 列表每项可带 pil（engine.py 加载后缓存）或 path（已有文件）
        - 只有 pil 的图 → 落临时 PNG（runner 的 ImageConditioningInput 只接受路径）
        - image_paths_out 收集所有图的最终路径 + frame_idx + strength
        - 推理结束（含失败）后清理所有临时文件

        Args:
            prompt: 正向提示词
            duration: 时长（秒，num_frames 未显式指定时用于推导帧数）
            fps: 帧率
            width: 宽度（runner 内部对齐 64 整除）
            height: 高度（runner 内部对齐 64 整除）
            seed: 随机种子（-1=随机）
            num_frames: 显式帧数（None=由 duration×fps 对齐 8k+1）
            num_inference_steps: 仅回填 metadata（原生管线固定 sigma 表）
            guidance_scale: 仅回填 metadata（原生蒸馏管线 CFG=1）
            image: 【向后兼容】单张首帧图 PIL.Image
            images: 多图条件列表（新接口，优先于 image）
            auto_duration: 自动时长开关

        Returns:
            dict 含 video_frames / audio_data / metadata

        Raises:
            RuntimeError: 推理失败
        """
        import tempfile

        import torch

        # 帧数：未显式指定时由 duration×fps 对齐 8k+1
        if num_frames is None:
            num_frames = _align_num_frames(duration, fps)

        # 种子（-1 = 随机，与 diffusers 分支一致）
        if seed < 0:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        # === 向后兼容：image（旧签名）→ images（新签名）===
        if images is None and image is not None:
            images = [{"pil": image, "path": None, "role": "reference", "frame_idx": 0, "strength": 1.0}]

        # === 多图 → 临时文件列表 ===
        from pathlib import Path as _P

        temp_files: list[str] = []           # 推理完统一清理
        image_paths_out: list[dict] = []     # 最终传给 native_runner.generate()

        if images:
            for i, img in enumerate(images):
                # 支持 dict 和 schemas.ImageInput（有 .path / .pil / .frame_idx / .strength / .crf）
                if isinstance(img, dict):
                    img_path = img.get("path")
                    img_pil = img.get("pil")
                    frame_idx = img.get("frame_idx", 0)
                    strength = img.get("strength", 1.0)
                    crf = img.get("crf")                    # None = 模型训练默认
                else:
                    img_path = getattr(img, "path", None)
                    img_pil = getattr(img, "pil", None)
                    frame_idx = getattr(img, "frame_idx", 0)
                    strength = getattr(img, "strength", 1.0)
                    crf = getattr(img, "crf", None)

                # 已有有效路径 → 直接用
                if img_path and _P(img_path).exists():
                    entry: dict[str, Any] = {
                        "path": img_path,
                        "frame_idx": int(frame_idx),
                        "strength": float(strength),
                    }
                    if crf is not None:
                        entry["crf"] = crf
                    image_paths_out.append(entry)
                elif img_pil is not None:
                    # 只有 PIL → 落临时文件
                    tmp = tempfile.NamedTemporaryFile(
                        prefix=f"ltx25_cond_{i}_", suffix=".png", delete=False
                    )
                    try:
                        img_pil.save(tmp, format="PNG")
                        tmp_path = tmp.name
                    finally:
                        tmp.close()
                    temp_files.append(tmp_path)
                    entry = {
                        "path": tmp_path,
                        "frame_idx": int(frame_idx),
                        "strength": float(strength),
                    }
                    if crf is not None:
                        entry["crf"] = crf
                    image_paths_out.append(entry)
                else:
                    self._logger.warning(
                        "跳过第 %d 张条件图：无有效 path 也无 pil，img=%s", i, img
                    )

        # 日志
        n_imgs = len(image_paths_out)
        if n_imgs == 0:
            mode_label = "t2v"
        elif n_imgs == 1:
            mode_label = "i2v"
        else:
            f0 = image_paths_out[0]["frame_idx"]
            fN = image_paths_out[-1]["frame_idx"]
            mode_label = "first_last_frame_v" if (f0 == 0 and fN >= num_frames - 2) else "multi_keyframe_v"

        self._logger.info(
            "LTX-2.5 原生推理开始（G8 ltxv blocks）: size=%dx%d, frames=%d, fps=%d, seed=%d, "
            "mode=%s, num_cond_imgs=%d, frame_idxs=%s, auto_duration=%s",
            width, height, num_frames, fps, seed,
            mode_label, n_imgs,
            [ip["frame_idx"] for ip in image_paths_out],
            auto_duration,
        )

        # === 调用原生 runner ===
        infer_start = time.time()
        try:
            result = self._native_runner.generate(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=float(fps),
                output_path=None,
                image_paths=image_paths_out if image_paths_out else None,
                auto_duration=auto_duration,
                temporal_upscalings=temporal_upscalings,
                workflow=workflow,
            )
        except Exception as e:
            self._logger.exception("LTX-2.5 原生推理失败")
            raise RuntimeError(f"LTX-2.5 原生推理失败: {e}") from e
        finally:
            # 所有临时文件清理（推理成功/失败都删）
            for tf in temp_files:
                _P(tf).unlink(missing_ok=True)
        infer_time = time.time() - infer_start

        # 补齐 metadata 字段（与 diffusers 分支返回结构对齐）
        metadata = result.setdefault("metadata", {})
        metadata.setdefault("infer_time", infer_time)
        metadata.setdefault(
            "num_inference_steps",
            num_inference_steps if num_inference_steps is not None else 8,
        )
        metadata.setdefault(
            "guidance_scale",
            guidance_scale if guidance_scale is not None else 1.0,
        )
        metadata.setdefault("is_placeholder", False)

        video_frames = result.get("video_frames")
        self._logger.info(
            "LTX-2.5 原生推理完成: infer_time=%.1fs, frames=%s, size=%s",
            metadata.get("infer_time", infer_time),
            video_frames.shape[0] if video_frames is not None else 0,
            video_frames.shape[1:3] if video_frames is not None else "-",
        )
        return result

    # ===== 卸载 =====

    def unload(self) -> None:
        """卸载模型，释放显存（diffusers pipeline 与 G8 原生 runner）。"""
        if self._pipeline is None and self._native_runner is None:
            self._logger.debug("模型未加载，无需卸载")
            return

        vram_freed = self.get_vram_used()

        # 释放 G8 原生 runner（blocks 分时加载架构，无常驻权重，做兜底清理）
        if self._native_runner is not None:
            try:
                self._native_runner.unload()
            except Exception as e:
                self._logger.warning("卸载 native runner 时出错（继续清理）: %s", e)
            self._native_runner = None

        # 释放 pipeline
        try:
            # 清空 pipeline 内部引用
            components = [
                getattr(self._pipeline, name, None)
                for name in ("transformer", "text_encoder", "vae", "scheduler",
                            "tokenizer", "processor")
            ]
            self._pipeline = None
            del components
        except Exception as e:
            self._logger.warning("卸载 pipeline 时出错（继续清理）: %s", e)

        # Python GC + CUDA 缓存清理
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        self._loaded_variant = None
        self._loaded_scheme_id = None
        self._offload_level = "L1"

        self._logger.info(
            "LTX-2.5 卸载完成: vram_freed≈%.1fMB", vram_freed,
        )
