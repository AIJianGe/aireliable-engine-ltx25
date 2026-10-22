"""DFR (Diffusion Fidelity Rendering) Pipeline 适配器。

封装 ltx_pipelines.dfr_pipeline.DFRPipeline，提供：
- DFR 所需模型就绪检测（detailing_lora 强制，其他可选）
- ModelPaths 从本地文件布局构建
- 与 LTXVNativeRunner.generate() 兼容的输入/输出格式
- 显存不足或模型缺失时的自动降级（返回 ``None`` 让调用方回退 P0 blocks 路径）

DFRPipeline vs P0 blocks 路径的核心差异：
| 特性                | P0 原生 blocks（已实现）     | DFRPipeline（本文件）               |
| ------------------- | ---------------------------- | ----------------------------------- |
| transformer 格式    | GGUF Q8/Q4（反量化）         | 原生 safetensors（FP16/BF16）       |
| 加载架构            | 分时加载（用完释放）          | 全量加载（~40GB 峰值显存）          |
| temporal upsample   | ❌ 不支持                    | ✅ 多轮 x2 tile 分割补帧（0/1/2）   |
| 关键帧 carry-forward| ❌ 不支持                    | ✅ 前后轮关键帧传递                 |
| 多图条件            | ✅ VideoConditionByLatent/Key | ✅ 同 + VideoGeneratedKeyframeSlots |
| detailing_lora      | ❌ 不使用                    | ✅ Stage 2 IC-LoRA 精修             |
| 显存需求            | ~24GB（G8）/ ~16GB（G4）     | ~48GB+（需要原生 transformer）     |

触发条件（generate() 入口路由）：
  当 temporal_upscalings > 0 或 workflow == "multi_reference_v" 时，
  优先检测 DFR 模型就绪 → 走本适配器；否则自动降级 P0。
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# --- DFR 所需的最小文件集合 ---
# detailing_lora 是 DFRPipeline.__init__ 的强制依赖（源码 L422-423 直接 raise ValueError），
# 缺失时无法构建 DFRPipeline 实例 → 适配器返回 None 让调用方降级。
# 原生 transformer（safetensors 格式）也必需（GGUF 不兼容 DFRPipeline），
# 但路径从 diffusers_format 的 marker key_files 推导，这里只做可选存在性检测。
_DFR_OPTIONAL_FILES = {
    "detailing_lora": "ltx-2.5-detailing-lora.safetensors",
    "temporal_upsampler": "ltx-2.5-temporal-upsampler-x2-bf16.safetensors",
    "duration_head": "ltx-2.5-duration-head-bf16.safetensors",
}
_DFR_MIN_FILE_BYTES = 1024 * 1024  # 1MB


class DFRAdapter:
    """DFRPipeline 轻量适配器。

    只做路径推导 + 就绪检测 + 调用转发；**不在 __init__ 加载任何模型**
    （遵循 blocks 分时加载架构——模型在 ``generate_dfr()`` 调用内部构建，
    用完即释放，避免长期持有显存）。

    使用方式::

        adapter = DFRAdapter(model_dir, logger)
        if adapter.is_available(temporal_upscalings=2):
            result = adapter.generate_dfr(...)
        else:
            # 回退 P0 blocks 路径
            result = runner.generate(...)
    """

    # 原生 transformer（safetensors 分片）目录 — DFRPipeline 必需
    _NATIVE_TRANSFORMER_DIR = Path("diffusers_format") / "transformer"
    # 共享组件路径（与 LTXVNativeRunner 共用 ComfyUI 格式）
    _GEMMA_RELPATH = Path("comfyui_format") / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    _VIDEO_VAE_RELPATH = Path("comfyui_format") / "vae" / "ltx-2.5-video-vae-conv-bf16.safetensors"
    _AUDIO_VAE_RELPATH = Path("comfyui_format") / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    _UPSAMPLER_RELPATH = Path("comfyui_format") / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    # DFR 可选组件目录（manifest model_files target_dir）
    _DFR_RELPATH = Path("dfr_format")

    def __init__(
        self,
        model_dir: Path,
        device: str,
        dtype: torch.dtype = torch.float16,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self._logger = logger or logging.getLogger(__name__)

        # 推导全部候选路径（只做拼接，不校验）
        self._native_transformer_dir = self.model_dir / self._NATIVE_TRANSFORMER_DIR
        self._gemma_path = self.model_dir / self._GEMMA_RELPATH
        self._video_vae_path = self.model_dir / self._VIDEO_VAE_RELPATH
        self._audio_vae_path = self.model_dir / self._AUDIO_VAE_RELPATH
        self._upsampler_path = self.model_dir / self._UPSAMPLER_RELPATH
        self._dfr_dir = self.model_dir / self._DFR_RELPATH

    # ===== 就绪检测 =====

    def _check(self, p: Path, label: str) -> bool:
        exists = p.exists()
        if exists and p.is_file():
            ok = p.stat().st_size > _DFR_MIN_FILE_BYTES
            if not ok:
                self._logger.warning("DFR 文件残缺：%s（%s, size=%d）", label, p, p.stat().st_size)
            return ok
        if exists and p.is_dir():
            # transformer 分片目录 — 有 .safetensors 文件即可
            safes = list(p.glob("*.safetensors"))
            ok = len(safes) > 0
            if not ok:
                self._logger.warning("DFR 目录空：%s（%s）", label, p)
            return ok
        return False

    def is_available(self, temporal_upscalings: int = 0) -> bool:
        """检测 DFRPipeline 所需模型是否就绪。

        最低要求：
          1. 原生 transformer（safetensors 分片，diffusers_format/transformer/）
          2. detailing_lora（强制，DFRPipeline.__init__ 会 raise）
          3. spatial_upsampler（comfyui_format 下已有的共享组件）
          4. Gemma TE + 双 VAE（comfyui_format 下已有的共享组件）

        可选：
          - temporal_upscalings > 0 时需要 temporal_upsampler
          - duration_head（auto_duration 时需要）
        """
        missing: list[str] = []

        # --- 必需 ---
        if not self._check(self._native_transformer_dir, "原生 transformer (safetensors)"):
            missing.append(f"原生 transformer（{self._native_transformer_dir}）— 需下载 F16/I8 档")
        if not self._check(self._dfr_dir / _DFR_OPTIONAL_FILES["detailing_lora"], "detailing_lora"):
            missing.append(f"detailing_lora（{_DFR_OPTIONAL_FILES['detailing_lora']}）")
        if not self._check(self._upsampler_path, "spatial_upsampler"):
            missing.append(f"spatial_upsampler（共享组件，需下载 ComfyUI 格式）")
        if not self._check(self._gemma_path, "Gemma TE"):
            missing.append(f"Gemma TE（共享组件，需下载 ComfyUI 格式）")
        if not self._check(self._video_vae_path, "Video VAE"):
            missing.append(f"Video VAE（共享组件，需下载 ComfyUI 格式）")
        if not self._check(self._audio_vae_path, "Audio VAE"):
            missing.append(f"Audio VAE（共享组件，需下载 ComfyUI 格式）")

        # --- temporal_upscalings 触发 ---
        if temporal_upscalings > 0 and temporal_upscalings <= 2:
            if not self._check(self._dfr_dir / _DFR_OPTIONAL_FILES["temporal_upsampler"], "temporal_upsampler"):
                missing.append(
                    f"temporal_upsampler（temporal_upscalings={temporal_upscalings} 需要）— "
                    f"{_DFR_OPTIONAL_FILES['temporal_upsampler']}"
                )
        elif temporal_upscalings > 2:
            self._logger.warning(
                "temporal_upscalings=%d 超出 DFRPipeline 支持范围（仅 0/1/2），忽略",
                temporal_upscalings,
            )
            temporal_upscalings = 0

        if missing:
            self._logger.info(
                "DFRPipeline 不可用（缺 %d 个组件）：\n  %s\n回退 P0 原生 blocks 路径",
                len(missing), "\n  ".join(missing),
            )
            return False

        self._logger.info(
            "DFRPipeline 就绪 ✅：原生 transformer + detailing_lora + spatial_upsampler + "
            "双 VAE + Gemma TE + temporal_upsampler（可选，当 temporal_upscalings>0 时使用）"
        )
        return True

    # ===== 显存预检 =====

    def _check_vram(self, headroom_gb: float = 4.0) -> bool:
        """显存预检：DFRPipeline 全量加载需要 ≥ 44GB（BF16 原生 transformer）。

        RTX 4090D 24GB / RTX 5090D 32GB / A6000 48GB / RTX 8000 48GB。
        显存不足时返回 False，让调用方降级。
        """
        if not torch.cuda.is_available():
            self._logger.warning("CUDA 不可用，DFR 路径跳过")
            return False
        total, free = torch.cuda.mem_get_info()
        total_gb = total / 1e9
        free_gb = free / 1e9
        # 原生 BF16 transformer ~35GB + TE ~24GB + 其他 ~5GB ≈ 64GB 峰值（blocks 分时加载可以降到 ~48GB）
        # 保守判定：总显存 ≥ 48GB 且有 ≥ 20GB 空闲
        ok = total_gb >= 48.0 and free_gb >= 20.0
        self._logger.info(
            "DFR 显存预检：total=%.1fGB, free=%.1fGB → %s",
            total_gb, free_gb, "✅ 通过" if ok else "❌ 不足（需 ≥48GB，建议 Quadro RTX 8000 / A6000）",
        )
        return ok

    # ===== generate =====

    @torch.inference_mode()
    def generate_dfr(  # noqa: PLR0912, PLR0913, PLR0915 — DFRPipeline.__call__ 签名较长
        self,
        prompt: str,
        image_paths: list[dict],
        seed: int = 42,
        width: int = 512,
        height: int = 256,
        num_frames: int = 33,
        fps: float = 24.0,
        temporal_upscalings: int = 0,
        spatial_upscalings: int = 1,
        auto_duration: bool = False,
        output_path: Optional[str] = None,
    ) -> Optional[dict]:
        """执行 DFRPipeline 推理。

        返回与 LTXVNativeRunner.generate() 相同格式的 dict：
            {
                "video_frames": (T, H, W, C) uint8 numpy,
                "audio_data": (channels, samples) float32 numpy 或 None,
                "metadata": {...},
            }

        若模型就绪检测失败，返回 ``None``（调用方回退 P0）。
        """
        # --- 前置检查 ---
        if not self.is_available(temporal_upscalings=temporal_upscalings):
            return None
        if not self._check_vram():
            self._logger.warning("显存不足，DFR 路径跳过，回退 P0")
            return None

        from ltx_pipelines.dfr_pipeline import DFRPipeline
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps

        self._logger.info(
            "=== DFRPipeline 启动 ===\n"
            "  prompt=%s\n  images=%d 张\n  frame_idx=%s\n  temporal_upscalings=%d\n  spatial_upscalings=%d",
            prompt, len(image_paths),
            [ip["frame_idx"] for ip in image_paths],
            temporal_upscalings, spatial_upscalings,
        )

        # --- ModelPaths ---
        # transformer: safetensors 分片目录 — ModelPaths.from_split 需要 str
        # DFRPipeline 内部用 ModelPaths.transformer() 访问（返回 str）
        # SingleGPUModelBuilder 可以接受目录路径（会 glob *.safetensors）
        transformer_path = str(self._native_transformer_dir)
        detailing_path = str(self._dfr_dir / _DFR_OPTIONAL_FILES["detailing_lora"])
        temporal_up_path = (
            str(self._dfr_dir / _DFR_OPTIONAL_FILES["temporal_upsampler"])
            if temporal_upscalings > 0 and self._check(self._dfr_dir / _DFR_OPTIONAL_FILES["temporal_upsampler"], "temporal_upsampler")
            else None
        )
        duration_head_path = (
            str(self._dfr_dir / _DFR_OPTIONAL_FILES["duration_head"])
            if self._check(self._dfr_dir / _DFR_OPTIONAL_FILES["duration_head"], "duration_head")
            else None
        )

        model_paths = ModelPaths.from_split(
            transformer_path=transformer_path,
            text_encoder_path=str(self._gemma_path),
            video_vae_path=str(self._video_vae_path),
            audio_vae_path=str(self._audio_vae_path),
            duration_head_path=duration_head_path,  # None = DFRPipeline duration_predictor 返回 None
        )

        # --- detailing_lora（强制，DFRPipeline.__init__ 会检查非空）---
        detailing_lora = [LoraPathStrengthAndSDOps(detailing_path, strength=0.5, sd_ops=None)]

        # --- 构建 DFRPipeline（全量加载，~40GB+）---
        infer_start = time.time()
        try:
            pipeline = DFRPipeline(
                model_paths=model_paths,
                spatial_upsampler_path=str(self._upsampler_path),
                loras=(),                      # 用户 LoRA（无）
                detailing_lora=detailing_lora, # 强制，DFRStage 2 用
                temporal_upsampler_path=temporal_up_path,  # None = 不做 temporal upsample
                device=self.device,
                diffvae_optimization="CHUNKED_EAGER",
            )
            self._logger.info("DFRPipeline 构建完成（%.1fs）", time.time() - infer_start)
        except Exception as e:
            self._logger.error("DFRPipeline 构建失败，回退 P0：%s", e, exc_info=True)
            self._cleanup()
            return None

        # --- ImageConditioningInput 列表 ---
        images = [
            ImageConditioningInput(
                path=str(ip["path"]),
                frame_idx=int(ip.get("frame_idx", 0)),
                strength=float(ip.get("strength", 1.0)),
                crf=int(ip["crf"]) if ip.get("crf") is not None else None,
            )
            for ip in image_paths
        ]

        # --- 推理 ---
        self._logger.info("DFRPipeline 推理开始...")
        try:
            # 注意：DFRPipeline.__call__ 不接受 output_path，需要调用方自己 encode_video
            output = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                frame_rate=fps,
                images=images,
                num_frames=num_frames,           # 具体帧数（None = auto_duration）
                temporal_upscalings=temporal_upscalings,
                spatial_upscalings=spatial_upscalings,
            )
        except Exception as e:
            self._logger.error("DFRPipeline 推理失败，回退 P0：%s", e, exc_info=True)
            self._cleanup()
            return None

        infer_time = time.time() - infer_start

        # --- 提取结果（对齐 LTXVNativeRunner.generate() 的返回格式）---
        # output.video: PipelineOutput.video = (T, H, W, C) float32 [0, 1] tensor
        # output.audio: PipelineOutput.audio = Audio (waveform + sampling_rate)
        decoded_video = output.video.detach().float().cpu()  # (T, H, W, C)
        video_frames = (decoded_video.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)

        audio_data: Optional[np.ndarray] = None
        audio_sample_rate = 48000
        try:
            if output.audio is not None:
                audio_data = output.audio.waveform.detach().cpu().float().numpy()
                audio_sample_rate = int(output.audio.sampling_rate)
        except Exception as e:
            self._logger.warning("DFR 音频提取失败，降级无音轨：%s", e)

        # --- mode 精细分类 ---
        n_imgs = len(image_paths)
        if n_imgs == 1:
            mode_label = "i2v"
        elif n_imgs == 2:
            mode_label = "first_last_frame_v"
        else:
            mode_label = "multi_keyframe_v"
        if temporal_upscalings > 0:
            mode_label += f"+temporal_x{2**temporal_upscalings}"

        num_out_frames = output.num_frames
        playback_fps = fps * (2 ** temporal_upscalings)

        # --- 可选 mp4 导出 ---
        if output_path is not None:
            try:
                from ltx_pipelines.utils.media_io import encode_video
                from ltx_core.model.video_vae import get_video_chunks_number
                encode_video(
                    video=output.video,
                    fps=int(playback_fps),
                    audio=output.audio,
                    output_path=output_path,
                    video_chunks_number=get_video_chunks_number(num_out_frames, output.tiling_config),
                )
                self._logger.info("DFR 输出完成：%s", output_path)
            except Exception as e:
                self._logger.warning("DFR mp4 导出失败（不影响帧返回）：%s", e)

        self._logger.info(
            "DFRPipeline 推理完成（总耗时 %.1fs）：%d 帧 @ %.1ffps, frames=%s",
            infer_time, num_out_frames, playback_fps, video_frames.shape,
        )

        # --- 清理（释放 pipeline 持有的全部模型引用）---
        del pipeline, output, decoded_video
        self._cleanup()

        return {
            "video_frames": video_frames,
            "audio_data": audio_data,
            "metadata": {
                "width": width,
                "height": height,
                "num_frames": num_out_frames,
                "fps": playback_fps,
                "seed": seed,
                "infer_time": infer_time,
                "mode": mode_label,
                "num_conditioning_images": n_imgs,
                "conditioning_frame_idxs": [ip["frame_idx"] for ip in image_paths],
                "audio_sample_rate": audio_sample_rate if audio_data is not None else 0,
                "has_audio": audio_data is not None,
                "auto_duration": False,
                "pipeline": "DFR",
                "temporal_upscalings": temporal_upscalings,
                "spatial_upscalings": spatial_upscalings,
            },
        }

    # ===== 卸载 =====

    def _cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._logger.info(
                "DFR VRAM: %.1f GB free / %.1f GB",
                torch.cuda.mem_get_info()[0] / 1e9, torch.cuda.mem_get_info()[1] / 1e9,
            )
