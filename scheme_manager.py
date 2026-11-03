"""LTX-2.5 多方案管理器。

加载与管理引擎的三个量化方案（F16 / I8 / G8），提供：
    - 方案配置加载（从 schemes/scheme_F16.yaml 等）
    - 按显存自动选档（auto_select_scheme）
    - 方案合法性校验（适用 variant 检查）
    - 方案列表查询（供 UI / 平台 API 调用）

方案与 variant 的关系（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.2）：
    - 方案 F16 / I8：使用 diffusers 格式实体（Lightricks/LTX-2.5-Diffusers 权重）
    - 方案 G8：使用 GGUF 格式实体（Abiray/LTX-2.5-Distilled-GGUF 权重）
    - GGUF 档的共享组件（TE/VAE/配置）仍来自 diffusers 仓库（双路径特例，见 §7.1.6）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# 任务 71：设备探测统一走 SDK（平台与引擎同一份事实源）
from aireliable_engine.device import device_info as _sdk_device_info


# ===== 默认方案选择阈值（GB）=====
# 按显存自动选档策略（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.1.6 + M4-T4.2）：
#   ≥48GB → 方案 I8（INT8 均衡档，48G 卡余量充足，质量近乎无损）
#   24-48GB → 方案 G8（GGUF Q8_0 档）
#   16-24GB → 方案 G4（GGUF Q4_K_M 低配档，M4-T4.2 引入）
#   <16GB → 仍选 G4（最低档）
#
# 注意阈值容差：Quadro RTX 8000（名义 48GB）经 torch/nvidia-smi 报告
# 为 47.99GiB（ECC 保留），若阈值取 48.0 会因 47.99 < 48.0 误降级到
# G8 档，故 I8 阈值下调 1GB 容差。
_AUTO_SELECT_THRESHOLDS = {
    "I8": 47.0,
    "G8": 24.0,
    "G4": 16.0,
}


@dataclass
class QuantizationConfig:
    """量化方案配置（LTX-2.5 特有，H3 无此结构）。

    四档取值：
        - type=none + compute_dtype=float16 → F16 档（BF16 权重 cast 为 FP16）
        - type=torchao_int8 + compute_dtype=float16 → I8 档（TorchAoConfig）
        - type=gguf_q8 + compute_dtype=float16 → G8 档（Q8_0 权重，ltxv 原生管线）
        - type=gguf_q4 + compute_dtype=float16 → G4 档（Q4_K_M 权重，M4-T4.2）
    """
    type: str = "none"              # none / torchao_int8 / gguf_q8 / gguf_q4
    compute_dtype: str = "float16"  # 计算精度（Turing 不支持 BF16，固定 fp16）


@dataclass
class BlockSwapConfig:
    """BlockSwap 显存优化配置（低显存档用，v1 预留）。"""
    enabled: bool = False
    swap_blocks: int = 20       # 把 N 个 transformer 块卸载到 CPU
    swap_strategy: str = "even"  # 均匀分布


@dataclass
class VAETilingConfig:
    """VAE tiling 显存优化配置。"""
    enabled: bool = False
    tile_size: int = 256


@dataclass
class InferenceConfig:
    """推理参数配置。

    LTX-2.5 distilled 约束：
        - num_inference_steps 固定 8（蒸馏版）
        - guidance_scale 固定 1.0（CFG=1）
        - 分辨率需被 32 整除
    """
    num_inference_steps: int = 8
    scheduler: str = "flow_match"
    sampler: str = "euler"
    guidance_scale: float = 1.0
    default_resolution: str = "960x544"
    default_fps: int = 24
    default_duration: int = 5


@dataclass
class SchemeConfig:
    """单方案配置（对应 schemes/scheme_F16.yaml 等）。

    Attributes:
        scheme: 方案 ID（F16 / I8 / G8）
        name: 方案中文名
        description: 方案描述
        vram_required: 显存需求标签（如 "48GB" / "24GB"）
        model_repo: 模型 HuggingFace 仓库 ID
        model_subdir: 模型本地子目录（相对 model_dir，即 video/ltx25/）
        precision: 精度标签（fp16 / int8_torchao / gguf_q8_0）
        size_gb: 模型大小（GB，用于下载提示）
        offload_level: CPU offload 级别（L0 / L1 / L2 / L3）
        device: 推理设备（cuda:0 / cpu）
        quantization: 量化配置（LTX-2.5 特有）
        block_swap: BlockSwap 配置
        vae_tiling: VAE tiling 配置
        inference: 推理参数
        applicable_variants: 适用 variant 名称列表
    """
    scheme: str
    name: str
    description: str
    vram_required: str
    model_repo: str
    model_subdir: str
    precision: str
    size_gb: int
    offload_level: str
    device: str = "cuda:0"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    block_swap: BlockSwapConfig = field(default_factory=BlockSwapConfig)
    vae_tiling: VAETilingConfig = field(default_factory=VAETilingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    applicable_variants: list[str] = field(default_factory=list)


class SchemeManager:
    """方案管理器。

    负责：
        - 从 schemes/ 目录加载所有方案 yaml
        - 按显存自动选档
        - 方案合法性校验
        - 提供方案列表给 UI / 平台 API

    用法：
        manager = SchemeManager(schemes_dir=Path("engines/ltx25_video_b9c4d2e1/schemes"))
        scheme = manager.auto_select_scheme(vram_gb=48.0)
        all_schemes = manager.list_schemes()
    """

    def __init__(
        self,
        schemes_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化方案管理器。

        Args:
            schemes_dir: schemes/ 目录路径（包含 scheme_F16.yaml 等）
            logger: 可选的日志器
        """
        self.schemes_dir = schemes_dir
        self._logger = logger or logging.getLogger(__name__)
        self._cache: dict[str, SchemeConfig] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """懒加载所有方案 yaml 到缓存。"""
        if self._loaded:
            return

        if not self.schemes_dir.exists():
            self._logger.warning("schemes 目录不存在: %s", self.schemes_dir)
            self._loaded = True
            return

        for yaml_file in sorted(self.schemes_dir.glob("scheme_*.yaml")):
            try:
                scheme = self._load_yaml(yaml_file)
                if scheme is not None:
                    self._cache[scheme.scheme] = scheme
                    self._logger.info(
                        "加载方案 %s: %s（vram_required=%s, precision=%s）",
                        scheme.scheme, scheme.name, scheme.vram_required, scheme.precision,
                    )
            except Exception as e:
                self._logger.exception("加载方案文件失败: %s (%s)", yaml_file, e)

        self._loaded = True

        if not self._cache:
            self._logger.warning("未在 %s 中找到任何方案文件", self.schemes_dir)

    def _load_yaml(self, yaml_path: Path) -> Optional[SchemeConfig]:
        """解析单个 scheme_X.yaml 为 SchemeConfig。

        Args:
            yaml_path: yaml 文件路径

        Returns:
            SchemeConfig 实例，解析失败返回 None
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        # 解析 quantization 块（LTX-2.5 特有）
        quant_data = data.get("quantization", {}) or {}
        quantization = QuantizationConfig(
            type=str(quant_data.get("type", "none")),
            compute_dtype=str(quant_data.get("compute_dtype", "float16")),
        )

        # 解析 block_swap 块
        bs_data = data.get("block_swap", {}) or {}
        block_swap = BlockSwapConfig(
            enabled=bool(bs_data.get("enabled", False)),
            swap_blocks=int(bs_data.get("swap_blocks", 20)),
            swap_strategy=str(bs_data.get("swap_strategy", "even")),
        )

        # 解析 vae_tiling 块
        vt_data = data.get("vae_tiling", {}) or {}
        vae_tiling = VAETilingConfig(
            enabled=bool(vt_data.get("enabled", False)),
            tile_size=int(vt_data.get("tile_size", 256)),
        )

        # 解析 inference 块
        inf_data = data.get("inference", {}) or {}
        inference = InferenceConfig(
            num_inference_steps=int(inf_data.get("num_inference_steps", 8)),
            scheduler=str(inf_data.get("scheduler", "flow_match")),
            sampler=str(inf_data.get("sampler", "euler")),
            guidance_scale=float(inf_data.get("guidance_scale", 1.0)),
            default_resolution=str(inf_data.get("default_resolution", "960x544")),
            default_fps=int(inf_data.get("default_fps", 24)),
            default_duration=int(inf_data.get("default_duration", 5)),
        )

        return SchemeConfig(
            scheme=str(data["scheme"]),
            name=str(data["name"]),
            description=str(data["description"]),
            vram_required=str(data["vram_required"]),
            model_repo=str(data["model_repo"]),
            model_subdir=str(data["model_subdir"]),
            precision=str(data["precision"]),
            size_gb=int(data["size_gb"]),
            offload_level=str(data["offload_level"]),
            device=str(data.get("device", "cuda:0")),
            quantization=quantization,
            block_swap=block_swap,
            vae_tiling=vae_tiling,
            inference=inference,
            applicable_variants=list(data.get("applicable_variants", [])),
        )

    def load_scheme(self, scheme_id: str) -> Optional[SchemeConfig]:
        """按 ID 加载单个方案。

        Args:
            scheme_id: 方案 ID（F16 / I8 / G8）

        Returns:
            SchemeConfig，不存在返回 None
        """
        self._ensure_loaded()
        return self._cache.get(scheme_id)

    def list_schemes(self) -> list[SchemeConfig]:
        """列出所有方案（按 ID 字母序）。"""
        self._ensure_loaded()
        return [self._cache[k] for k in sorted(self._cache.keys())]

    def auto_select_scheme(self, vram_gb: Optional[float] = None) -> Optional[SchemeConfig]:
        """按 GPU 显存自动选择方案。

        策略（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.1.6）：
            ≥48GB → 方案 I8（INT8 均衡档，推荐默认）
            ≥24GB → 方案 G8（GGUF Q8_0 档）
            <24GB → 仍选 G8（最低档兜底）

        Args:
            vram_gb: GPU 显存（GB），None 时自动检测

        Returns:
            推荐的 SchemeConfig，无可用方案返回 None
        """
        self._ensure_loaded()

        if vram_gb is None:
            vram_gb = _detect_gpu_vram_gb()

        self._logger.info("自动选档: 检测到 GPU 显存=%.1fGB", vram_gb)

        # 按阈值降序选择
        for scheme_id in ["I8", "G8", "G4"]:
            threshold = _AUTO_SELECT_THRESHOLDS.get(scheme_id, 0.0)
            if vram_gb >= threshold and scheme_id in self._cache:
                selected = self._cache[scheme_id]
                self._logger.info(
                    "自动选档结果: 方案 %s（%s），阈值=%.0fGB，实际=%.1fGB",
                    selected.scheme, selected.name, threshold, vram_gb,
                )
                return selected

        # 都不满足，返回最低档（G4）或第一个可用方案
        fallback = self._cache.get("G4") or (
            self.list_schemes()[0] if self.list_schemes() else None
        )
        if fallback:
            self._logger.warning(
                "显存 %.1fGB 不满足任何方案阈值，降级使用方案 %s",
                vram_gb, fallback.scheme,
            )
        return fallback

    def validate_scheme_for_variant(
        self,
        scheme: SchemeConfig,
        variant_name: str,
    ) -> bool:
        """校验方案是否适用于指定 variant。

        方案与 variant 必须匹配：
            - 方案 F16 / I8 适用于 diffusers 格式 variant（Lightricks/LTX-2.5-Distilled）
            - 方案 G8 适用于 GGUF 格式 variant（Abiray/LTX-2.5-Distilled-GGUF）

        Args:
            scheme: 方案配置
            variant_name: variant 名称

        Returns:
            是否适用
        """
        if not scheme.applicable_variants:
            # 未声明 applicable_variants 时不做限制
            return True

        applicable = variant_name in scheme.applicable_variants
        if not applicable:
            self._logger.warning(
                "方案 %s 不适用于 variant %s（适用列表: %s）",
                scheme.scheme, variant_name, scheme.applicable_variants,
            )
        return applicable

    def get_scheme_for_variant(self, variant_name: str) -> Optional[SchemeConfig]:
        """根据 variant 名称反查适用的方案。

        优先级：
            1. 如果 variant 名含 GGUF-Q4 / Q4_K_M，返回方案 G4
            2. 如果 variant 名含 GGUF，返回方案 G8
            3. 否则（diffusers 格式）返回方案 I8
            4. 都不存在时按显存自动选档

        Args:
            variant_name: variant 名称

        Returns:
            推荐的 SchemeConfig
        """
        self._ensure_loaded()

        name_upper = variant_name.upper()
        if "GGUF" in name_upper:
            # GGUF Q4 variant → 方案 G4；其余 GGUF variant → 方案 G8
            if "Q4" in name_upper or "q4" in variant_name:
                return self._cache.get("G4") or self.auto_select_scheme()
            return self._cache.get("G8") or self.auto_select_scheme()
        else:
            # diffusers variant → 方案 I8（推荐默认档）
            return self._cache.get("I8") or self.auto_select_scheme()


def _detect_gpu_vram_gb() -> float:
    """检测当前 GPU 显存（GB），无 GPU 或检测失败返回 0.0。

    任务 71：委托 SDK ``device.device_info``（与 config.py 同一数据源）。
    """
    try:
        return _sdk_device_info(mode="auto").total_vram_gb or 0.0
    except Exception:
        return 0.0
