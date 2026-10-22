"""LTX-2.5 模型文件扫描器（三 variant 版）。

扫描 models/video/ltx25/ 目录，识别可用的 LTX-2.5 模型变体。

三 variant 设计（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.1.6 + M4-T4.2）：
    - diffusers_format/：diffusers 格式实体（Lightricks/LTX-2.5-Diffusers distilled）
      适用于方案 F16（BF16→FP16 cast）/ I8（torchao INT8 运行时量化）
    - gguf_format/：GGUF 格式实体（Abiray/LTX-2.5-Distilled-GGUF）
      适用于方案 G8（GGUF Q8_0）/ G4（GGUF Q4_K_M，M4-T4.2 新增）
      注意：GGUF 档的共享组件（TE/VAE/配置）仍来自 comfyui_format/

v0.2.11+ 方案 B 改造：
    优先调用平台 API（`GET /engines/{id}/variants`）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描（检查标志文件存在性）。

本地 fallback 检查策略（遵循 project-rules.md 第二部分 §5）：
    - 只检查标志文件存在性，不严格校验大小
    - 让 variant 在 UI 中可见（可触发下载），不阻断启动
    - 完整性校验由平台 API（key_files 机制）或 wrapper.load() 兜底

模型目录结构（与 manifest.yaml 一致）：
    models/video/ltx25/
    ├── diffusers_format/                 ← diffusers 格式（方案 F16/I8 用）
    │   ├── model_index.json              ← 标志文件
    │   ├── transformer/                  ← DiT 主模型（BF16 权重分片）
    │   │   └── config.json
    │   ├── text_encoder/                 ← Gemma 4 12B 文本编码器
    │   ├── vae/                          ← 视频 VAE
    │   ├── audio_vae/                    ← 音频 VAE
    │   ├── diffusion_decoder/            ← 扩散视频解码器
    │   ├── scheduler/
    │   ├── processor/
    │   └── tokenizer/
    └── gguf_format/                       ← GGUF 格式（方案 G8 用）
        └── transformer-Q8_0.gguf          ← GGUF 量化权重（共享组件用 diffusers_format/）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import VideoVariantInfo

logger = logging.getLogger(__name__)


# ===== 已知模型变体定义（三 variant：diffusers / GGUF Q8 / GGUF Q4）=====
# subdir 字段为相对于 model_dir（即 video/ltx25/）的子目录。
_KNOWN_VARIANTS: list[dict] = [
    # ============================================================
    # 1. diffusers 格式实体（方案 F16 / I8 用，推荐默认）
    # ============================================================
    {
        "name": "LTX/LTX-2.5-Diffusers",
        "alias": "LTX25-Diffusers",
        "base_model": "LTX-2.5-distilled",
        "precision": "bf16",
        "loader": "diffusers",
        "vram_required": "24GB",  # I8 档最低显存需求（transformer ~19GB）
        "subdir": "diffusers_format",  # 相对 model_dir（video/ltx25/）
        "resolution": "960x544",
        "marker_files": ["model_index.json"],
        "required_files": [
            "diffusers_format/model_index.json",
            "diffusers_format/transformer/config.json",
            "diffusers_format/text_encoder/config.json",
            "diffusers_format/vae/config.json",
        ],
        "supports_audio": True,  # LTX-2.5 原生音视频同步（v1 推理链路未启用）
        "schemes": ["F16", "I8"],  # 适用的量化方案
    },
    # ============================================================
    # 2. GGUF 格式实体（方案 G8 用，ltxv 原生管线）
    # ============================================================
    {
        "name": "LTX/LTX-2.5-GGUF-Q8",
        "alias": "LTX25-GGUF-Q8_0",
        "base_model": "LTX-2.5-distilled",
        "precision": "gguf_q8_0",
        "loader": "gguf",
        "vram_required": "24GB",
        "subdir": "gguf_format",  # 相对 model_dir（video/ltx25/）
        "resolution": "960x544",
        "marker_files": ["transformer-Q8_0.gguf"],
        "required_files": [
            "gguf_format/transformer-Q8_0.gguf",
            # ComfyUI 格式组件（ltxv 原生管线，不依赖 diffusers_format/）
            "comfyui_format/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            "comfyui_format/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
            "comfyui_format/vae/ltx-2.5-audio-vae-bf16.safetensors",
            "comfyui_format/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        ],
        "supports_audio": True,
        "schemes": ["G8"],
    },
    # ============================================================
    # 3. GGUF Q4_K_M 格式实体（方案 G4 用，M4-T4.2 新增）
    # ============================================================
    {
        "name": "LTX/LTX-2.5-GGUF-Q4",
        "alias": "LTX25-GGUF-Q4_K_M",
        "base_model": "LTX-2.5-distilled",
        "precision": "gguf_q4_k_m",
        "loader": "gguf",  # ltxv 原生管线（与 Q8 档同架构，仅权重不同）
        "vram_required": "16GB",
        "subdir": "gguf_format",  # 相对 model_dir（video/ltx25/）
        "resolution": "960x544",
        "marker_files": ["transformer-Q4_K_M.gguf"],
        "required_files": [
            "gguf_format/transformer-Q4_K_M.gguf",
            # ComfyUI 格式组件（与 G8 档共用）
            "comfyui_format/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            "comfyui_format/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
            "comfyui_format/vae/ltx-2.5-audio-vae-bf16.safetensors",
            "comfyui_format/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        ],
        "supports_audio": True,
        "schemes": ["G4"],
    },
]


class ModelScanner:
    """模型目录扫描器（三 variant 版）。

    扫描 model_dir/ 下的 diffusers_format/ 与 gguf_format/ 子目录，
    识别已下载的 LTX-2.5 模型变体（diffusers + GGUF Q8/Q4）。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        engine=None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化扫描器。

        Args:
            models_root: 模型根目录（如 models/），用于 LoRA 扫描相对路径解析
            model_dir: LTX-2.5 模型目录（video/ltx25/，包含 diffusers_format/ 与 gguf_format/ 子目录）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑。
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self.engine = engine
        self._logger = logger or logging.getLogger(__name__)

    async def scan_variants_async(self) -> list[VideoVariantInfo]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。

        调用链：
            1. engine.get_variants() → 平台 API（声明式检测）
            2. 平台返回 None → fallback 到 _scan_locally()
            3. 合并平台 available 字段 + _KNOWN_VARIANTS 引擎特有字段
        """
        if self.engine is not None:
            platform_variants = await self.engine.get_variants()
            if platform_variants is not None:
                return self._merge_with_known(platform_variants)

        # fallback：本地扫描
        return self._scan_locally()

    def scan_variants(self) -> list[VideoVariantInfo]:
        """同步扫描接口（向后兼容）。

        v0.2.11+ 注：若需使用方案 B 平台 API，请调用 scan_variants_async()。
        此同步方法走 fallback 本地扫描逻辑。
        """
        return self._scan_locally()

    def _merge_with_known(self, platform_variants: list[dict]) -> list[VideoVariantInfo]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS 引擎特有字段。

        Args:
            platform_variants: 平台 API 返回的 variant 列表（含 available 字段）

        Returns:
            VideoVariantInfo 列表（available 来自平台，schemes 等来自 _KNOWN_VARIANTS）
        """
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[VideoVariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            available = bool(pv.get("available", False))

            result.append(VideoVariantInfo(
                name=v_name,
                base_model=pv.get("base_model", kv.get("base_model", "")),
                precision=pv.get("precision", kv.get("precision", "bf16")),
                loader=pv.get("loader", kv.get("loader", "diffusers")),
                vram_required=pv.get("vram_required", kv.get("vram_required", "未知")),
                available=available,
                resolution=pv.get("resolution", kv.get("resolution", "960x544")),
                fps=24,
                duration=5,
                max_duration=60,
                supports_audio=kv.get("supports_audio", True),
            ))
            if available:
                self._logger.info("发现模型变体: %s (%s)", v_name, pv.get("alias", ""))

        if not any(pv.get("available") for pv in platform_variants):
            self._logger.warning(
                "未在 %s 中找到任何可用的 LTX-2.5 模型变体，"
                "请到模型管理界面下载（diffusers 仓库：Lightricks/LTX-2.5-Diffusers，"
                "GGUF 仓库：Abiray/LTX-2.5-Distilled-GGUF）",
                self.model_dir,
            )

        return result

    def _scan_locally(self) -> list[VideoVariantInfo]:
        """fallback 本地扫描（原 scan_variants 逻辑）。

        本地 fallback 策略（遵循 project-rules.md 第二部分 §5）：
            - 只检查标志文件（marker_files）存在性
            - 不严格校验大小（避免断点续传残文件被误判可用）
            - 让 variant 在 UI 中可见，触发下载入口
        """
        capabilities: list[VideoVariantInfo] = []

        if not self.model_dir.exists():
            self._logger.warning("模型目录不存在: %s", self.model_dir)
            return capabilities

        # 检查双套目录
        diffusers_dir = self.model_dir / "diffusers_format"
        gguf_dir = self.model_dir / "gguf_format"
        if not diffusers_dir.exists() and not gguf_dir.exists():
            self._logger.warning(
                "未找到 diffusers 目录（%s）或 GGUF 目录（%s），"
                "请到模型管理界面下载",
                diffusers_dir, gguf_dir,
            )
            return capabilities

        for variant_def in _KNOWN_VARIANTS:
            variant_dir = self.model_dir / variant_def["subdir"]

            if not variant_dir.exists():
                self._logger.debug(
                    "变体 %s 目录不存在: %s", variant_def["name"], variant_dir,
                )
                continue

            # 本地 fallback 只检查标志文件（marker_files）
            has_model = self._check_variant_files(variant_def["name"])
            if not has_model:
                self._logger.debug(
                    "变体 %s 标志文件缺失: %s", variant_def["name"], variant_dir,
                )
                continue

            cap = VideoVariantInfo(
                name=variant_def["name"],
                base_model=variant_def["base_model"],
                precision=variant_def["precision"],
                loader=variant_def["loader"],
                vram_required=variant_def["vram_required"],
                available=True,
                resolution=variant_def.get("resolution", "960x544"),
                fps=24,
                duration=5,
                max_duration=60,
                supports_audio=variant_def.get("supports_audio", True),
            )
            capabilities.append(cap)
            self._logger.info(
                "发现模型变体: %s (%s, schemes=%s)",
                cap.name, variant_def["alias"], variant_def.get("schemes", []),
            )

        if not capabilities:
            self._logger.warning(
                "未在 %s 中找到任何 LTX-2.5 模型变体，"
                "请到模型管理界面下载（diffusers 仓库：Lightricks/LTX-2.5-Diffusers，"
                "GGUF 仓库：Abiray/LTX-2.5-Distilled-GGUF）",
                self.model_dir,
            )

        return capabilities

    def _check_variant_files(self, variant_name: str) -> bool:
        """检查 variant 所需的标志文件是否存在。

        本地 fallback 策略：只检查 marker_files，
        不严格校验共享组件（text_encoder / vae）的大小。
        完整性校验由平台 API 或 wrapper.load() 兜底。

        Args:
            variant_name: variant 名称（如 LTX/LTX-2.5-Diffusers）

        Returns:
            标志文件是否齐全
        """
        variant_def = None
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                variant_def = v
                break
        if variant_def is None:
            return False

        variant_dir = self.model_dir / variant_def["subdir"]
        for marker in variant_def.get("marker_files", []):
            if not (variant_dir / marker).exists():
                return False
        return True

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析模型路径。

        Args:
            variant_name: 变体名（如 LTX/LTX-2.5-Diffusers）

        Returns:
            模型目录路径（如 models/video/ltx25/diffusers_format），
            不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                path = self.model_dir / variant_def["subdir"]
                return path if path.exists() else None
        return None

    def get_variant_definition(self, variant_name: str) -> Optional[dict]:
        """获取 variant 的完整定义（含 required_files 清单 + schemes）。

        供 wrapper.load() 用于文件完整性校验，以及 SchemeManager 用于方案匹配。

        Args:
            variant_name: variant 名称

        Returns:
            variant 定义字典（含 schemes 字段），不存在返回 None
        """
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                return v
        return None

    def get_variants_by_scheme(self, scheme_id: str) -> list[dict]:
        """按方案 ID 查询适用的 variant 定义列表。

        Args:
            scheme_id: 方案 ID（F16 / I8 / G8）

        Returns:
            适用的 variant 定义列表（_KNOWN_VARIANTS 中的字典）
        """
        return [
            v for v in _KNOWN_VARIANTS
            if scheme_id in v.get("schemes", [])
        ]

    def scan_loras(self) -> list[str]:
        """扫描 LoRA 目录下的 .safetensors 文件名。

        LTX-2.5 v1 不支持 LoRA（分析文档非目标项），预留接口返回空列表。

        Returns:
            LoRA 名称列表（文件 stem，不含扩展名）
        """
        # v1 预留：LTX-2.5 LoRA 支持为后续版本扩展项
        return []
