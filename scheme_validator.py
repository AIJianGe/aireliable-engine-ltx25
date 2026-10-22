"""方案合法性校验器。

在切换方案前校验：
    - 方案与 variant 是否匹配（diffusers variant 不能用 G8 档，GGUF variant 只能用 G8 档）
    - 显存是否满足方案要求
    - 模型文件是否完整（标志文件检查）

LTX-2.5 特有（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.1.6）：
    G8 档的路径特例：scheme.model_subdir 指向 gguf/，但共享组件
    （TE/VAE/config）在 diffusers/ —— 校验时需按双目录分别检查标志文件。

返回 ValidationResult，含通过/失败 + 失败原因列表。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .scheme_manager import SchemeConfig, SchemeManager


@dataclass
class ValidationResult:
    """方案校验结果。"""
    ok: bool
    scheme_id: str
    variant_name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SchemeValidator:
    """方案校验器。

    用法：
        validator = SchemeValidator(scheme_manager, models_root=Path("models"))
        result = validator.validate(scheme_id="I8", variant_name="Lightricks/LTX-2.5-Distilled")
        if not result.ok:
            print(result.errors)
    """

    def __init__(
        self,
        scheme_manager: SchemeManager,
        models_root: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化校验器。

        Args:
            scheme_manager: 方案管理器实例
            models_root: 模型根目录（如 models/）
            logger: 可选的日志器
        """
        self.scheme_manager = scheme_manager
        self.models_root = models_root
        self._logger = logger or logging.getLogger(__name__)

    def validate(
        self,
        scheme_id: str,
        variant_name: str,
        vram_gb: Optional[float] = None,
    ) -> ValidationResult:
        """校验方案是否适用于指定 variant。

        校验项：
            1. 方案是否存在
            2. 方案与 variant 是否匹配（通过 SchemeManager.validate_scheme_for_variant）
            3. 显存是否满足方案要求
            4. 模型文件是否存在（按 scheme.model_subdir 检查标志文件；
               G8 档额外校验 diffusers/ 共享组件目录）

        Args:
            scheme_id: 方案 ID（F16 / I8 / G8）
            variant_name: variant 名称
            vram_gb: GPU 显存（GB），None 时自动检测

        Returns:
            ValidationResult
        """
        result = ValidationResult(ok=True, scheme_id=scheme_id, variant_name=variant_name)

        # 1. 方案是否存在
        scheme = self.scheme_manager.load_scheme(scheme_id)
        if scheme is None:
            result.ok = False
            result.errors.append(f"方案 {scheme_id} 不存在")
            return result

        # 2. 方案与 variant 是否匹配
        if not self.scheme_manager.validate_scheme_for_variant(scheme, variant_name):
            result.ok = False
            result.errors.append(
                f"方案 {scheme_id}（precision={scheme.precision}）不适用于 variant {variant_name}。"
                f"方案 {scheme_id} 适用：{scheme.applicable_variants}"
            )

        # 3. 显存检查
        if vram_gb is None:
            from .scheme_manager import _detect_gpu_vram_gb
            vram_gb = _detect_gpu_vram_gb()

        # 解析方案显存需求（如 "24GB" → 24.0，"48GB+" → 48.0）
        required_vram = self._parse_vram_required(scheme.vram_required)
        if required_vram is not None and vram_gb < required_vram:
            result.warnings.append(
                f"显存不足：方案 {scheme_id} 需要 {scheme.vram_required}，实际检测到 {vram_gb:.1f}GB。"
                f"可能影响性能或导致 OOM。"
            )

        # 4. 模型文件存在性检查（只检查标志文件，不校验大小）
        # scheme.model_subdir 是相对 model_dir（video/ltx25/）的子目录
        from .config import MODEL_SUBDIR
        model_dir = self.models_root / MODEL_SUBDIR / scheme.model_subdir
        if not model_dir.exists():
            result.ok = False
            result.errors.append(
                f"模型目录不存在：{model_dir}\n"
                f"请先下载方案 {scheme_id} 对应的模型（仓库：{scheme.model_repo}）"
            )
        else:
            self._check_marker_files(scheme, model_dir, result)

        # 5. G8/G4 档 ltxv 原生管线：ComfyUI 格式组件（Gemma TE/双 VAE/上采样器）
        #    在 comfyui_format/ 目录（manifest.yaml ltx25-comfyui-components），
        #    不依赖 diffusers_format/ 共享组件
        if scheme.quantization.type in ("gguf_q8", "gguf_q4"):
            comfyui_dir = self.models_root / MODEL_SUBDIR / "comfyui_format"
            if not comfyui_dir.exists():
                result.ok = False
                result.errors.append(
                    f"GGUF 档依赖的 ComfyUI 组件目录不存在：{comfyui_dir}\n"
                    f"GGUF 档（ltxv 原生管线）的 Gemma TE/双 VAE/上采样器来自 "
                    f"Lightricks/LTX-2.5 仓库（comfyui_format/），请到模型管理界面下载"
                )
            else:
                # ComfyUI 组件标志文件：Gemma TE（最大权重，model_files.filename）
                marker = comfyui_dir / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
                if not marker.exists():
                    result.warnings.append(
                        f"ComfyUI 组件标志文件不存在：{marker}\n"
                        f"可能 Lightricks/LTX-2.5 仓库未完整下载，建议到模型管理界面重新下载"
                    )

        return result

    def _check_marker_files(
        self,
        scheme: SchemeConfig,
        model_dir: Path,
        result: ValidationResult,
    ) -> None:
        """按方案精度选择正确的标志文件并检查。

        标志文件策略：
            - fp16 / int8_torchao（diffusers 格式）：model_index.json
            - gguf_q8_0（GGUF Q8 格式）：transformer-Q8_0.gguf（manifest 落地名，
              兼容源文件名 LTX-2.5-Distilled-Q8_0.gguf，本地无 .gguf 时兜底 glob）
            - gguf_q4_k_m（GGUF Q4 格式，M4-T4.2）：transformer-Q4_K_M.gguf
              （兼容源文件名 LTX-2.5-Distilled-Q4_K_M.gguf，兜底 glob 同上）

        Args:
            scheme: 方案配置
            model_dir: 方案模型目录
            result: 校验结果（警告追加到此对象）
        """
        if scheme.precision == "gguf_q8_0":
            # manifest ltx25-transformer-gguf-q8 的本地落地名为 transformer-Q8_0.gguf；
            # 兼容手工下载的源文件名 LTX-2.5-Distilled-Q8_0.gguf
            marker_files = ["transformer-Q8_0.gguf", "LTX-2.5-Distilled-Q8_0.gguf"]
            if not any((model_dir / mf).exists() for mf in marker_files):
                marker_files = sorted(p.name for p in model_dir.glob("*.gguf")) or marker_files
        elif scheme.precision == "gguf_q4_k_m":
            # manifest ltx25-transformer-gguf-q4 的本地落地名为 transformer-Q4_K_M.gguf；
            # 兼容手工下载的源文件名 LTX-2.5-Distilled-Q4_K_M.gguf
            marker_files = ["transformer-Q4_K_M.gguf", "LTX-2.5-Distilled-Q4_K_M.gguf"]
            if not any((model_dir / mf).exists() for mf in marker_files):
                marker_files = sorted(p.name for p in model_dir.glob("*.gguf")) or marker_files
        else:
            marker_files = ["model_index.json"]

        marker_exists = False
        for mf in marker_files:
            marker = model_dir / mf
            if marker.exists():
                marker_exists = True
                break
        if not marker_exists:
            result.warnings.append(
                f"标志文件不存在：{marker_files}（在 {model_dir}）\n"
                f"可能模型未完整下载，建议到模型管理界面重新下载"
            )

    @staticmethod
    def _parse_vram_required(vram_str: str) -> Optional[float]:
        """解析显存需求字符串为数值。

        Args:
            vram_str: 显存需求字符串（如 "24GB" / "48GB+" / "8GB"）

        Returns:
            显存数值（GB），解析失败返回 None
        """
        if not vram_str:
            return None
        # 提取数字部分
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", vram_str)
        if match:
            return float(match.group(1))
        return None
