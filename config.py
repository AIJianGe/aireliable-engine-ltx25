"""LTX-2.5 视频生成引擎配置。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.1 设计（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.1.1）：
    - engine_id: ltx25_video_b9c4d2e1（目录/manifest/config 三同名契约）
    - 端口: 7008（7001-7007 已占用）
    - 模型布局: video/ltx25/（diffusers/ + gguf/ 双子目录）

双源真相消除策略（同 H3 引擎）：
    - MODEL_SUBDIR 默认读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback

CPU offload 分级（规范：development-guide.md §37.2.7）：
    - L0: 全 GPU（所有模型常驻 GPU，需 ≥80GB 显存）
    - L1: Text Encoder 卸载（TE 编码完后移到 CPU，需 ≥48GB 显存）
    - L2: 层级别 offload（Transformer 每层按需搬入 GPU，需 ≥24GB 显存，慢 3-5×）
    - L3: 激进 offload（全部模型在 CPU，仅计算时搬入 + VAE tiling，慢 10×+）
    - auto: 根据 GPU 显存自动选择（resolve_offload_level 解析）
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path


# ===== 常量 =====

ENGINE_ID = "ltx25_video_b9c4d2e1"
ENGINE_NAME = "LTX-2.5 Video"
CAPABILITY = "text-to-video"
ENGINE_TYPE = "video"

# 默认端口（与 config/engines/ltx25_video_b9c4d2e1.yaml 一致）
DEFAULT_PORT = 7008

# 模型目录布局（相对 MODELS_ROOT）
# 优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE（平台注入）
_DEFAULT_MODEL_SUBDIR = "video/ltx25/"


def _resolve_model_subdir() -> str:
    """解析模型子目录（优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()

# v0.2.15+ 日志统一到 logs/engines/（项目顶层）
# 旧的 LOG_SUBDIR = "logs/"（相对 data_root）已废弃

# CPU offload 级别常量
OFFLOAD_LEVELS = ("L0", "L1", "L2", "L3")
# 各级别所需的最小 GPU 显存（GB），用于 auto 自动解析
# 必须与 manifest.yaml env_config.cpu_offload.levels.Lx.min_vram_gb 保持一致
# 规范：development-guide.md §37.2.7 CPU Offload 等级规范
_OFFLOAD_VRAM_THRESHOLDS = {
    "L0": 80,   # 全 GPU，所有模型常驻 GPU
    "L1": 48,   # Text Encoder 卸载，TE 编码完后移到 CPU
    "L2": 24,   # 层级别 offload，Transformer 每层按需搬入 GPU
}

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为）
#   gpu     ：强制全 GPU，显存不足直接报错
#   gpu+cpu ：GPU 优先，显存不足时组件溢出 CPU 内存
#   cpu     ：强制 CPU（调试 / 无卡环境）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")


def _detect_gpu_vram_gb() -> float:
    """检测当前 GPU 显存（GB），无 GPU 或检测失败返回 0.0。

    优先使用 torch.cuda，失败时尝试 nvidia-smi。
    """
    # 优先用 torch
    try:
        import torch
        if torch.cuda.is_available():
            # 取第一块 GPU 的总显存
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes / (1024 ** 3)
    except Exception:
        pass

    # fallback：nvidia-smi（Windows / Linux 均可用）
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # 取第一块 GPU 的显存（MB → GB）
            return float(result.stdout.strip().splitlines()[0].strip()) / 1024
    except Exception:
        pass

    return 0.0


@dataclass
class EngineConfig:
    """LTX-2.5 视频生成引擎运行配置。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础 =====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"

    # ===== 路径 =====
    models_root: Path = Path("models")
    data_root: Path = Path("data")
    project_root: Path = Path(".")
    # v0.2.5+ 输出文件根目录统一为 output（项目顶层目录）
    # 规范：{project_root}/output/{category}/{filename}
    output_root: Path = Path("output")

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # ===== 设备 =====
    # auto: 自动检测 GPU / cpu: 强制 CPU / cuda:0: 指定 GPU
    device: str = "auto"

    # ===== 日志 =====
    log_level: str = "INFO"

    # ===== CPU offload =====
    # auto: 根据 GPU 显存自动选择级别（resolve_offload_level 解析）
    # L0/L1/L2/L3: 显式指定卸载级别
    cpu_offload_level: str = "auto"

    # ===== 设备策略（v0.3.5+）=====
    # 平台通过 AIRELIABLE_DEVICE_POLICY 环境变量注入（UI 设置 > yaml > 默认 auto）。
    # 映射规则见 parse_args：gpu→L0 / gpu+cpu→auto（显存分级）/ cpu→device=cpu
    device_policy: str = "auto"

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关（video/ltx25/，含 diffusers/ 与 gguf/ 双子目录）
        self.model_dir = self.models_root / MODEL_SUBDIR

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = self.project_root / "logs" / "engines"

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def resolve_offload_level(self) -> str:
        """解析 CPU offload 级别。

        若 cpu_offload_level 为 "auto"，则根据 GPU 显存自动选择：
            - 显存 ≥ 80GB → L0（全 GPU，所有模型常驻 GPU）
            - 显存 ≥ 48GB → L1（Text Encoder 卸载）
            - 显存 ≥ 24GB → L2（层级别 offload）
            - 显存 < 24GB / 无 GPU → L3（激进 offload）

        若已显式指定 L0/L1/L2/L3，则直接返回。

        阈值必须与 manifest.yaml env_config.cpu_offload.levels.Lx.min_vram_gb 保持一致。
        规范：development-guide.md §37.2.7 CPU Offload 等级规范

        Returns:
            解析后的 offload 级别（"L0" / "L1" / "L2" / "L3"）
        """
        if self.cpu_offload_level != "auto":
            # 显式指定，校验合法性后返回
            level = self.cpu_offload_level.upper()
            if level not in OFFLOAD_LEVELS:
                raise ValueError(
                    f"非法的 cpu_offload_level: {self.cpu_offload_level}，"
                    f"应为 auto / {(' / '.join(OFFLOAD_LEVELS))}"
                )
            return level

        # auto: 根据 GPU 显存自动选择
        vram_gb = _detect_gpu_vram_gb()
        # 按阈值从高到低匹配
        for level in ("L0", "L1", "L2"):
            if vram_gb >= _OFFLOAD_VRAM_THRESHOLDS[level]:
                return level
        # 显存不足 24GB 或无 GPU，使用 L3（激进 offload）
        return "L3"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "device_policy": self.device_policy,
            "log_level": self.log_level,
            "cpu_offload_level": self.cpu_offload_level,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="LTX-2.5 视频生成引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")
    parser.add_argument(
        "--cpu-offload-level", type=str, default="auto",
        help="CPU offload 级别：auto / L0 / L1 / L2 / L3（默认 auto）",
    )

    args = parser.parse_args()

    # 环境变量 fallback
    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))

    # CPU offload 级别：命令行 > 环境变量 > 默认值
    cpu_offload_level = (
        args.cpu_offload_level
        or os.environ.get("AIRELIABLE_CPU_OFFLOAD_LEVEL", "auto")
    )

    # v0.3.5+ 设备策略（平台注入 AIRELIABLE_DEVICE_POLICY）→ 映射到既有 offload 机制：
    #   gpu+cpu → cpu_offload_level=auto（显存自动分级 L0-L3，即溢出 CPU）
    #   gpu     → L0（强制全 GPU）
    #   cpu     → device=cpu（resolve_offload_level 无 GPU 自动 L3）
    #   auto    → 不干预（现状行为）
    # 仅当用户未显式配置 cpu_offload_level（值为 auto）时映射
    device_policy = os.environ.get("AIRELIABLE_DEVICE_POLICY", "auto").strip()
    if device_policy not in DEVICE_POLICIES:
        # 非法值按未设置处理（引擎侧容错，平台侧已校验）
        device_policy = "auto"
    device = args.device
    if device_policy == "gpu" and cpu_offload_level == "auto":
        cpu_offload_level = "L0"
    elif device_policy == "cpu":
        device = "cpu"

    config = EngineConfig(
        port=args.port,
        host=args.host,
        device=device,
        log_level=args.log_level,
        models_root=models_root,
        cpu_offload_level=cpu_offload_level,
        device_policy=device_policy,
    )

    return config
