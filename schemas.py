"""LTX-2.5 视频生成引擎 Pydantic 数据模型。

扩展 SDK 的 VariantInfo / InferRequest，添加视频专属字段。

视频生成模式（task_type）：
    - t2v: Text to Video（文本生成视频，distilled 版默认）
    - i2v: Image to Video（单张参考图做首帧）
    - first_last_frame_v: 首尾帧插值（2 张图，role=first_frame/last_frame）
    - multi_image_v: 多关键帧插值（2~4 张 reference 图均匀分布时间轴）
    - multi_reference_v: 多图参考（2~4 张，暂用时间插值近似，待 DFRPipeline 空间条件）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import VariantInfo


# ===== 视频生成模式枚举 =====
VIDEO_MODES = ("t2v", "i2v", "first_last_frame_v", "multi_image_v", "multi_reference_v")

# ===== 各模式需要的图片数量范围（min, max）=====
MODE_IMAGE_COUNT: dict[str, tuple[int, int]] = {
    "t2v": (0, 0),
    "i2v": (1, 1),
    "first_last_frame_v": (2, 2),
    "multi_image_v": (2, 4),
    "multi_reference_v": (2, 4),
}

# ===== 默认 strength 按 role 映射 =====
ROLE_DEFAULT_STRENGTH: dict[str, float] = {
    "first_frame": 1.0,
    "last_frame": 1.0,
    "reference": 0.8,
    "grid": 0.7,  # 多宫格（空间条件暂用时间条件近似）
}


# ===== role → frame_idx + strength 映射（核心工具函数）=====
def role_to_frame_idx(
    role: str,
    index: int,
    total: int,
    num_frames: int,
) -> tuple[int, float]:
    """把图片的语义 role 映射为原生管线需要的 (frame_idx, strength)。

    frame_idx 决定条件图放在视频时间轴的哪个位置（0=首帧，num_frames-1=尾帧）。
    strength 决定条件强度（1.0=完全替换，0.8=强约束，0.7=弱约束）。

    Args:
        role: 图片角色（"first_frame" / "last_frame" / "reference" / "grid_*"）
        index: 当前图在列表中的序号（0-based）
        total: 图片总数
        num_frames: 目标视频总帧数（8k+1 对齐后的值，如 121）

    Returns:
        (frame_idx, strength)
    """
    # 先按 role 查 strength，grid_N 都算 "grid" 类
    if role.startswith("grid_"):
        strength = ROLE_DEFAULT_STRENGTH["grid"]
    else:
        strength = ROLE_DEFAULT_STRENGTH.get(role, ROLE_DEFAULT_STRENGTH["reference"])

    last_frame_idx = max(0, num_frames - 1)

    if role == "first_frame":
        return (0, strength)
    elif role == "last_frame":
        return (last_frame_idx, strength)
    elif role.startswith("grid_"):
        # 多宫格：暂时都放在首帧位置（P1 空间条件支持后再升级）
        return (0, strength)
    else:
        # "reference" 或未知 role → 按顺序均匀分布
        # total <= 2 时：第一张 0，最后一张 last_frame_idx
        # total > 2 时：首末固定 + 中间均匀分布
        if total <= 1:
            return (0, strength)
        elif total == 2:
            if index == 0:
                return (0, strength)
            else:
                return (last_frame_idx, strength)
        else:
            if index == 0:
                return (0, strength)
            elif index == total - 1:
                return (last_frame_idx, strength)
            else:
                # 中间关键帧均匀分布
                pos = round(index * last_frame_idx / (total - 1))
                return (pos, strength)


class ImageInput(BaseModel):
    """引擎内部多图条件输入（InferRequest.images → ImageConditioningInput 桥梁）。

    在 engine.py 中由 SDK 的 InferRequest.images 转换得到，
    携带 role_to_frame_idx 计算后的 frame_idx，以及（可能）用户显式指定的 strength/crf，
    最终传给 ltxv_native.generate() 变成 ImageConditioningInput。

    优先级（高→低）：
        1. request.images[i].strength / crf（用户显式传）
        2. role_to_frame_idx / 模型训练默认值（引擎自动推断）
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str = Field(..., description="图片文件路径（绝对路径）")
    role: str = Field("reference", description="语义角色：first_frame / last_frame / reference / grid_N")
    frame_idx: int = Field(0, ge=0, description="时间轴位置（0=首帧，num_frames-1=尾帧）")
    strength: float = Field(1.0, ge=0.0, le=1.0, description="条件强度 0-1（用户显式值优先，默认按 role 推断）")
    crf: Optional[int] = Field(default=None, ge=0, le=40, description="H.264 CRF 重压缩参数（None=模型训练默认）")
    pil: Optional[Any] = Field(None, description="PIL.Image 对象（engine.py 加载后缓存，wrapper 内部用）")


# ===== CPU offload 级别（与 config.py 的 OFFLOAD_LEVELS 保持一致） =====
OFFLOAD_LEVELS = ("L0", "L1", "L2", "L3")


class VideoVariantInfo(VariantInfo):
    """视频变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充视频特有信息：分辨率 / 帧率 / 时长 / 适用的量化档位。
    Pydantic model_dump() 会序列化全部字段，前端可读取这些扩展字段
    渲染视频专属 UI。
    """

    model_config = ConfigDict(extra="allow")

    resolution: str = Field(
        "960x544",
        description="默认输出分辨率（宽 x 高，需被 32 整除）",
    )
    fps: int = Field(24, ge=1, le=60, description="默认帧率")
    duration: int = Field(5, ge=1, le=60, description="默认时长（秒）")
    max_duration: int = Field(60, ge=1, description="最大支持时长（秒）")
    supports_audio: bool = Field(
        False,
        description="是否支持音频生成（LTX-2.5 v1 关闭，M3 音频链路启用后打开）",
    )


class VideoInputs(BaseModel):
    """视频推理输入参数（放入 InferRequest.inputs）。

    所有模式共用此 schema，根据 task_type 字段决定使用哪些参数。

    LTX-2.5 约束：
        - 分辨率需被 32 整除（VAE 空间压缩 32×）
        - 帧数需满足 8k+1（时间压缩 8×，如 121 帧 ≈ 5s@24fps）
        - distilled 版固定 8 步、CFG=1（负向提示词无效）
    """

    model_config = ConfigDict(extra="allow")

    # ===== 必填 =====
    prompt: str = Field(..., description="正向提示词")

    # ===== 任务类型与负向提示词 =====
    negative_prompt: str = Field("", description="负向提示词（distilled CFG=1 时无效，dev 版预留）")
    task_type: str = Field(
        "t2v",
        description=f"生成模式：{VIDEO_MODES}",
    )

    # ===== 视频参数 =====
    duration: int = Field(5, ge=1, le=60, description="视频时长（秒，num_frames 未显式指定时用于换算 8k+1）")
    fps: int = Field(24, ge=1, le=60, description="帧率")
    width: int = Field(960, ge=256, le=1920, description="视频宽度（32 的倍数）")
    height: int = Field(544, ge=256, le=1920, description="视频高度（32 的倍数）")
    num_frames: Optional[int] = Field(
        None, ge=1, le=1000,
        description="显式指定帧数（None=由 duration×fps 对齐 8k+1）",
    )

    # ===== 推理参数 =====
    # 默认 None，让 _apply_scheme_defaults 使用方案默认值（如 I8 档的 8 步）
    # 如果显式传入，则使用传入值（覆盖方案默认值）
    num_inference_steps: Optional[int] = Field(
        None, ge=1, le=200, description="推理步数（去噪步数，None=使用方案默认值；distilled 固定 8）"
    )
    guidance_scale: Optional[float] = Field(
        None, ge=0.0, le=20.0, description="CFG 引导强度（None=使用方案默认值；distilled 固定 1.0）"
    )
    seed: int = Field(-1, ge=-1, description="随机种子（-1=随机）")

    # ===== 自动时长（T3.3，仅 G8 原生链路生效）=====
    # DurationHead 按提示词预测自然时长（回归目标 log-seconds），
    # 覆盖 duration 推导的帧数；文件缺失或预测失败时降级手动时长
    auto_duration: bool = Field(
        False,
        description="自动时长开关：DurationHead 按提示词预测帧数（仅 G8 档生效，默认关闭）",
    )

    # ===== DFRPipeline 进阶参数（P1-A：多宫格参考 + 帧间补帧）=====
    # temporal_upscalings > 0 触发 DFRPipeline 路由；DFR 模型缺失时自动降级 P0
    temporal_upscalings: int = Field(
        0, ge=0, le=2,
        description="DFRPipeline 帧间补帧轮数（0=不补帧，1=2×上采样，2=4×上采样；仅 DFR 模型就绪时生效）",
    )
    workflow: Optional[str] = Field(
        None,
        description="工作流标识：multi_reference_v / temporal_upscale_v 等；触发 DFR 路由的辅助判断",
    )

    # ===== CPU offload 级别 =====
    # auto: 根据 GPU 显存自动选择（config.resolve_offload_level 解析）
    # L0/L1/L2/L3: 显式指定卸载级别
    # 注意：此字段会覆盖引擎启动时的 cpu_offload_level 配置
    cpu_offload_level: str = Field(
        "auto",
        description=f"CPU offload 级别：auto / {' / '.join(OFFLOAD_LEVELS)}",
    )


class VideoStatusExtra(BaseModel):
    """视频引擎状态附加信息。"""

    device: str = Field("cpu", description="当前推理设备")
    vram_used: float = Field(0.0, description="已用显存（MB）")
    offload_level: str = Field("L1", description="当前 CPU offload 级别")
    loaded_variant: Optional[str] = Field(None, description="已加载的模型变体")
    loaded_loras: list[str] = Field(
        default_factory=list, description="已加载的 LoRA 列表"
    )
