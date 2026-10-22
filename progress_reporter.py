"""SSE 进度上报工具（LTX-2.5 视频生成专用）。

把视频生成的推理过程转换为 AIReliable 进度帧。

进度阶段：
    start             0.00          开始视频生成
    model_loading     0.05          模型已加载确认
    prompt_processing 0.10          提示词处理
    generating        0.10 - 0.85   视频帧生成中（含扩散去噪）
    vae_decoding      0.85          VAE 解码（潜空间 → 像素帧）
    saving            0.92          保存视频文件
    completed         1.00          完成

v0.3.4+ 集成 SDK 任务上报：
    - emit_start → fire_report_running（gateway status=running）
    - emit (通用) → fire_report_progress（gateway 更新进度）
    - emit_completed → fire_report_succeeded（gateway status=succeeded）
    - emit_error → fire_report_failed（gateway status=failed）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from aireliable_engine import (
    SSEStream,
    make_frame,
    make_final_frame,
    make_error_frame,
    PlatformTaskReporter,
    get_platform_task_reporter,
)

logger = logging.getLogger(__name__)


# ===== 进度阶段定义 =====

class ProgressStage:
    """视频生成推理进度阶段。"""

    START = "video.start"                       # 0.00
    MODEL_READY = "video.model_ready"           # 0.05
    PROMPT_PROCESSING = "video.prompt_processing"  # 0.10
    GENERATING = "video.generating"             # 0.10 - 0.85
    VAE_DECODING = "video.vae_decoding"         # 0.85
    SAVING = "video.saving"                     # 0.92
    MUX_WARNING = "video.mux_warning"           # 0.95
    COMPLETED = "video.completed"               # 1.00
    ERROR = "video.error"


# 进度区间
STAGE_RANGES: dict[str, tuple[float, float]] = {
    ProgressStage.START: (0.00, 0.05),
    ProgressStage.MODEL_READY: (0.05, 0.10),
    ProgressStage.PROMPT_PROCESSING: (0.10, 0.15),
    ProgressStage.GENERATING: (0.15, 0.85),
    ProgressStage.VAE_DECODING: (0.85, 0.90),
    ProgressStage.SAVING: (0.92, 0.95),
    ProgressStage.MUX_WARNING: (0.95, 1.00),
}


class ProgressReporter:
    """视频生成进度上报器。

    封装 SSEStream，提供按阶段构造帧的便捷方法。
    v0.3.4+ 同时通过 SDK PlatformTaskReporter 上报任务状态到 gateway。
    """

    def __init__(
        self,
        stream: SSEStream,
        task_reporter: Optional[PlatformTaskReporter] = None,
    ):
        """初始化进度上报器。

        Args:
            stream: SSE 流
            task_reporter: 可选的 SDK 任务上报器（v0.3.4+）。
                为 None 时不上报任务状态到 gateway（仅 SSE 推送）。
        """
        self.stream = stream
        self._start_time: float = time.time()
        self._task_reporter: Optional[PlatformTaskReporter] = task_reporter
        self._task_id: Optional[str] = None

    @property
    def elapsed(self) -> float:
        """已用时（秒）。"""
        return time.time() - self._start_time

    def set_task_id(self, task_id: str) -> None:
        """设置任务 ID（用于 SDK 上报）。"""
        self._task_id = task_id

    async def init_task_reporter(self) -> None:
        """v0.3.4+ 异步初始化 SDK 任务上报器（捕获主 loop + 获取单例）。

        必须在 emit_* 方法之前调用（在 _infer_impl 入口处）。
        失败不阻断推理。
        """
        if self._task_reporter is None and self._task_id:
            try:
                self._task_reporter = await get_platform_task_reporter()
            except Exception as e:
                logger.warning("init_task_reporter 失败（不阻断推理）: %s", e)

    async def report_created(
        self,
        request_summary: dict,
        variant: str = "",
        task_type: str = "",
    ) -> None:
        """v0.3.4+ 上报任务创建到 gateway（status=queued）。

        v0.2.16+ task_type：直连链路 INSERT 新任务时写入任务类型
        （如 video_generation），与任务中心筛选语义对齐；
        gateway 转发链路（request.task_id 非空）下 gateway 已建任务，
        引擎上报不覆盖。
        """
        if self._task_reporter and self._task_id:
            try:
                await self._task_reporter.report_created(
                    task_id=self._task_id,
                    request_summary=request_summary,
                    variant=variant,
                    task_type=task_type,
                )
            except Exception as e:
                logger.warning("report_created 失败（不阻断推理）: %s", e)

    def _fire_running(self, stage: str) -> None:
        """v0.3.4+ fire-and-forget 上报任务进入 running 状态。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_running(self._task_id, stage=stage)

    def _fire_progress(self, progress: float, stage: str) -> None:
        """v0.3.4+ fire-and-forget 上报任务进度。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_progress(
                self._task_id, progress=progress, stage=stage,
            )

    def _fire_succeeded(self, result: dict) -> None:
        """v0.3.4+ fire-and-forget 上报任务成功。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_succeeded(self._task_id, result=result)

    def _fire_failed(self, error: dict) -> None:
        """v0.3.4+ fire-and-forget 上报任务失败。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_failed(self._task_id, error=error)

    # ===== 通用帧 =====

    def emit(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（线程安全）。"""
        frame = make_frame(progress, stage, **extra)
        self.stream.put_threadsafe(frame)
        # v0.3.4+ 同步上报进度到 gateway（fire-and-forget，不阻塞 SSE 流）
        self._fire_progress(progress, stage)

    async def emit_async(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（协程内调用）。"""
        frame = make_frame(progress, stage, **extra)
        await self.stream.put(frame)
        # v0.3.4+ 同步上报进度到 gateway
        self._fire_progress(progress, stage)

    def emit_error(self, message: str, error_type: str = "video_error") -> None:
        """发送错误帧。"""
        frame = make_error_frame(message, error_type)
        self.stream.put_threadsafe(frame)
        # v0.3.4+ 上报任务失败到 gateway
        self._fire_failed({"type": error_type, "message": message})

    def emit_heartbeat(self) -> None:
        """发送心跳帧（重置 gateway httpx read timeout）。

        视频生成耗时长（数十秒到数分钟），generate() 期间无 SSE 帧输出，
        会导致 gateway httpx read timeout 触发 → 502 Bad Gateway。
        在长时间阻塞期间定期推送心跳帧，重置 gateway 的 read timeout 计时。
        """
        self.emit(
            0.50,
            "video.heartbeat",
            message="生成中（心跳）",
            elapsed=self.elapsed,
        )

    # ===== 阶段便捷方法 =====

    def emit_start(self, task_id: str, description: str = "") -> None:
        """开始推理。

        Args:
            task_id: 任务 ID
            description: 任务描述（如 prompt 摘要）
        """
        self._task_id = task_id
        self.emit(
            0.0,
            ProgressStage.START,
            message="开始视频生成",
            task_id=task_id,
            description=description[:200],
        )
        # v0.3.4+ 上报任务进入 running 状态
        self._fire_running(ProgressStage.START)

    def emit_model_ready(self, variant: str, device: str) -> None:
        """模型已就绪。"""
        self.emit(
            0.05,
            ProgressStage.MODEL_READY,
            message="模型已加载",
            variant=variant,
            device=device,
        )

    def emit_prompt_processing(self, prompt: str) -> None:
        """提示词处理中。"""
        self.emit(
            0.10,
            ProgressStage.PROMPT_PROCESSING,
            message="提示词处理中",
            prompt_length=len(prompt),
        )

    def emit_generating(self, progress: float = 0.15, **extra: Any) -> None:
        """视频帧生成中。"""
        message = extra.pop("message", "生成视频帧中...")
        self.emit(progress, ProgressStage.GENERATING, message=message, **extra)

    def emit_vae_decoding(self) -> None:
        """VAE 解码中。"""
        self.emit(0.85, ProgressStage.VAE_DECODING, message="VAE 解码（潜空间 → 像素帧）")

    def emit_saving(self) -> None:
        """保存中。"""
        self.emit(0.92, ProgressStage.SAVING, message="保存视频文件")

    async def emit_mux_warning(self, message: str) -> None:
        """音视频合成警告。

        在 emit_saving 之后、emit_completed 之前推送，
        前端据此显示 toast 警告提示。

        Args:
            message: 警告信息
        """
        await self.emit_async(0.95, ProgressStage.MUX_WARNING, message=message)

    async def emit_completed(
        self,
        task_id: str,
        result: dict,
    ) -> None:
        """完成。

        Args:
            task_id: 任务 ID
            result: 结果字典，含 video_url / audio_url / duration / fps 等字段
        """
        timing = {
            "total_seconds": self.elapsed,
            "video_duration": result.get("duration", 0),
            "fps": result.get("fps", 24),
        }
        final_payload = {
            "task_id": task_id,
            **result,
        }
        frame = make_final_frame(final_payload, timing)
        await self.stream.put(frame)
        # v0.3.4+ 上报任务成功到 gateway
        self._fire_succeeded(final_payload)
