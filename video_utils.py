"""视频后处理工具。

提供视频帧序列保存、音视频合成、图片/视频加载等功能。

依赖：
    - imageio（帧序列 → 视频文件）
    - numpy（数组处理）
    - PIL（图片加载）
    - ffmpeg（音视频合成，通过 subprocess 调用）

ffmpeg 不可用时降级为仅保存无音轨视频。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def save_video(
    frames: np.ndarray,
    audio: Optional[np.ndarray],
    output_dir: Path,
    task_id: str,
    fps: int = 24,
    sample_rate: int = 44100,
) -> dict:
    """将帧序列保存为视频文件（含可选音频合成）。

    输出文件结构：
        {output_dir}/{task_id}_video_only.mp4   仅视频（无音轨）
        {output_dir}/{task_id}_audio.wav        音频（若有）
        {output_dir}/{task_id}.mp4              最终视频（视频 + 音频合成）

    Args:
        frames: 视频帧序列，形状 (T, H, W, C) 或 (T, H, W)，dtype=uint8
        audio: 音频数据，形状 (samples,) 或 (channels, samples)，dtype=float32。
            为 None 时只保存无音轨视频。
        output_dir: 输出目录（自动创建）
        task_id: 任务 ID（用于文件名前缀）
        fps: 视频帧率
        sample_rate: 音频采样率（仅在 audio 不为 None 时使用）

    Returns:
        dict 含：
            - video_url: 最终视频相对路径（如 /output/video/{task_id}.mp4）
            - video_only_url: 仅视频相对路径
            - audio_url: 音频相对路径（无音频时为空字符串）
            - duration: 视频时长（秒，基于帧数/fps，不受 mux 截断影响）
            - fps: 帧率
            - warnings: 合成过程中产生的警告列表（如时长不匹配提示）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1. 保存仅视频文件 =====
    video_only_filename = f"{task_id}_video_only.mp4"
    video_only_path = output_dir / video_only_filename
    _write_frames_to_mp4(frames, video_only_path, fps)

    # ===== 2. 保存音频（若有）=====
    audio_filename = ""
    audio_path: Optional[Path] = None
    if audio is not None:
        audio_filename = f"{task_id}_audio.wav"
        audio_path = output_dir / audio_filename
        _write_audio_to_wav(audio, audio_path, sample_rate)

    # ===== 3. 计算视频和音频时长（用于 mux 策略选择）=====
    video_duration = frames.shape[0] / fps if fps > 0 else 0.0
    audio_duration: Optional[float] = None
    if audio is not None:
        if audio.ndim == 2:
            audio_duration = audio.shape[1] / sample_rate if sample_rate > 0 else None
        else:
            audio_duration = audio.shape[0] / sample_rate if sample_rate > 0 else None

    # ===== 4. 合成视频 + 音频（若音频存在且 ffmpeg 可用）=====
    final_filename = f"{task_id}.mp4"
    final_path = output_dir / final_filename

    audio_merged = False
    mux_warnings: list[str] = []
    if audio_path is not None and audio_path.exists():
        if _ffmpeg_available():
            try:
                mux_result = _mux_video_audio(
                    video_only_path, audio_path, final_path,
                    video_duration, audio_duration,
                )
                audio_merged = True
                mux_warnings = mux_result.get("warnings", [])
            except Exception as e:
                # ffmpeg 合成失败时降级为仅视频（不阻断流程）
                logger.warning(
                    "ffmpeg 合成音视频失败，降级为仅视频（无音轨）: %s。错误: %s",
                    final_path, e,
                )
                mux_warnings.append(f"音视频合成失败，已降级为无音轨视频: {e}")
                shutil.copy2(video_only_path, final_path)
        else:
            # ffmpeg 不可用，降级为仅视频（复制 video_only → final）
            logger.warning(
                "ffmpeg 不可用，无法合成音视频，最终视频无音轨: %s",
                final_path,
            )
            mux_warnings.append("ffmpeg 不可用，最终视频无音轨")
            shutil.copy2(video_only_path, final_path)
    else:
        # 无音频，直接复制 video_only 为最终视频
        shutil.copy2(video_only_path, final_path)

    # duration 基于帧数/fps 计算，不受 mux 截断影响
    duration = video_duration
    # 若 ffmpeg 合成失败（降级为仅视频），audio_url 返回空字符串
    final_audio_filename = audio_filename if audio_merged else ""
    result = {
        "video_url": f"/output/video/{final_filename}",
        "video_only_url": f"/output/video/{video_only_filename}",
        "audio_url": f"/output/video/{final_audio_filename}" if final_audio_filename else "",
        "duration": duration,
        "fps": fps,
        "warnings": mux_warnings,
    }
    logger.info(
        "视频已保存: task_id=%s, final=%s, duration=%.2fs, fps=%d, has_audio=%s, warnings=%d",
        task_id, final_path, duration, fps, audio is not None, len(mux_warnings),
    )
    return result


def _write_frames_to_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    """使用 imageio 将帧序列写入 MP4 文件。

    Args:
        frames: 帧序列，形状 (T, H, W, C) 或 (T, H, W)
        path: 输出路径
        fps: 帧率
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError(
            f"imageio 未安装，无法保存视频: {e}。"
            f"请在 envs/video-ltx25/ 环境中运行 `pip install imageio[ffmpeg]`"
        ) from e

    # 确保帧为 uint8
    if frames.dtype != np.uint8:
        frames = _to_uint8(frames)

    # imageio 要求 (T, H, W, C) 格式，灰度图自动扩展维度
    if frames.ndim == 3:
        # (T, H, W) → (T, H, W, 1)
        frames = np.repeat(frames[..., np.newaxis], 3, axis=-1)
    elif frames.ndim == 4 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    finally:
        writer.close()

    logger.debug("帧序列已写入: %s (frames=%d, fps=%d)", path, frames.shape[0], fps)


def _write_audio_to_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    """将音频数据写入 WAV 文件。

    Args:
        audio: 音频数据，形状 (samples,) 或 (channels, samples)，dtype=float32
        path: 输出路径
        sample_rate: 采样率
    """
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            f"soundfile 未安装，无法保存音频: {e}。"
            f"请在 envs/video-ltx25/ 环境中运行 `pip install soundfile`"
        ) from e

    # 确保数据类型为 float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 归一化到 [-1.0, 1.0]
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    # soundfile 期望 (samples, channels) 或 (samples,)，
    # 部分模型音频输出是 (channels, samples) 立体声。
    # 当 2D 数组的第一维小于第二维时，判定为 (channels, samples) 并转置。
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        logger.debug("音频形状转置: %s -> %s", audio.shape, audio.T.shape)
        audio = audio.T

    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    logger.debug("音频已写入: %s (sr=%d, shape=%s)", path, sample_rate, audio.shape)


# 缓存 ffmpeg 可执行文件路径，避免每次调用重复查找。
# 系统 PATH 中的 ffmpeg 可能是裁剪版，不支持 aac 编码器，
# 优先使用 imageio_ffmpeg 提供的完整版。
_FFMPEG_CACHE: Optional[str] = None
_FFMPEG_PROBED: bool = False


def _get_ffmpeg_executable() -> Optional[str]:
    """查找可用的 ffmpeg 可执行文件路径。

    优先级：
        1. imageio_ffmpeg.get_ffmpeg_exe() — 随环境分发的完整版，支持全部编码器
        2. shutil.which("ffmpeg") — 系统 PATH 中的 ffmpeg（可能是裁剪版，作为兼容回退）

    结果会被缓存（模块级 _FFMPEG_CACHE / _FFMPEG_PROBED），避免每次调用都重复查找。

    Returns:
        ffmpeg 可执行文件路径字符串；找不到返回 None。
    """
    global _FFMPEG_CACHE, _FFMPEG_PROBED
    if _FFMPEG_PROBED:
        return _FFMPEG_CACHE
    _FFMPEG_PROBED = True

    # 1) 优先使用 imageio_ffmpeg 提供的完整版 ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            _FFMPEG_CACHE = exe
            logger.debug("ffmpeg 使用 imageio_ffmpeg 提供的版本: %s", exe)
            return _FFMPEG_CACHE
    except Exception as e:
        logger.debug("imageio_ffmpeg 不可用，回退到系统 PATH: %s", e)

    # 2) 回退到系统 PATH 中的 ffmpeg（注意：可能是裁剪版，不支持 aac）
    sys_exe = shutil.which("ffmpeg")
    if sys_exe:
        _FFMPEG_CACHE = sys_exe
        logger.debug("ffmpeg 使用系统 PATH 版本: %s", sys_exe)
    else:
        logger.debug("ffmpeg 未找到（imageio_ffmpeg 和系统 PATH 均无）")
    return _FFMPEG_CACHE


def _ffmpeg_available() -> bool:
    """检查 ffmpeg 是否可用。"""
    return _get_ffmpeg_executable() is not None


def _mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_duration: Optional[float] = None,
    audio_duration: Optional[float] = None,
) -> dict:
    """使用 ffmpeg 合成视频和音频（兼容性优先，不截断视频）。

    策略：
        - 视频长于音频（>0.1s 差异）：用 apad 补静音到视频时长，保证视频完整
        - 音频长于视频（>0.1s 差异）：用 -shortest 截断音频尾部（安全，不影响视频）
        - 时长接近（<0.1s）：直接 mux
        - 加 -fflags +genpts -avoid_negative_ts make_zero 修复 PTS 负值

    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径
        output_path: 输出路径
        video_duration: 视频时长（秒），None 时无法做时长对齐
        audio_duration: 音频时长（秒），None 时无法做时长对齐

    Returns:
        dict 含：
            - warnings: 合成过程中的警告信息列表

    Raises:
        RuntimeError: ffmpeg 调用失败
    """
    ffmpeg_exe = _get_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg 未找到（imageio_ffmpeg 与系统 PATH 均无）")

    # ---- 时长差异分析 ----
    warnings: list[str] = []
    DURATION_THRESHOLD = 0.1  # 100ms 以内视为一致

    # 构建命令
    cmd = [ffmpeg_exe, "-y"]
    # PTS 修复：补充缺失 PTS + 消除负值时间戳
    cmd.extend(["-fflags", "+genpts"])
    cmd.extend(["-i", str(video_path), "-i", str(audio_path)])

    # 视频流直接复制（不重新编码）
    cmd.extend(["-c:v", "copy"])

    if video_duration is not None and audio_duration is not None:
        diff = video_duration - audio_duration
        if diff > DURATION_THRESHOLD:
            # 视频长于音频：补静音到视频时长（保证视频完整不被截断）
            cmd.extend(["-af", f"apad=whole_dur={video_duration:.3f}"])
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            warnings.append(
                f"音频短于视频 {diff:.1f}s，已补静音尾帧（视频时长 {video_duration:.1f}s / 音频时长 {audio_duration:.1f}s）"
            )
        elif diff < -DURATION_THRESHOLD:
            # 音频长于视频：截断音频尾部（安全，不影响视频）
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
            warnings.append(
                f"音频长于视频 {-diff:.1f}s，已截断音频尾部（视频时长 {video_duration:.1f}s / 音频时长 {audio_duration:.1f}s）"
            )
        else:
            # 时长接近，直接 mux
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        # 时长未知：不截断，用 -t 限制为视频时长（若已知）
        if video_duration is not None:
            cmd.extend(["-t", f"{video_duration:.3f}"])
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        warnings.append("音视频时长未检测，按视频时长对齐")

    # PTS 修复 + 流式播放优化
    cmd.extend(["-avoid_negative_ts", "make_zero"])
    cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(output_path))

    logger.info("ffmpeg 合成音视频: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 合成失败 (returncode={result.returncode}): {result.stderr[:500]}"
            )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg 合成超时（120s）: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg 未找到: {e}") from e

    return {"warnings": warnings}


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    """将帧序列归一化并转换为 uint8。

    Args:
        frames: 任意浮点或整数类型的帧序列

    Returns:
        uint8 类型的帧序列，范围 [0, 255]
    """
    if frames.dtype == np.uint8:
        return frames
    # 浮点数据假定范围 [0, 1]，转换到 [0, 255]
    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames, 0.0, 1.0)
        return (frames * 255).astype(np.uint8)
    # 其他整数类型直接 clip 到 [0, 255]
    return np.clip(frames, 0, 255).astype(np.uint8)


def load_image(url_or_path: str):
    """加载图片为 PIL.Image 对象。

    支持：
        - 本地文件路径
        - http(s) URL（通过 urllib 下载后加载）

    Args:
        url_or_path: 图片路径或 URL

    Returns:
        PIL.Image.Image 对象

    Raises:
        RuntimeError: 加载失败
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            f"Pillow 未安装，无法加载图片: {e}。"
            f"请在 envs/video-ltx25/ 环境中运行 `pip install pillow`"
        ) from e

    try:
        if url_or_path.startswith(("http://", "https://")):
            # HTTP URL：通过 io.BytesIO 加载
            import io
            import urllib.request
            with urllib.request.urlopen(url_or_path, timeout=30) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        else:
            # 本地路径
            return Image.open(url_or_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"图片加载失败: {url_or_path}: {e}") from e


def load_video(url_or_path: str) -> list[np.ndarray]:
    """加载视频为帧序列列表。

    支持：
        - 本地文件路径
        - http(s) URL

    Args:
        url_or_path: 视频路径或 URL

    Returns:
        帧列表，每帧为 (H, W, C) uint8 数组

    Raises:
        RuntimeError: 加载失败
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError(
            f"imageio 未安装，无法加载视频: {e}。"
            f"请在 envs/video-ltx25/ 环境中运行 `pip install imageio[ffmpeg]`"
        ) from e

    try:
        reader = imageio.get_reader(url_or_path)
        frames: list[np.ndarray] = []
        try:
            for frame in reader:
                frames.append(frame.astype(np.uint8))
        finally:
            reader.close()
        return frames
    except Exception as e:
        raise RuntimeError(f"视频加载失败: {url_or_path}: {e}") from e
