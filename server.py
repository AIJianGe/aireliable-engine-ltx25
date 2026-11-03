"""LTX-2.5 视频生成引擎 HTTP 服务（FastAPI）。

8 个标准端点 / CORS / 日志端点 / 静态挂载 / 根路径跳转由 SDK 骨架
（``build_engine_app``）提供；本文件保留方案管理端点与 ``/upload``。

启动：
    uv run --no-sync --project envs/video-ltx25 \\
        python -m engines.ltx25_video_b9c4d2e1.server --port 7008

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体，diffusers + GGUF Q8/Q4 实体）
    POST /load           加载 LTX-2.5 模型权重（fire-and-forget，立即返回 loading）
    POST /unload         卸载模型（释放显存）
    POST /infer          视频生成推理（SSE 流式进度，t2v / i2v）
    GET  /status         状态查询（含已加载方案 / 四档方案列表）
    POST /lora/load      LoRA 加载（v1 预留，不支持）
    POST /lora/unload    LoRA 卸载（v1 预留，不支持）
    GET  /schemes        列出四档量化方案（F16 / I8 / G8 / G4）
    GET  /schemes/default  获取默认方案
    POST /schemes/default  设置默认方案（持久化）
    GET  /output/*       静态视频文件（推理产物）
    GET  /log_level      查询/调整日志级别
    POST /upload         首帧图上传（WebUI i2v 模式，返回引擎侧路径）
    GET  /               307 重定向到 /ui/
    GET  /ui/*           原生 WebUI 静态文件（webui/ 三件套）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException

from aireliable_engine import build_engine_app, mount_upload, run_engine_server

from .config import EngineConfig, parse_args
from .engine import LTX25Engine

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).parent / "webui"


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[LTX25Engine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> LTX25Engine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: LTX25Engine) -> None:
    """启动钩子：绑定全局实例 + 异步刷新 variant 扫描。"""
    global _engine

    _engine = engine
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_scheme=%s",
        engine.state_machine.state.value,
        len(engine._available_variants),
        engine.get_default_scheme(),
    )


async def _on_shutdown(engine: LTX25Engine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== LTX-2.5 视频生成引擎关闭 =====")
    try:
        if engine.wrapper.is_loaded:
            try:
                await engine._unload_impl()
            except Exception as e:
                logger.warning("关闭时卸载模型失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由 =====

router = APIRouter()


mount_upload(
    router,
    lambda: get_config().output_root / "uploads",
    log_label="首帧图",
)


@router.get("/schemes")
async def list_schemes() -> dict:
    """列出四档量化方案（F16 / I8 / G8 / G4）。

    返回方案列表，每项含 id / name / description / vram_required /
    precision / quantization_type / size_gb / offload_level。
    """
    engine = get_engine()
    return {
        "schemes": engine.list_schemes(),
        "default_scheme_id": engine.get_default_scheme(),
        "loaded_scheme_id": engine.wrapper.loaded_scheme_id,
    }


@router.get("/schemes/default")
async def get_default_scheme() -> dict:
    """获取当前默认方案 ID。"""
    engine = get_engine()
    return {
        "default_scheme_id": engine.get_default_scheme(),
        "loaded_scheme_id": engine.wrapper.loaded_scheme_id,
    }


@router.post("/schemes/default")
async def set_default_scheme(body: dict) -> dict:
    """设置默认方案并持久化。

    请求体：
        {"scheme_id": "I8"}

    设置后下次加载 diffusers variant 时将自动使用该方案。
    已加载的模型不会立即切换（需 unload + load 新方案）。
    GGUF variant 固定使用方案 G8，不受默认方案影响。
    """
    engine = get_engine()
    scheme_id = body.get("scheme_id")
    if not scheme_id:
        raise HTTPException(status_code=400, detail="缺少 scheme_id 参数")

    ok = engine.set_default_scheme(scheme_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"方案 {scheme_id} 不存在（可用: F16 / I8 / G8）",
        )
    return {
        "ok": True,
        "default_scheme_id": scheme_id,
        "message": f"默认方案已设置为 {scheme_id}，下次加载 diffusers variant 时生效",
    }


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        LTX25Engine,
        config=config,
        title="LTX-2.5 Video Engine",
        description=(
            "AIReliable 平台的 LTX-2.5 视频生成引擎"
            "（四档量化：F16 / I8 / G8 / G4，Turing 架构适配）"
        ),
        version="0.1.0",
        static_dirs={"/ui": _WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

def main() -> None:
    """命令行启动入口。"""
    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
