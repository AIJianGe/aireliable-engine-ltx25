"""LTX-2.5 视频生成引擎推理核心。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 LTX25Wrapper 的加载/推理/卸载流程。

四档量化方案（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §5 + M4-T4.2）：
    - F16：BF16 权重 cast 为 FP16（Turing SM 7.5 不支持 BF16，数值转换无损）
    - I8：torchao INT8 运行时量化（transformer 量化，其余 FP16）
    - G8：GGUF Q8_0 社区格式（ltxv 原生管线，blocks 分时加载）
    - G4：GGUF Q4_K_M 低配档（M4-T4.2，与 G8 同管线，仅权重不同）

8 端点契约：
    - /capabilities    返回扫描到的 LTX-2.5 模型变体（diffusers + GGUF Q8/Q4 实体）
    - /load            调用 LTX25Wrapper.load() 加载模型权重（fire-and-forget）
    - /unload          调用 LTX25Wrapper.unload() 释放显存
    - /infer           解析 VideoInputs → wrapper.generate → 保存视频 → SSE 推送
    - /status          返回引擎状态（已加载变体 / 方案 / 设备 / 显存 / offload）
    - /lora/load       v1 不支持 LoRA（预留接口，M2 扩展项）
    - /lora/unload     v1 不支持 LoRA（预留接口，M2 扩展项）
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    EngineBusyError,
    InferRequest,
    LoadRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)
from aireliable_engine.state_machine import RuntimeState

from .config import EngineConfig
from .ltx25_wrapper import LTX25Wrapper
from .model_scanner import ModelScanner
from .progress_reporter import ProgressReporter
from .scheme_manager import SchemeConfig, SchemeManager
from .scheme_state_store import SchemeStateStore
from .scheme_validator import SchemeValidator
from .schemas import VIDEO_MODES, MODE_IMAGE_COUNT, VideoInputs, VideoVariantInfo, ImageInput, role_to_frame_idx
from .video_utils import load_image, save_video

logger = logging.getLogger(__name__)


class LTX25Engine(CustomEngine):
    """LTX-2.5 视频生成引擎。

    全自定义模式（integrated 范式），直接控制模型加载/推理流程，
    不走工作流节点系统。支持 t2v / i2v 双模式视频生成，
    四档量化方案（F16 / I8 / G8 / G4）适配 Turing 架构。
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # ===== 方案管理组件 =====
        # schemes/ 目录与 engine.py 同级
        schemes_dir = Path(__file__).parent / "schemes"
        self.scheme_manager = SchemeManager(schemes_dir=schemes_dir, logger=logger)
        # 方案状态持久化（存储用户选择的默认方案）
        self.scheme_store = SchemeStateStore(
            data_root=config.data_root,
            engine_id=config.engine_id,
            logger=logger,
        )
        # 方案校验器（加载前校验合法性：匹配性 + 显存 + 标志文件）
        self.scheme_validator = SchemeValidator(
            scheme_manager=self.scheme_manager,
            models_root=config.models_root,
            logger=logger,
        )

        # ===== 核心组件 =====
        self.wrapper = LTX25Wrapper(
            models_root=config.models_root,
            model_dir=config.model_dir,
            logger=logger,
        )
        # v0.2.11+ 方案 B：注入 engine=self，启用平台 API 声明式检测
        self.scanner = ModelScanner(
            models_root=config.models_root,
            model_dir=config.model_dir,
            engine=self,
            logger=logger,
        )

        # 启动时扫描模型目录（同步初始化为空，异步扫描在 lifespan 中触发）
        self._available_variants: list = []
        self._refresh_scan()

        # LoRA 列表（LTX-2.5 v1 不支持，恒为空）
        self._available_loras: list[str] = self.scanner.scan_loras()

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        # 优先级：request.variant > self.default_variant > 第一个 available variant
        _manifest_path = Path(__file__).parent / "manifest.yaml"
        self.default_variant = self._read_default_variant_from_manifest(_manifest_path)

        # 读取 manifest 中的 capabilities 多能力声明（v0.4.3+ 多能力引擎）
        # 如 LTX-2.5 同时支持 t2v / i2v / 首尾帧 / 多图 / 多宫格
        self._manifest_capabilities: list[str] = self._read_capabilities_from_manifest(_manifest_path)

        # 加载默认方案（从 state store 读取，下次启动自动使用）
        self._default_scheme_id: Optional[str] = self.scheme_store.get_default_scheme()
        if self._default_scheme_id:
            logger.info("从 state store 加载默认方案: %s", self._default_scheme_id)
        else:
            # 启动时按显存自动选档并持久化
            auto_scheme = self.scheme_manager.auto_select_scheme()
            if auto_scheme:
                self._default_scheme_id = auto_scheme.scheme
                self.scheme_store.set_default_scheme(auto_scheme.scheme)
                logger.info(
                    "自动选档并持久化默认方案: %s（%s）",
                    auto_scheme.scheme, auto_scheme.name,
                )

    # ============ manifest 辅助读取 ============

    def _read_capabilities_from_manifest(self, manifest_path: Path) -> list[str]:
        """从 manifest.yaml 读取 capabilities 列表（多能力引擎声明）。

        manifest.yaml 格式：
            capabilities:
              - text-to-video
              - image-to-video
              - first-last-frame-to-video
              - multi-image-to-video
              - multi-reference-to-video

        若未声明 / 读取失败 / 文件不存在，返回空列表（由 _capabilities_impl 降级
        为 [self.capability]）。
        """
        try:
            import yaml
            if not manifest_path.exists():
                logger.warning("manifest.yaml 不存在: %s", manifest_path)
                return []
            with open(manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            caps = data.get("capabilities", [])
            if isinstance(caps, list) and len(caps) > 0:
                logger.info("从 manifest 读取 capabilities: %s", caps)
                return caps
            return []
        except Exception as e:
            logger.warning("读取 manifest capabilities 失败: %s", e)
            return []

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新模型扫描结果（同步 fallback，用于 __init__ 阶段）。

        v0.2.11+ 方案 B：此方法走 fallback 本地扫描逻辑。
        平台 API 异步扫描请调用 `_refresh_scan_async()`（lifespan startup 阶段调用）。
        """
        logger.info("刷新模型扫描（本地 fallback）: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()
        # 同步刷新 LoRA 列表
        self._available_loras = self.scanner.scan_loras()

    async def _refresh_scan_async(self) -> None:
        """v0.2.11+ 方案 B：异步刷新模型扫描结果（优先平台 API）。

        在 lifespan startup 阶段调用，此时平台 API 已就绪。
        平台不可达时 fallback 到本地扫描。
        """
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()
        # 刷新 LoRA 列表
        self._available_loras = self.scanner.scan_loras()

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用模型变体。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            variants.append(VideoVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required,
                available=v.available,
                resolution=v.resolution,
                fps=v.fps,
                duration=v.duration,
                max_duration=v.max_duration,
                supports_audio=v.supports_audio,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=self._available_loras,
            # LTX-2.5 分辨率需被 32 整除（VAE 空间压缩 32×）
            supported_sizes=["960x544", "1280x704", "512x320", "1920x1088"],
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）
            input_schema=VideoInputs.model_json_schema(),
            # v0.4.3+ 多能力声明：manifest.yaml capabilities 列表
            # 旧客户端读 capability（单数）="text-to-video"
            # 新客户端读 capabilities（复数）=["text-to-video", "image-to-video", ...]
            capabilities=self._manifest_capabilities or [self.capability],
        )

    async def load(
        self,
        request: LoadRequest,
    ) -> LoadResponse:
        """POST /load - 非阻塞 fire-and-forget 模式（LTX-2.5 专用）。

        LTX-2.5 模型很大（transformer ~19GB / INT8 ~10GB / GGUF ~20GB），
        同步加载需要数十秒到数分钟。传统同步 /load 会导致：
            1. HTTP 请求长时间挂起（容易超时）
            2. asyncio.to_thread 的工作线程持有 GIL，阻塞事件循环
            3. /health 端点无法响应 → 平台健康检查失败 → 引擎被 kill 重启

        采用 fire-and-forget 架构：
            1. /load 收到请求 → 立即转换状态到 LOADING → 返回 state="loading"
            2. 实际加载在后台 asyncio task 中执行（不阻塞 /load HTTP 响应）
            3. 加载完成 → 状态转为 READY；失败 → 状态转为 ERROR
            4. /health 可以正常响应（即使 GIL 被后台线程持有，事件循环仍能
               在 GIL 释放的间隙处理 /health 的轻量级请求）

        调用方（API/WebUI）应在收到 state="loading" 后轮询 /status 直到
        state="ready" 或 "error"。

        兼容性：保留 _op_lock 并发防护，但在 LOADING 状态下允许 /health
        和 /status 查询。
        """
        async with self._op_lock:
            if self.state_machine.is_busy():
                raise EngineBusyError(
                    f"引擎忙（state={self.state_machine.state.value}），请稍后再试"
                )
            if self.state_machine.state == RuntimeState.INFERRING:
                raise EngineBusyError("推理进行中，无法加载")

            # READY/ERROR 状态时先清理再 load
            current = self.state_machine.state
            if current == RuntimeState.READY:
                logger.info(
                    "load()：当前状态 READY，先 unload 释放资源再加载 variant=%s",
                    request.variant,
                )
                try:
                    await self._unload_impl()
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("load()：unload 失败，强制 reset 状态机: %s", e)
                    try:
                        self.state_machine.transition(RuntimeState.ERROR)
                        self.state_machine.reset()
                    except Exception:
                        pass
            elif current == RuntimeState.ERROR:
                logger.info("load()：当前状态 ERROR，reset 到 IDLE 再加载")
                try:
                    self.state_machine.reset()
                except Exception as e:
                    logger.warning("load()：reset 失败: %s", e)

            # 转换状态到 LOADING
            self.state_machine.transition(RuntimeState.LOADING)

        # 在 _op_lock 外启动后台加载任务（避免长时间持锁阻塞其他操作）
        # 注意：此时 _op_lock 已释放，但 state_machine 已是 LOADING，
        # 其他 load/unload 调用会因 is_busy() 被拒绝
        variant = request.variant or self.default_variant
        if not variant:
            # 没有 variant 参数，也没有 default_variant，转 ERROR
            self.state_machine.transition(RuntimeState.ERROR)
            raise RuntimeError(
                "未指定 variant 且引擎未配置 default_variant，无法加载"
            )

        # 同步预校验 variant 是否存在（不存在的 variant 同步返回 404，
        # 不会等到后台任务才失败）
        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            # 回滚状态机：LOADING → IDLE（因为还没真正开始加载）
            self.state_machine.transition(RuntimeState.IDLE)
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        logger.info(
            "load()：已启动后台加载任务 variant=%s，/load 立即返回 state=loading",
            variant,
        )

        # 创建后台任务（不 await，fire-and-forget）
        asyncio.create_task(self._load_in_background(variant))

        # 立即返回 state="loading"
        return LoadResponse(
            variant=variant,
            state="loading",
            vram_used="0MB",
            load_time=0.0,
        )

    async def _load_in_background(self, variant: str) -> None:
        """后台加载模型权重（fire-and-forget 模式的实际加载逻辑）。

        流程：
            1. 解析 variant 路径（二次校验，防止扫描结果过期）
            2. 解析方案配置（diffusers variant → F16/I8，GGUF variant → G8）
            3. SchemeValidator 校验方案合法性（硬错误抛出，警告记录）
            4. 线程池执行 wrapper.load()（不阻塞事件循环）
            5. 记录方案使用历史到 state store

        加载完成后更新 state_machine：
            - 成功 → READY
            - 失败 → ERROR

        注意：此任务在后台运行，不阻塞 /load 的 HTTP 响应。
        """
        load_start_time = time.time()
        try:
            logger.info("后台加载任务开始: variant=%s", variant)

            variant_path = self.scanner.resolve_variant_path(variant)
            if variant_path is None:
                raise FileNotFoundError(
                    f"variant 不存在: {variant} "
                    f"（可用: {[v.name for v in self._available_variants]}）"
                )

            # 解析方案配置
            scheme = self._resolve_scheme_for_variant(variant)
            if scheme is None:
                raise RuntimeError(
                    f"无法为 variant={variant} 解析方案配置，"
                    f"请检查 schemes/ 目录下的 yaml 文件"
                )

            # 方案合法性校验（显存 + 模型文件）
            validation = self.scheme_validator.validate(
                scheme_id=scheme.scheme,
                variant_name=variant,
            )
            if not validation.ok:
                # 硬错误（方案不存在 / 方案与 variant 不匹配 / 模型目录缺失）→ 抛出
                raise RuntimeError(f"方案校验失败: {validation.errors}")
            if validation.warnings:
                # 警告（显存不足 / 标志文件缺失）→ 记录但继续
                for w in validation.warnings:
                    logger.warning("方案校验警告: %s", w)

            logger.info(
                "开始加载 variant: %s (path=%s, scheme=%s, precision=%s, quant=%s, offload=%s)",
                variant, variant_path, scheme.scheme, scheme.precision,
                scheme.quantization.type, scheme.offload_level,
            )

            # 在线程池中执行同步加载（不阻塞事件循环）
            await asyncio.to_thread(
                self.wrapper.load,
                variant,
                str(variant_path),
                scheme,
                self.config.device,
            )

            load_time = time.time() - load_start_time
            vram_used = self.wrapper.get_vram_used()

            logger.info(
                "后台加载完成: variant=%s, scheme=%s, device=%s, offload=%s, "
                "load_time=%.2fs, vram_used=%.1fMB",
                variant, self.wrapper.loaded_scheme_id, self.wrapper.device,
                self.wrapper.offload_level, load_time, vram_used,
            )

            # 记录方案使用历史
            self.scheme_store.record_history(
                scheme=scheme.scheme, variant=variant, success=True,
            )

            # 加载成功，转换状态到 READY
            # 若加载期间有并发请求将状态改为 ERROR，需先 reset 回 IDLE 再转到 READY
            if self.state_machine.state == RuntimeState.ERROR:
                logger.warning("加载期间状态被并发请求改为 ERROR，reset 后恢复到 READY")
                self.state_machine.reset()  # ERROR → IDLE
                self.state_machine.transition(RuntimeState.LOADING)  # IDLE → LOADING
            self.state_machine.transition(RuntimeState.READY)
            logger.info(
                "状态已转换到 READY: variant=%s, scheme=%s, vram_used=%.1fMB",
                variant, scheme.scheme, vram_used,
            )

        except Exception as e:
            logger.exception("后台加载失败: variant=%s", variant)
            # 记录失败历史
            try:
                self.scheme_store.record_history(
                    scheme=self._default_scheme_id or "?",
                    variant=variant, success=False, error=str(e),
                )
            except Exception:
                pass
            # 加载失败，转换状态到 ERROR
            try:
                self.state_machine.transition(RuntimeState.ERROR)
            except Exception:
                # 状态转换失败，强制 reset
                try:
                    self.state_machine.reset()
                except Exception:
                    pass
            logger.error(
                "后台加载失败，状态已转换到 ERROR: variant=%s, error=%s",
                variant, e,
            )

    def _resolve_scheme_for_variant(self, variant_name: str) -> Optional[SchemeConfig]:
        """根据 variant 名反查适用方案。

        策略（LTX25_ENGINE_INTEGRATION_ANALYSIS.md §7.2 + M4-T4.2）：
            1. GGUF Q4 格式 variant（名含 GGUF 且含 Q4）：
               - 使用方案 G4（唯一适用档）
            2. GGUF 格式 variant（名含 GGUF）：
               - 使用方案 G8（唯一适用档）
            3. diffusers 格式 variant：
               - 优先使用 state store 的默认方案（F16 / I8）
               - 若默认方案不适用或未设置，使用推荐默认档 I8
            4. 兜底：按显存自动选档

        Args:
            variant_name: variant 名称

        Returns:
            SchemeConfig，失败返回 None
        """
        name_upper = variant_name.upper()
        is_gguf = "GGUF" in name_upper
        is_q4 = is_gguf and "Q4" in name_upper

        if is_q4:
            # GGUF Q4 variant：唯一适用方案 G4
            scheme = self.scheme_manager.load_scheme("G4")
            if scheme:
                logger.info("方案反查（GGUF Q4）: 使用方案 G4")
                return scheme
            logger.warning("方案反查（GGUF Q4）: 方案 G4 不存在，按显存自动选档")
            return self.scheme_manager.auto_select_scheme()

        if is_gguf:
            # GGUF Q8 variant：唯一适用方案 G8
            scheme = self.scheme_manager.load_scheme("G8")
            if scheme:
                logger.info("方案反查（GGUF）: 使用方案 G8")
                return scheme
            logger.warning("方案反查（GGUF）: 方案 G8 不存在，按显存自动选档")
            return self.scheme_manager.auto_select_scheme()

        # diffusers variant：优先 state store 默认方案（F16 / I8）
        if self._default_scheme_id in ("F16", "I8"):
            scheme = self.scheme_manager.load_scheme(self._default_scheme_id)
            if scheme and self.scheme_manager.validate_scheme_for_variant(scheme, variant_name):
                logger.info(
                    "方案反查（diffusers）: 使用默认方案 %s（来自 state store）",
                    scheme.scheme,
                )
                return scheme
        # 默认方案不适用或未设置 → 推荐默认档 I8
        scheme = self.scheme_manager.load_scheme("I8")
        if scheme:
            logger.info("方案反查（diffusers）: 使用推荐默认方案 I8")
            return scheme
        # I8 不存在 → 按显存自动选档
        logger.warning("方案反查（diffusers）: 方案 I8 不存在，按显存自动选档")
        return self.scheme_manager.auto_select_scheme()

    async def _load_impl(self, variant: str) -> LoadResponse:
        """POST /load 的实际加载逻辑（已被 _load_in_background 替代）。

        保留此方法是为了兼容 BaseEngine 抽象接口（infer auto-load 链路
        会调用此同步版本）。实际加载逻辑已移至 _load_in_background。
        """
        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        # 解析方案
        scheme = self._resolve_scheme_for_variant(variant)
        if scheme is None:
            raise RuntimeError(f"无法为 variant={variant} 解析方案配置")

        logger.info(
            "开始加载 variant: %s (path=%s, scheme=%s)",
            variant, variant_path, scheme.scheme,
        )

        start_time = time.time()
        try:
            await asyncio.to_thread(
                self.wrapper.load,
                variant,
                str(variant_path),
                scheme,
                self.config.device,
            )
        except Exception as e:
            logger.exception("LTX-2.5 模型加载失败: variant=%s", variant)
            raise RuntimeError(f"LTX-2.5 模型加载失败: {e}") from e

        load_time = time.time() - start_time
        vram_used = self.wrapper.get_vram_used()

        logger.info(
            "加载完成: variant=%s, scheme=%s, device=%s, offload=%s, load_time=%.2fs",
            variant, self.wrapper.loaded_scheme_id, self.wrapper.device,
            self.wrapper.offload_level, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used=f"{vram_used:.1f}MB",
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：卸载模型，释放显存。

        即使 wrapper 未加载模型（如 ERROR 状态下），
        也要确保状态机回到 IDLE，避免 ERROR 状态卡死无法恢复。
        """
        # 先处理状态机：若处于 ERROR 状态，调用 reset() 回到 IDLE
        if self.state_machine.state == RuntimeState.ERROR:
            try:
                self.state_machine.reset()
                logger.info("unload()：状态机从 ERROR 重置为 IDLE")
            except Exception as e:
                logger.warning("unload()：reset 状态机失败: %s", e)

        if not self.wrapper.is_loaded:
            # 确保状态机为 IDLE（可能从 LOADING/READY 来）
            if self.state_machine.state != RuntimeState.IDLE:
                try:
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("unload()：状态转换到 IDLE 失败: %s", e)
            return UnloadResponse(state="idle", vram_freed="0MB")

        vram_freed = self.wrapper.get_vram_used()
        await asyncio.to_thread(self.wrapper.unload)

        # 确保状态机为 IDLE
        if self.state_machine.state != RuntimeState.IDLE:
            try:
                self.state_machine.transition(RuntimeState.IDLE)
            except Exception as e:
                logger.warning("unload()：unload 后状态转换到 IDLE 失败: %s", e)

        logger.info("卸载完成: vram_freed=%.1fMB", vram_freed)
        return UnloadResponse(
            state="idle",
            vram_freed=f"{vram_freed:.1f}MB",
        )

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：视频生成推理（SSE 流式推送进度）。

        流程：
            1. 生成 task_id（若 request.task_id 为 None）
            2. 解析 VideoInputs（从 request.inputs）
            3. 确定生成模式（task_type → t2v / i2v）
            4. i2v 模式加载首帧图（video_utils.load_image）
            5. 上报 start / model_ready / prompt_processing 帧
            6. 解析当前已加载方案（推理默认值来源）
            7. 线程池执行 wrapper.generate()（带心跳）
            8. 保存视频到 output/video/{task_id}.mp4
            9. 上报 completed 帧（含 video_url）

        错误处理：内部捕获异常，推送 error 帧后 return，
        不重新抛出（避免 BaseEngine.infer 把状态置为 ERROR）。
        """
        # ----- 0. 生成 task_id -----
        task_id = request.task_id or f"video_{uuid.uuid4().hex[:16]}"
        reporter = ProgressReporter(stream)

        # ----- 1. 解析输入 -----
        try:
            inputs = VideoInputs(**request.inputs)
        except Exception as e:
            logger.error("VideoInputs 解析失败: %s", e)
            reporter.emit_error(f"输入参数无效: {e}", "invalid_inputs")
            return

        # ----- 2. 上报任务创建到 gateway -----
        reporter.set_task_id(task_id)
        await reporter.init_task_reporter()
        await reporter.report_created(
            request_summary={
                "prompt": inputs.prompt[:200] if inputs.prompt else "",
                "task_type": inputs.task_type,
                "duration": inputs.duration,
                "fps": inputs.fps,
                "size": f"{inputs.width}x{inputs.height}",
            },
            variant=request.variant or "",
            task_type="video_generation",  # 任务中心筛选语义
        )

        # ----- 3. 上报开始阶段 -----
        reporter.emit_start(task_id, inputs.prompt)
        reporter.emit_model_ready(
            self.wrapper.loaded_variant or "unknown",
            self.wrapper.device,
        )

        # ----- 4. 模式参数校验 -----
        mode = inputs.task_type
        if mode not in VIDEO_MODES:
            reporter.emit_error(
                f"未知生成模式: {mode}（支持: {' / '.join(VIDEO_MODES)}）",
                "invalid_mode",
            )
            return

        # ----- 5. 多图统一加载（所有需要图片的 mode 都走这里）-----
        # 先算 num_frames（role_to_frame_idx 需要）
        if inputs.num_frames is not None:
            num_frames = inputs.num_frames
        else:
            # duration × fps 对齐 8k+1（与 wrapper._align_num_frames 同款）
            target = inputs.duration * inputs.fps
            k = max(0, round((target - 1) / 8))
            num_frames = k * 8 + 1

        # 各 mode 需要的图片数量范围
        min_n, max_n = MODE_IMAGE_COUNT.get(mode, (0, 0))
        images_out: list[ImageInput] = []

        if min_n > 0:
            # mode 需要图片 → 校验数量
            if not request.images:
                reporter.emit_error(
                    f"{mode} 模式需要图片输入（{min_n}~{max_n} 张，通过 images 字段传入）",
                    "missing_image",
                )
                return
            actual = len(request.images)
            if actual < min_n or actual > max_n:
                reporter.emit_error(
                    f"{mode} 模式需要 {min_n}~{max_n} 张图，收到 {actual} 张",
                    "invalid_image_count",
                )
                return

            # role → frame_idx 映射（只算时间位置，strength 看用户显式覆盖）
            for i, img_ref in enumerate(request.images):
                url_or_path = img_ref.url or img_ref.path
                if not url_or_path:
                    reporter.emit_error(
                        f"第 {i} 张图片缺少 url / path 字段",
                        "invalid_image",
                    )
                    return
                # role 从 SDK ImageInput.role 取（Gateway 会把 role 直接放 ImageInput.role 里）
                role = img_ref.role or "reference"
                frame_idx, default_strength = role_to_frame_idx(role, i, actual, num_frames)
                # 用户显式传的 strength / crf 优先；None 时走 role 默认值
                strength = img_ref.strength if img_ref.strength is not None else default_strength
                crf = img_ref.crf  # None = 模型训练默认值
                try:
                    pil_img = load_image(url_or_path)
                except Exception as e:
                    logger.exception("第 %d 张图加载失败: %s", i, url_or_path)
                    reporter.emit_error(f"第 {i} 张图加载失败: {e}", "image_load_failed")
                    return
                images_out.append(ImageInput(
                    pil=pil_img,
                    path=url_or_path,
                    role=role,
                    frame_idx=frame_idx,
                    strength=strength,
                    crf=crf,
                ))
                logger.info(
                    "多图条件 #%d: role=%s, frame_idx=%d, strength=%.2f(crf=%s), path=%s",
                    i, role, frame_idx, strength, crf, url_or_path,
                )

        # 汇总日志
        if images_out:
            logger.info(
                "多图条件加载完成: mode=%s, 共 %d 张, frame_idxs=%s",
                mode, len(images_out), [img.frame_idx for img in images_out],
            )

        # ----- 6. 提示词处理（LTX-2.5 无 Context-IR，直通）-----
        reporter.emit_prompt_processing(inputs.prompt)

        # ----- 7. 解析当前已加载方案 -----
        # 推理参数优先级：inputs 显式传入 > scheme.inference 默认 > wrapper 默认
        # offload 级别由 scheme 决定（加载时固定），inputs.cpu_offload_level 不再生效
        current_scheme: Optional[SchemeConfig] = None
        if self.wrapper.loaded_scheme_id:
            current_scheme = self.scheme_manager.load_scheme(self.wrapper.loaded_scheme_id)
            if current_scheme:
                logger.info(
                    "推理使用已加载方案: %s（steps=%d, cfg=%.2f, resolution=%s）",
                    current_scheme.scheme,
                    current_scheme.inference.num_inference_steps,
                    current_scheme.inference.guidance_scale,
                    current_scheme.inference.default_resolution,
                )

        # inputs.cpu_offload_level 与实际方案不一致时仅警告（offload 加载时已固定）
        if (
            inputs.cpu_offload_level != "auto"
            and current_scheme
            and inputs.cpu_offload_level != current_scheme.offload_level
        ):
            logger.warning(
                "inputs.cpu_offload_level=%s 与已加载方案 %s 的 offload=%s 不一致，"
                "offload 级别在加载时已固定，本次推理仍使用 %s"
                "（如需切换请卸载后修改方案再加载）",
                inputs.cpu_offload_level, current_scheme.scheme,
                current_scheme.offload_level, current_scheme.offload_level,
            )

        # ----- 8. 线程池执行推理（带心跳）-----
        def _generate() -> dict:
            return self.wrapper.generate(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                duration=inputs.duration,
                fps=inputs.fps,
                width=inputs.width,
                height=inputs.height,
                num_frames=inputs.num_frames,
                num_inference_steps=inputs.num_inference_steps,
                guidance_scale=inputs.guidance_scale,
                seed=inputs.seed,
                images=images_out if images_out else None,
                auto_duration=inputs.auto_duration,
                scheme=current_scheme,
                temporal_upscalings=getattr(inputs, "temporal_upscalings", 0),
                workflow=getattr(inputs, "workflow", None),
            )

        # 心跳任务：generate() 是同步阻塞调用，期间定期推送心跳帧，
        # 重置 gateway 的 read timeout 计时。
        async def _heartbeat():
            elapsed = 0
            while True:
                await asyncio.sleep(30)
                elapsed += 30
                progress = min(0.80, 0.15 + elapsed * 0.005)
                reporter.emit_generating(progress, elapsed=elapsed)

        reporter.emit_generating(0.15, message="开始扩散去噪")
        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            result = await asyncio.to_thread(_generate)
        except Exception as e:
            logger.exception("LTX-2.5 推理失败")
            reporter.emit_error(f"推理失败: {e}", "infer_failed")
            return
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # ----- 9. VAE 解码（推理已包含，此处只上报进度）-----
        reporter.emit_vae_decoding()

        # ----- 10. 保存视频 -----
        reporter.emit_saving()
        # 构造输出目录：{output_root}/video/（save_video 内部会自动创建）
        output_dir = self.config.output_root / "video"

        try:
            # 音频采样率：G8 原生链路解码的 vocoder 实际输出（T3.2，48kHz），
            # 无音频时用默认值（save_video 不读）
            audio_data = result.get("audio_data")
            sample_rate = result.get("metadata", {}).get("audio_sample_rate") or 48000
            video_result = await asyncio.to_thread(
                save_video,
                result["video_frames"],
                audio_data,
                output_dir,
                task_id,
                inputs.fps,
                sample_rate,
            )
        except Exception as e:
            logger.exception("保存视频失败")
            reporter.emit_error(f"保存视频失败: {e}", "save_failed")
            return

        # ----- 10.5 推送合成警告（音视频时长不匹配等）-----
        mux_warnings = video_result.get("warnings", [])
        for w in mux_warnings:
            logger.warning("视频合成警告: %s", w)
            await reporter.emit_mux_warning(w)

        # ----- 11. 上报完成 -----
        metadata = result.get("metadata", {})
        final_result = {
            "video_url": video_result["video_url"],
            "video_only_url": video_result["video_only_url"],
            "audio_url": video_result["audio_url"],
            "duration": video_result["duration"],
            "fps": video_result["fps"],
            "mode": mode,
            "variant": self.wrapper.loaded_variant,
            "scheme_id": self.wrapper.loaded_scheme_id,
            "device": self.wrapper.device,
            "offload_level": self.wrapper.offload_level,
            "seed": metadata.get("seed", inputs.seed),
            "num_inference_steps": metadata.get("num_inference_steps"),
            "is_placeholder": metadata.get("is_placeholder", False),
            "warnings": mux_warnings,
        }

        logger.info(
            "视频生成完成: task_id=%s, mode=%s, duration=%.2fs, video_url=%s",
            task_id, mode, video_result["duration"], video_result["video_url"],
        )

        await reporter.emit_completed(task_id, final_result)

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        vram_used = self.wrapper.get_vram_used()
        # device_name 给 UI 显示友好标签（GPU (CUDA) / CPU），
        # device 仍保留原始字符串（cuda:0 / cpu）供程序使用
        raw_device = self.wrapper.device
        if raw_device.startswith("cuda"):
            device_name = "GPU (CUDA)"
        elif raw_device == "cpu":
            device_name = "CPU"
        else:
            device_name = raw_device
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.wrapper.loaded_variant,
            vram_used=f"{vram_used:.1f}MB",
            loaded_loras=[],  # v1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": raw_device,
                "device_name": device_name,
                "offload_level": self.wrapper.offload_level,
                "is_loaded": self.wrapper.is_loaded,
                "available_loras": self._available_loras,
                # 方案信息
                "loaded_scheme_id": self.wrapper.loaded_scheme_id,
                "default_scheme_id": self._default_scheme_id,
                "available_schemes": [
                    {
                        "id": s.scheme,
                        "name": s.name,
                        "description": s.description,
                        "vram_required": s.vram_required,
                        "precision": s.precision,
                        "quantization_type": s.quantization.type,
                    }
                    for s in self.scheme_manager.list_schemes()
                ],
            },
        )

    # ============ 方案管理 API ============

    def list_schemes(self) -> list[dict]:
        """列出所有可用方案（供 UI / 平台 API 调用）。

        Returns:
            方案信息列表（每项含 id / name / description / vram_required /
            precision / quantization_type / size_gb / offload_level /
            model_repo / applicable_variants）
        """
        return [
            {
                "id": s.scheme,
                "name": s.name,
                "description": s.description,
                "vram_required": s.vram_required,
                "precision": s.precision,
                "quantization_type": s.quantization.type,
                "size_gb": s.size_gb,
                "offload_level": s.offload_level,
                "model_repo": s.model_repo,
                "applicable_variants": s.applicable_variants,
            }
            for s in self.scheme_manager.list_schemes()
        ]

    def get_default_scheme(self) -> Optional[str]:
        """获取当前默认方案 ID。"""
        return self._default_scheme_id

    def set_default_scheme(self, scheme_id: str) -> bool:
        """设置默认方案并持久化。

        Args:
            scheme_id: 方案 ID（F16 / I8 / G8）

        Returns:
            是否设置成功
        """
        scheme = self.scheme_manager.load_scheme(scheme_id)
        if scheme is None:
            logger.warning("设置默认方案失败: 方案 %s 不存在", scheme_id)
            return False
        self._default_scheme_id = scheme_id
        self.scheme_store.set_default_scheme(scheme_id)
        logger.info("默认方案已设置并持久化: %s（%s）", scheme_id, scheme.name)
        return True

    # ============ LoRA（v1 预留）============

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：LTX-2.5 v1 不支持 LoRA（M2 扩展项）。"""
        raise RuntimeError(
            "LTX-2.5 v1 不支持 LoRA（预留接口，后续版本扩展）"
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：LTX-2.5 v1 不支持 LoRA（M2 扩展项）。"""
        raise RuntimeError(
            "LTX-2.5 v1 不支持 LoRA（预留接口，后续版本扩展）"
        )
