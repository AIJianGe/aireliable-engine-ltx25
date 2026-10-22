"""LTX-2.5 G8 档（GGUF Q8_0）ltxv 原生推理模块。

从 ANALYSIS-TASKS/g8_prototype.py（原型已验证输出 mp4 成功）逐行等价移植，
仅做工程化包装：
  - 原型全局变量（GGUF_PATH / GEMMA_PATH / DEVICE / DTYPE / cleanup / logger）
    → LTXVNativeRunner 实例属性 / GGUFTransformerBuilder 构造参数
  - ltx_pipelines / ltx_core / gguf / safetensors 的 import 全部保持在方法内部
    （与原型一致，避免引擎启动时重依赖）

基于 ltx_pipelines 1.3.0 blocks 架构（DistilledPipeline 同款组件复用）：
  DiffusionStage          ← 自定义 GGUFTransformerBuilder（ModelBuilderProtocol）
  VideoUpsampler          ← conv VAE checkpoint + spatial upsampler（标准 Builder）
  VideoDecoder / AudioDecoder / encode_video ← 标准块

组件与文件映射（全部已确认方言兼容）：
  transformer          gguf_format/*Q8_0*.gguf（ltx_core 方言 config 内嵌于元数据）
                       - 4090/4186 模型键匹配；96 个 ff 零 bias 补零
                       - video/audio_embeddings_connector.* 259 键喂给 EmbeddingsProcessor
  GemmaTextEncoder     comfyui_format/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
                       （ComfyUI 打包 TE：comfy-flat 键 + 内嵌 config/tokenizer/processor 资产，
                        官方 get_gemma_ops 处理键映射 + tied lm_head + buffer 填充）
  EmbeddingsProcessor  connectors 从 GGUF 提取 + feature_extractor 从 gemma-with-proj 提取
  video VAE (conv)     comfyui_format/vae/ltx-2.5-video-vae-conv-bf16.safetensors
  audio VAE + vocoder  comfyui_format/vae/ltx-2.5-audio-vae-bf16.safetensors
  spatial upsampler    comfyui_format/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

硬件约束：Quadro RTX 8000（Turing SM75，48GB）
  - 不支持 BF16 → 全部 FP16
  - blocks 分时加载：Gemma(24GB) → 释放 → transformer(44GB) → 释放 → VAE 解码

禁止兜底下载（project-rules.md 第二部分 §2）：
  本模块只做加载与推理，不下载。文件缺失由上层（wrapper._load_gguf_branch）
  校验并指引用户到模型管理界面下载。
"""

from __future__ import annotations

import gc
import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# 文件完整性阈值：小于 1MB 视为残缺文件（断点续传残片）
_MIN_FILE_BYTES = 1 * 1024 * 1024

# 两阶段蒸馏管线的分辨率/帧数约束（generate 入口对齐用）
_SPATIAL_ALIGNMENT = 64   # width/height 需 64 整除（Stage 1 半分辨率后仍满足 32 整除）
_TEMPORAL_ALIGNMENT = 8  # num_frames 需 8k+1（VAE 时间压缩 8×）


def _remap_diffusers_attention_sd(sd: dict) -> dict:
    """把 diffusers 导出的分离注意力投影重组为 nn.MultiheadAttention 布局。

    Lightricks diffusers 仓库的 duration_head checkpoint 把 MHA 拆为
    to_q / to_k / to_v / to_out 四个 Linear；ltx_core 的 DurationHead 用
    nn.MultiheadAttention（in_proj_weight / in_proj_bias / out_proj.* 合并布局）。
    这里把四块投影重组成 MHA 布局，键不完整时保持原样（交由 strict=True 报错）。
    """
    split_suffixes = (
        ".to_q.weight", ".to_q.bias", ".to_k.weight", ".to_k.bias",
        ".to_v.weight", ".to_v.bias", ".to_out.weight", ".to_out.bias",
    )
    out = {}
    pieces: dict[str, dict[str, "torch.Tensor"]] = {}
    for key, tensor in sd.items():
        if key.endswith(split_suffixes):
            base, _proj, _param = key.rsplit(".", 2)
            pieces.setdefault(base, {})[key[len(base) + 1:]] = tensor
        else:
            out[key] = tensor
    for base, p in pieces.items():
        if not {"to_q.weight", "to_k.weight", "to_v.weight"} <= p.keys():
            continue  # 键不完整，保持原样交给 strict 加载报错
        out[f"{base}.cross_attn.in_proj_weight"] = torch.cat(
            [p["to_q.weight"], p["to_k.weight"], p["to_v.weight"]], dim=0,
        )
        if {"to_q.bias", "to_k.bias", "to_v.bias"} <= p.keys():
            out[f"{base}.cross_attn.in_proj_bias"] = torch.cat(
                [p["to_q.bias"], p["to_k.bias"], p["to_v.bias"]], dim=0,
            )
        if "to_out.weight" in p:
            out[f"{base}.cross_attn.out_proj.weight"] = p["to_out.weight"]
        if "to_out.bias" in p:
            out[f"{base}.cross_attn.out_proj.bias"] = p["to_out.bias"]
    return out


# =============================================================================
# 1. GGUF 读取（元数据 + 惰性反量化）—— 逐行移植自原型
# =============================================================================
def load_gguf_config(gguf_path: Path) -> dict:
    """从 GGUF 元数据读取 ltx_core 方言 config（含 "transformer" 子字典）。"""
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    for field in reader.fields.values():
        if field.name == "config":
            return json.loads(bytes(field.parts[field.data[0]]).decode("utf-8"))
    raise RuntimeError("GGUF 元数据中缺少 config")


def iter_gguf_tensors(gguf_path: Path, name_filter=None):
    """惰性遍历 GGUF 张量（名称, 反量化后的 torch tensor）。

    name_filter: 可选的键名过滤器（在反量化前应用，避免无关张量的反量化开销）。
    """
    import gguf
    from gguf import GGMLQuantizationType

    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if name_filter is not None and not name_filter(tensor.name):
            continue
        ttype = tensor.tensor_type
        if ttype == GGMLQuantizationType.F32:
            data = tensor.data.view(np.float32)
            shape = data.shape
        elif ttype == GGMLQuantizationType.F16:
            data = tensor.data.view(np.float16).astype(np.float32)
            shape = data.shape
        else:
            block_size, type_size = gguf.GGML_QUANT_SIZES[ttype]
            raw = tensor.data.reshape(-1, type_size)
            shape = gguf.quant_shape_from_byte_shape(tensor.data.shape, ttype)
            data = gguf.dequantize(raw, ttype).reshape(shape)
        yield tensor.name, torch.from_numpy(np.ascontiguousarray(data)).reshape(shape)


def gguf_checkpoint_metadata(gguf_path: Path, gemma_path: Path) -> dict:
    """构造 ltx_core checkpoint 元数据：GGUF config + 注入 gemma_source_checkpoint。

    GGUF 量化丢失了 gemma_source_checkpoint，按 gemma 文件内嵌 HF config 的
    gemma_version 注入（EmbeddingsProcessorConfigurator 的版本检查要求）。
    """
    from ltx_core.text_encoders.gemma import GemmaAssets

    gemma_version = GemmaAssets.load(str(gemma_path)).config_dict.get("gemma_version")
    meta = {"config": load_gguf_config(gguf_path)}
    if gemma_version:
        meta["gemma_source_checkpoint"] = {"gemma_version": gemma_version}
    return meta


# =============================================================================
# 2. GGUF Transformer Builder（实现 ModelBuilderProtocol，供 DiffusionStage 使用）
# =============================================================================
class GGUFTransformerBuilder:
    """从 ComfyUI 格式 GGUF 构建 LTXModel 的 ModelBuilderProtocol 实现。

    - meta model 由 GGUF 内嵌 config 构建（LTXModelConfigurator.from_metadata）
    - 逐 tensor 反量化直接写 GPU（避免 CPU 侧 44GB 峰值）
    - 首次反量化后写 fp16 磁盘缓存（.cache.bin + .cache.json），
      后续 build()（DiffusionStage 每个 stage 调用都会重建 transformer）
      走 mmap 快速加载，避免重复 32 分钟的 Python 反量化
    - 模型缺失的 ff bias 补零（GGUF 量化时删除了全零张量）
    - GGUF 中的 connectors 键（259 个）不属于 LTXModel，忽略

    与原型的差异（仅工程化，算法不变）：
    - 原型全局 GEMMA_PATH / DEVICE / logger → 构造参数（gemma_path / device / logger）
    - 磁盘缓存 offset 必须用元素单位（fp16 元素数），原型已修复，此处保持
    """

    def __init__(
        self,
        gguf_path: Path,
        dtype: torch.dtype,
        gemma_path: Path,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from ltx_core.loader.registry import ModelRegistry

        self._gguf_path = str(gguf_path)
        self._dtype = dtype
        self._gemma_path = Path(gemma_path)
        self._device = device
        self._logger = logger or logging.getLogger(__name__)
        self._registry = ModelRegistry(cache_models=True, cache_weights=False)
        self._metadata: dict | None = None
        self._meta_model = None
        self._cache_bin = Path(self._gguf_path).with_suffix(".fp16.cache.bin")
        self._cache_json = Path(self._gguf_path).with_suffix(".fp16.cache.json")

    # ---- fp16 磁盘缓存 ----
    def _cache_valid(self) -> dict | None:
        """缓存有效则返回索引 dict（entries: name -> [offset, numel, *shape]），否则 None。"""
        if not (self._cache_bin.exists() and self._cache_json.exists()):
            return None
        try:
            index = json.loads(self._cache_json.read_text(encoding="utf-8"))
            st = Path(self._gguf_path).stat()
            if index.get("source_size") != st.st_size or index.get("source_mtime") != int(st.st_mtime):
                return None
            if self._cache_bin.stat().st_size != index["total_bytes"]:
                return None
            return index
        except (OSError, ValueError, KeyError):
            return None

    def _load_from_cache(self, model_keys: dict, device: torch.device) -> tuple[dict, int]:
        """从磁盘缓存加载 state dict，返回 (sd, 加载键数)。"""
        index = self._cache_valid()
        assert index is not None
        mm = np.memmap(self._cache_bin, dtype=np.float16, mode="r")
        sd: dict[str, torch.Tensor] = {}
        for name, (offset, numel, *shape) in index["entries"].items():
            if name not in model_keys:
                continue
            arr = mm[offset : offset + numel].reshape(shape)
            sd[name] = torch.from_numpy(arr.copy()).to(device)
        return sd, len(sd)

    def _write_cache(self, sd: dict[str, torch.Tensor]) -> None:
        """把 GPU 上的 fp16 sd 逐 tensor 写入磁盘缓存（单 tensor CPU 峰值 ~200MB）。"""
        entries: dict[str, list] = {}
        total_bytes = 0
        with open(self._cache_bin, "wb") as f:
            for name, tensor in sd.items():
                cpu = tensor.detach().to("cpu").contiguous().numpy()
                assert cpu.dtype == np.float16, f"缓存要求 fp16，得到 {cpu.dtype}"
                # offset 以 fp16 元素为单位（读取端 np.memmap 直接做下标），
                # 不能记字节数，否则读取端 mm[offset:...] 会错位
                entries[name] = [total_bytes // 2, int(cpu.size), *cpu.shape]
                f.write(cpu.tobytes())
                total_bytes += cpu.nbytes
        st = Path(self._gguf_path).stat()
        self._cache_json.write_text(
            json.dumps({
                "source": self._gguf_path,
                "source_size": st.st_size,
                "source_mtime": int(st.st_mtime),
                "dtype": "float16",
                "total_bytes": total_bytes,
                "entries": entries,
            }),
            encoding="utf-8",
        )
        self._logger.info("fp16 磁盘缓存已写入：%s（%.1f GB）", self._cache_bin, total_bytes / 1e9)

    # ---- ModelBuilderProtocol 表面 ----
    @property
    def checkpoint(self) -> str:
        return self._gguf_path

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        return True

    @property
    def model_sd_ops(self):
        return None

    @property
    def module_ops(self) -> tuple:
        return ()

    @property
    def loras(self) -> tuple:
        return ()

    @property
    def registry(self):
        return self._registry

    def with_registry(self, registry):
        return self

    def with_sd_ops(self, sd_ops):
        return self

    def with_module_ops(self, module_ops):
        return self

    def with_loras(self, loras):
        return self

    def with_lora_load_device(self, device):
        return self

    def with_fuse_rule(self, fuse_rule):
        return self

    def model_config(self) -> dict:
        if self._metadata is None:
            self._metadata = gguf_checkpoint_metadata(Path(self._gguf_path), self._gemma_path)
        return self._metadata["config"]

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        from ltx_core.model.transformer import LTXModelConfigurator

        device = device or self._device
        target_dtype = dtype or self._dtype
        if self._metadata is None:
            self._metadata = gguf_checkpoint_metadata(Path(self._gguf_path), self._gemma_path)

        t0 = time.time()
        with torch.device("meta"):
            model = LTXModelConfigurator.from_metadata(self._metadata)
        model_keys = {k: v for k, v in model.state_dict().items()}

        sd: dict[str, torch.Tensor] = {}
        if self._cache_valid() is not None:
            sd, loaded = self._load_from_cache(model_keys, device)
            source = "磁盘缓存"
        else:
            loaded = 0
            for name, tensor in iter_gguf_tensors(Path(self._gguf_path), name_filter=lambda n: n in model_keys):
                sd[name] = tensor.to(target_dtype).to(device)
                loaded += 1
            source = "GGUF 反量化"
            self._write_cache(sd)

        # 补零缺失键（ff.net.*.bias，ff_bias=false 时全零）
        for k, meta_v in model_keys.items():
            if k not in sd:
                sd[k] = torch.zeros(meta_v.shape, dtype=target_dtype, device=device)

        model.load_state_dict(sd, strict=False, assign=True)
        del sd
        model.eval()
        self._logger.info(
            "transformer 加载完成（%s）：%d 个键 + %d 个补零键，耗时 %.0fs",
            source, loaded, len(model_keys) - loaded, time.time() - t0,
        )
        return model


# =============================================================================
# 3. ltx_core 兼容 patch（逐行移植自原型）
# =============================================================================
def _patch_ltx_core_rotary() -> None:
    """修复 ltx_core _populate_rotary_v5 与 transformers 5.x 分层配置的兼容问题。

    Gemma4 unified 的 head_dim 是 per-layer 属性（full_attention 层比 local 层宽），
    直接访问全局 config.head_dim 会抛 AmbiguousGlobalPerLayerAttributeError。
    按 transformers 自身 RotaryEmbedding.__init__ 的做法（modeling_gemma4_unified.py
    L221-224），改用 config.per_layer_config[layer_type] 解析后的配置传给 rope 函数。
    """
    import ltx_core.text_encoders.gemma.encoders.encoder_configurator as _enc

    if getattr(_enc, "_rotary_patched_by_proto", False):
        return

    def _populate_rotary_v5(l_model, config):
        rope_emb = l_model.rotary_emb
        for layer_type in dict.fromkeys(config.layer_types):
            rope_params = config.rope_parameters[layer_type]
            if rope_params is None:
                continue
            rope_type = rope_params["rope_type"]
            # view 自带正确的 per-layer head_dim（full_attention=512，sliding=256）；
            # 不能传 head_dim_key='global_head_dim'——view 上该属性已被 heterogeneity
            # spec 移出 __dict__（getattr 返回 None → 错误 fallback 到 240）
            cfg = config.per_layer_config[layer_type]
            if rope_type == "default":
                inv_freq, attn_scaling = rope_emb.compute_default_rope_parameters(cfg, layer_type=layer_type)
            else:
                inv_freq, attn_scaling = _enc.ROPE_INIT_FUNCTIONS[rope_type](cfg, layer_type=layer_type)
            rope_emb.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            rope_emb.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(rope_emb, f"{layer_type}_attention_scaling", attn_scaling)

    _enc._populate_rotary_v5 = _populate_rotary_v5
    _enc._rotary_patched_by_proto = True


# =============================================================================
# 4. LTXVNativeRunner（原型全局流程的类封装，G8 档推理入口）
# =============================================================================
class LTXVNativeRunner:
    """LTX-2.5 G8/G4 档（GGUF Q8_0 / Q4_K_M）ltxv 原生推理 runner。

    - __init__ / check_files：只做路径推导与文件校验，不加载模型
      （blocks 分时加载架构，模型在 generate() 时按阶段构建与释放）
    - generate：执行原型 run_t2v 两阶段蒸馏流程（逐行等价）
    - unload：释放持有引用并清理显存
    - G8/G4 档共用同一管线，仅 gguf_glob 匹配的权重文件不同
      （Q8_0 23.6GB / Q4_K_M 15.7GB，反量化均走 gguf.dequantize 通用接口）

    硬件约束（Quadro RTX 8000，Turing SM75，48GB）：
    blocks 分时加载：Gemma(24GB) → 释放 → transformer(44GB) → 释放 → VAE 解码，
    每阶段结束 gc + empty_cache，确保下一阶段有足够显存。
    """

    # model_dir 下的期望文件布局（相对路径）
    _GGUF_DIRNAME = "gguf_format"
    _GGUF_GLOB = "*Q8_0*.gguf"
    _GGUF_FALLBACK_NAME = "transformer-Q8_0.gguf"  # 目录为空时的期望名（供 check_files 报告）
    _GEMMA_RELPATH = Path("comfyui_format") / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    _VIDEO_VAE_RELPATH = Path("comfyui_format") / "vae" / "ltx-2.5-video-vae-conv-bf16.safetensors"
    _AUDIO_VAE_RELPATH = Path("comfyui_format") / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    _UPSAMPLER_RELPATH = Path("comfyui_format") / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    # duration_head（可选，T3.3 自动时长）：diffusers 格式共享组件，文件缺失时降级手动时长
    _DURATION_HEAD_DIR_RELPATH = Path("diffusers_format") / "duration_head"

    def __init__(
        self,
        model_dir: Path,
        device: str,
        dtype,
        logger: Optional[logging.Logger] = None,
        gguf_glob: str = _GGUF_GLOB,
        gguf_fallback_name: str = _GGUF_FALLBACK_NAME,
    ):
        """初始化 runner（只推导路径，不加载模型）。

        Args:
            model_dir: LTX-2.5 模型目录（models/video/ltx25/，
                含 gguf_format/ 与 comfyui_format/ 子目录）
            device: 推理设备（"cuda" / "cuda:0" / "cpu"）
            dtype: 推理 dtype（Turing SM75 无 BF16，应为 torch.float16）
            logger: 可选的日志器
            gguf_glob: GGUF 权重文件匹配模式（G8 档 "*Q8_0*.gguf" /
                G4 档 "*Q4_K_M*.gguf"，默认 Q8_0）
            gguf_fallback_name: 目录无匹配时报告缺失用的期望文件名
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self._logger = logger or logging.getLogger(__name__)

        # GGUF：按方案量化类型严格匹配（*Q8_0* / *Q4_K_M*；落地名与
        # 仓库源文件名均含量化标签，glob 均可命中）。无匹配时落到期望名
        # （供 check_files 报告缺失，指引用户下载对应量化权重；
        # 不回退任意 .gguf——Q8/Q4 共存时回退会错配权重）
        gguf_dir = self.model_dir / self._GGUF_DIRNAME
        gguf_candidates = sorted(gguf_dir.glob(gguf_glob))
        self.gguf_path = gguf_candidates[0] if gguf_candidates else gguf_dir / gguf_fallback_name
        self.gemma_path = self.model_dir / self._GEMMA_RELPATH
        self.video_vae_path = self.model_dir / self._VIDEO_VAE_RELPATH
        self.audio_vae_path = self.model_dir / self._AUDIO_VAE_RELPATH
        self.upsampler_path = self.model_dir / self._UPSAMPLER_RELPATH
        # DFRPipeline 可选组件目录（manifest model_files target_dir）
        self._dfr_dir = self.model_dir / "dfr_format"
        self._dfr_adapter = None  # 懒加载，首次需要 DFR 路由时构建

    # ===== 文件完整性 =====

    def check_files(self) -> list[str]:
        """校验全部依赖文件（存在性 + 大小 > 1MB），返回缺失/残缺文件路径列表。

        不抛错、不下载（遵循 project-rules.md 第二部分 §2：
        文件缺失由上层抛 RuntimeError 指引用户到模型管理界面下载）。
        """
        missing: list[str] = []
        for p in (
            self.gguf_path,
            self.gemma_path,
            self.video_vae_path,
            self.audio_vae_path,
            self.upsampler_path,
        ):
            if not p.exists() or p.stat().st_size <= _MIN_FILE_BYTES:
                missing.append(str(p))
        return missing

    @property
    def is_ready(self) -> bool:
        """全部依赖文件就绪（check_files() 为空）。"""
        return not self.check_files()

    # ===== 显存清理（原型 cleanup 移植）=====

    def _cleanup(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self._logger.info(
                "VRAM: %.1f GB free / %.1f GB",
                torch.cuda.mem_get_info()[0] / 1e9, torch.cuda.mem_get_info()[1] / 1e9,
            )

    # ===== 文本编码（原型 build_text_encoder / build_embeddings_processor /
    #       encode_prompt 移植，全局路径变量改为实例属性）=====

    def _build_text_encoder(self):
        """从 ComfyUI 打包 TE 单文件构建 GemmaTextEncoder（官方 Builder 路径）。

        get_gemma_ops 自动处理：comfy-flat 键映射、tied lm_head 复制、
        rotary/embed_scale buffer 填充、tokenizer/processor 资产附加。
        """
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.text_encoders.gemma import get_gemma_ops
        from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GemmaTextEncoderConfigurator

        _patch_ltx_core_rotary()
        from ltx_core.loader.registry import ModelRegistry

        gemma_sd_ops, gemma_module_ops = get_gemma_ops(str(self.gemma_path))
        builder = SingleGPUModelBuilder(
            model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(self.gemma_path)),
            model_path=(str(self.gemma_path),),
            model_sd_ops=gemma_sd_ops,
            module_ops=gemma_module_ops,
            # 关键：禁用 registry 缓存（默认 cache_models=True 会持有模型，
            # del encoder 后 ~24GB VRAM 不释放，导致后续 transformer 加载 OOM）
            registry=ModelRegistry(cache_models=False, cache_weights=False),
        )
        encoder = builder.build(device=self.device, dtype=self.dtype).eval()
        self._logger.info("GemmaTextEncoder 构建完成")
        return encoder

    def _build_embeddings_processor(self):
        """构建 EmbeddingsProcessor（权重跨两个文件，需自定义加载）。

        - 结构：EmbeddingsProcessorConfigurator.from_metadata(GGUF metadata + gemma_version 注入)
        - 权重：
          * video_connector.*   <- GGUF video_embeddings_connector.*
          * audio_connector.*   <- GGUF audio_embeddings_connector.*
          * feature_extractor.* <- gemma-with-proj 的 text_embedding_projection.*
        """
        import safetensors

        from ltx_core.text_encoders.gemma import EmbeddingsProcessorConfigurator

        metadata = gguf_checkpoint_metadata(self.gguf_path, self.gemma_path)
        with torch.device("meta"):
            processor = EmbeddingsProcessorConfigurator.with_gemma_model_path(str(self.gemma_path)).from_metadata(metadata)
        model_keys = set(processor.state_dict().keys())

        sd: dict[str, torch.Tensor] = {}
        # 1) connectors from GGUF
        def _connector_filter(n: str) -> bool:
            return n.startswith(("video_embeddings_connector.", "audio_embeddings_connector."))

        for name, tensor in iter_gguf_tensors(self.gguf_path, name_filter=_connector_filter):
            if name.startswith("video_embeddings_connector."):
                mapped = name.replace("video_embeddings_connector.", "video_connector.", 1)
            else:
                mapped = name.replace("audio_embeddings_connector.", "audio_connector.", 1)
            if mapped in model_keys:
                sd[mapped] = tensor.to(self.dtype).to(self.device)

        # 2) feature_extractor from gemma-with-proj
        with safetensors.safe_open(str(self.gemma_path), framework="pt") as f:
            for name in f.keys():
                if not name.startswith("text_embedding_projection."):
                    continue
                mapped = name.replace("text_embedding_projection.", "feature_extractor.", 1)
                if mapped in model_keys:
                    sd[mapped] = f.get_tensor(name).to(self.dtype).to(self.device)

        missing = model_keys - set(sd.keys())
        if missing:
            self._logger.warning("EmbeddingsProcessor 缺失 %d 键：%s", len(missing), sorted(missing)[:10])
        processor.load_state_dict(sd, strict=False, assign=True)
        del sd
        processor.eval().to(self.device)
        self._logger.info("EmbeddingsProcessor 加载完成：%d/%d 键", len(model_keys) - len(missing), len(model_keys))
        return processor

    def _encode_prompt(self, prompt: str) -> tuple:
        """完整文本编码流程：Gemma hidden states → EmbeddingsProcessor → (video_context, audio_context)。

        关键：必须 no_grad 包裹。ltx_core 的 encode() 内部没有停用 autograd，forward 会
        构建计算图（Linear 反向保存权重矩阵），返回的 hidden_states 使整张图存活 →
        整个 Gemma 权重（~42GB）被钉在显存，del encoder 也无法释放。
        """
        encoder = self._build_text_encoder()
        with torch.no_grad():
            (hidden_states, attention_mask), = encoder.encode([prompt])
        del encoder
        self._cleanup()

        processor = self._build_embeddings_processor()
        with torch.no_grad():
            out = processor.process_hidden_states(hidden_states, attention_mask)
        del processor
        self._cleanup()
        return out.video_encoding, out.audio_encoding

    # ===== duration_head 自动时长（T3.3，可选开关）=====

    def _predict_num_frames(
        self,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        fps: float,
        max_seconds: float = 10.0,
    ) -> Optional[int]:
        """用 DurationHead 从提示词 connector 输出预测自然时长（秒）→ 帧数（8k+1）。

        DurationHead 消费 EmbeddingsProcessor 的 video/audio connector 输出
        （即 generate() 阶段 0 产出的 video_context / audio_context），
        回归目标为 log-seconds，forward 内已 exp 回秒。

        文件缺失 / 加载失败 / 预测异常时返回 None（上层降级为手动时长，
        不阻断生成——可选组件，遵循"文件缺失指引下载"但开关默认关）。

        Args:
            video_context: (B, T, 4096) video connector 输出
            audio_context: (B, T, 2048) audio connector 输出
            fps: 帧率（秒 → 帧数换算）
            max_seconds: 预测时长上限（两阶段管线约束 10s）

        Returns:
            对齐 8k+1 的帧数；失败返回 None
        """
        dh_dir = self.model_dir / self._DURATION_HEAD_DIR_RELPATH
        config_path = dh_dir / "config.json"
        weights_path = dh_dir / "diffusion_pytorch_model.safetensors"
        if not (config_path.exists() and weights_path.exists()):
            self._logger.warning(
                "duration_head 文件缺失（%s），自动时长降级为手动时长。"
                "如需启用请到模型管理界面下载共享组件", dh_dir,
            )
            return None

        try:
            import safetensors.torch as st

            from ltx_core.duration_head import DurationHead

            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            head = DurationHead(
                video_cross_attention_dim=cfg.get("video_cross_attention_dim", 4096),
                audio_cross_attention_dim=cfg.get("audio_cross_attention_dim", 2048),
                pooler_hidden_dim=cfg.get("pooler_hidden_dim", 256),
                num_queries=cfg.get("num_queries", 1),
                num_pooler_heads=cfg.get("num_pooler_heads", 4),
                mlp_hidden=cfg.get("mlp_hidden_dim", 256),
            )
            sd = st.load_file(str(weights_path))
            # diffusers 导出的分离投影（to_q/to_k/to_v/to_out）→ MHA 合并布局
            sd = _remap_diffusers_attention_sd(sd)
            head.load_state_dict(
                {k: v.to(torch.float32) for k, v in sd.items()}, strict=True,
            )
            head.eval().to(self.device)

            with torch.no_grad():
                seconds = float(
                    head(
                        video_tokens=video_context.to(self.device, torch.float32),
                        audio_tokens=audio_context.to(self.device, torch.float32),
                    )[0].item(),
                )
            del head
            self._cleanup()

            raw_seconds = seconds
            seconds = max(0.5, min(seconds, max_seconds))
            target = int(round(seconds * fps))
            k = max(0, round((target - 1) / _TEMPORAL_ALIGNMENT))
            num_frames = k * _TEMPORAL_ALIGNMENT + 1
            self._logger.info(
                "duration_head 预测：%.2fs（截断后 %.2fs）→ %d 帧 @%dfps",
                raw_seconds, seconds, num_frames, fps,
            )
            return num_frames
        except Exception as e:
            self._logger.warning("duration_head 预测失败，降级为手动时长: %s", e)
            return None

    # ===== T2V 推理（原型 run_t2v 两阶段蒸馏流程逐行移植）=====

    @staticmethod
    def _cast_conditionings_dtype(conditionings: list, dtype: torch.dtype) -> list:
        """条件 latent 从 FP32（VAE 编码精度）cast 到模型 dtype。

        多图条件用 FP32 编码（避免 fp16 VAE encoder 溢出 NaN），但条件张量
        若保持 FP32，``VideoConditionByKeyframeIndex.apply_to`` 中的
        ``torch.cat([latent, zeros_like(tokens)])`` 会把整个
        ``latent_state.latent`` 类型提升为 FP32，与 FP16 transformer 权重
        冲突（RuntimeError: mat1 and mat2 must have the same dtype, but
        got Float and Half）。latent 数值范围小，编码后 cast 回模型 dtype
        无精度风险。

        覆盖两种条件类型：
        - VideoConditionByLatentIndex（frame_idx=0 首帧）: .latent
        - VideoConditionByKeyframeIndex（frame_idx>0 尾帧/关键帧）: .keyframes
        """
        for cond in conditionings:
            latent = getattr(cond, "latent", None)
            if torch.is_tensor(latent):
                cond.latent = latent.to(dtype)
            keyframes = getattr(cond, "keyframes", None)
            if torch.is_tensor(keyframes):
                cond.keyframes = keyframes.to(dtype)
        return conditionings

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int = 42,
        width: int = 512,
        height: int = 256,
        num_frames: int = 33,
        fps: float = 24.0,
        output_path: Optional[str] = None,
        image_path: Optional[str] = None,
        image_paths: Optional[list[dict]] = None,
        auto_duration: bool = False,
        temporal_upscalings: int = 0,
        spatial_upscalings: int = 1,
        workflow: Optional[str] = None,
    ) -> dict:
        """执行原型 run_t2v 完整两阶段流程（T2V / I2V / FL2V / 多关键帧）。

        整个方法用 @torch.inference_mode() 装饰——这是关键（与原型一致）：
        autograd 计算图会把全部 transformer 权重钉在显存，dispose() 把模块参数
        换成 meta 也无法释放（图仍持有原始张量引用），Stage 2 重建时必然 OOM。
        这与 Gemma 文本编码器 42GB 泄漏是同一机制。

        分辨率约束（两阶段蒸馏管线，方法入口对齐，对齐后记 log）：
        - width/height 需 64 整除（Stage 1 半分辨率后仍满足 32 整除）
        - num_frames 需 8k+1（VAE 时间压缩 8×）

        多图条件路径（官方 ImageConditioner + combined_image_conditionings 天然支持 list）：
        - image_paths 非空时走多图条件：每张图对应一个 ImageConditioningInput
          （path + frame_idx + strength），frame_idx 决定条件帧在时间轴位置
          （0=首帧，num_frames-1=尾帧，中间值=关键帧位置）
        - image_path 为向后兼容旧签名：若 image_paths 为空且 image_path 非空，
          自动包装为单图 image_paths=[{"path": image_path, "frame_idx": 0, "strength": 1.0}]
        - CRF 重压缩：resolve_crf 从 VAE checkpoint 元数据读取模型训练时的
          default_image_crf（H.264 内存重压缩，匹配训练分布），PyAV 实现

        Args:
            prompt: 正向提示词
            seed: 随机种子
            width: 宽度（入口对齐到 64 整除）
            height: 高度（入口对齐到 64 整除）
            num_frames: 帧数（入口对齐到 8k+1）
            fps: 帧率
            output_path: 可选 mp4 输出路径（None = 只返回帧序列，不写文件）
            image_path: 【向后兼容】可选首帧图路径（I2V 模式；None = T2V）
            image_paths: 多图条件列表（新接口，优先于 image_path）：
                [{"path": str, "frame_idx": int, "strength": float, "crf": int|None}, ...]
                frame_idx: 0=首帧, num_frames-1=尾帧, 中间值=关键帧
                strength: 0~1，1.0=完全替换，0.8=强约束
            auto_duration: 自动时长开关（T3.3）：DurationHead 按 video/audio
                connector 输出预测自然时长（log-seconds 回归 → exp → 8k+1 帧），
                覆盖 num_frames；文件缺失或预测失败时降级手动时长

        Returns:
            dict 含：
                - video_frames: (T, H, W, C) uint8 numpy 数组
                - audio_data: 音频数据 (channels, samples) float32 numpy 数组
                  （AudioDecoder 解码 vocoder latent 的 waveform；提取失败为 None，
                  降级为无音轨视频）
                - metadata: 元信息（对齐后的尺寸/帧数/种子/infer_time/mode/
                  num_conditioning_images/conditioning_frame_idxs/audio_sample_rate/
                  has_audio/auto_duration）
        """
        # === 向后兼容：image_path（单图旧签名）→ image_paths（多图新签名）===
        if image_paths is None and image_path is not None:
            image_paths = [{"path": str(image_path), "frame_idx": 0, "strength": 1.0}]
        # 去重 & 校验
        if image_paths:
            image_paths = [
                {
                    "path": str(ip["path"]),
                    "frame_idx": int(ip.get("frame_idx", 0)),
                    "strength": float(ip.get("strength", 1.0)),
                    "crf": ip.get("crf"),
                }
                for ip in image_paths
            ]

        # ============================================================
        # P1-A DFRPipeline 路由切换（进阶）
        # ============================================================
        # 触发条件：temporal_upscalings > 0（帧间补帧）或 workflow 声明多宫格
        # —— 这两种场景 P0 原生 blocks 路径做不到高质量。
        # 检测失败（缺原生 transformer / detailing_lora / 显存不足）→
        # 日志警告后自动降级 P0，不阻断生成。
        _want_dfr = (
            temporal_upscalings > 0
            or workflow in ("multi_reference_v", "temporal_upscale_v")
        )
        if _want_dfr:
            self._logger.info(
                "触发 DFRPipeline 路由：temporal_upscalings=%d, workflow=%s",
                temporal_upscalings, workflow,
            )
            if self._dfr_adapter is None:
                from dfr_adapter import DFRAdapter
                self._dfr_adapter = DFRAdapter(
                    self.model_dir, self.device, self.dtype, self._logger,
                )
            dfr_result = self._dfr_adapter.generate_dfr(
                prompt=prompt,
                image_paths=image_paths or [],
                seed=seed,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                temporal_upscalings=temporal_upscalings,
                spatial_upscalings=spatial_upscalings,
                auto_duration=auto_duration,
                output_path=output_path,
            )
            if dfr_result is not None:
                return dfr_result
            self._logger.warning(
                "DFR 路由降级回退 P0 原生 blocks（原因见上方 DFR 检测日志），"
                "忽略 temporal_upscalings=%d 参数。"
                "如需 DFR 能力请下载原生 transformer + detailing_lora + "
                "temporal_upsampler，并确保显存 ≥48GB。",
                temporal_upscalings,
            )
            # 显式清零，避免 P0 路径误用
            temporal_upscalings = 0

        from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.blocks import AudioDecoder, DiffusionStage, VideoDecoder, VideoUpsampler
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            ensure_tiling_config,
            tiling_scale_factors_for_vae,
        )
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
        from ltx_pipelines.utils.types import ModalitySpec

        infer_start = time.time()

        # ===== 分辨率/帧数对齐（两阶段蒸馏管线约束：64 整除 + 8k+1）=====
        orig = (width, height, num_frames)
        width = max(_SPATIAL_ALIGNMENT, (width // _SPATIAL_ALIGNMENT) * _SPATIAL_ALIGNMENT)
        height = max(_SPATIAL_ALIGNMENT, (height // _SPATIAL_ALIGNMENT) * _SPATIAL_ALIGNMENT)
        k = max(0, round((num_frames - 1) / _TEMPORAL_ALIGNMENT))
        num_frames = k * _TEMPORAL_ALIGNMENT + 1
        if (width, height, num_frames) != orig:
            self._logger.info(
                "分辨率/帧数对齐（两阶段管线约束 64 整除 + 8k+1）：%dx%d %d帧 → %dx%d %d帧",
                orig[0], orig[1], orig[2], width, height, num_frames,
            )

        # LTX-2.5（model_version 2.5.0 >= (2,5)）：Stage 1 使用 ancestral（SDE）采样器
        ANCESTRAL_ETA = 1.0
        ANCESTRAL_S_NOISE = 1.0
        ANCESTRAL_NOISE_SEED_OFFSET = 10000

        assert_resolution(height=height, width=width, is_two_stage=True)
        if torch.cuda.is_available():
            self._logger.info(
                "基线 VRAM：%.1f GB free / %.1f GB（含 Windows/其他进程占用）",
                *torch.cuda.mem_get_info(),
            )

        # --- 文本编码（Gemma + EmbeddingsProcessor，用完释放显存） ---
        self._logger.info("=== 阶段 0：文本编码 ===")
        video_context, audio_context = self._encode_prompt(prompt)
        self._logger.info(
            "video_context: %s, audio_context: %s",
            tuple(video_context.shape), tuple(audio_context.shape),
        )

        # --- 自动时长（T3.3，可选开关）：DurationHead 预测帧数覆盖手动值 ---
        predicted_duration = False
        if auto_duration:
            pred_frames = self._predict_num_frames(video_context, audio_context, fps)
            if pred_frames is not None and pred_frames > 0:
                num_frames = pred_frames
                predicted_duration = True

        # === 多图条件构建（ImageConditioningInput list，天然支持多图！）===
        # ImageConditioner 持有 video encoder 生命周期（构建 → 编码 → 释放），
        # 内部与 Stage 2 的 upsampler 复用同一 conv VAE checkpoint。
        # combined_image_conditionings(images=[...]) 接受多图列表，每张图对应一个
        # VideoConditionByLatentIndex / VideoConditionByReferenceLatent 条件。
        stage_1_conditionings: list = []
        stage_2_conditionings: list = []
        image_conditioner = None
        i2v_images = None
        if image_paths:
            from ltx_pipelines.utils.args import ImageConditioningInput
            from ltx_pipelines.utils.blocks import ImageConditioner
            from ltx_pipelines.utils.helpers import combined_image_conditionings

            cond_dtype = torch.float32  # FP32 编码，避免 fp16 encoder 溢出 NaN
            image_conditioner = ImageConditioner(str(self.video_vae_path), cond_dtype, self.device)

            # 多图列表 → ImageConditioningInput list
            i2v_images = image_conditioner.resolve_crf([
                ImageConditioningInput(
                    path=ip["path"],
                    frame_idx=ip["frame_idx"],
                    strength=ip["strength"],
                    crf=ip.get("crf"),
                )
                for ip in image_paths
            ])
            self._logger.info(
                "多图条件构建（FP32）：%d 张图，frame_idx=%s, strength=%s, crf=%s",
                len(i2v_images),
                [img.frame_idx for img in i2v_images],
                [img.strength for img in i2v_images],
                [img.crf for img in i2v_images],
            )

            # Stage 1 条件（半分辨率编码，多图 latent 在各自 frame_idx 位置替换）
            stage_1_conditionings = image_conditioner(
                lambda enc: self._cast_conditionings_dtype(
                    combined_image_conditionings(
                        images=i2v_images,
                        height=height // 2,
                        width=width // 2,
                        video_encoder=enc,
                        dtype=cond_dtype,
                        device=self.device,
                    ),
                    self.dtype,
                )
            )
            self._logger.info(
                "Stage 1 多图条件编码完成（半分辨率，已 cast 到 %s）：%d 个 conditioning",
                self.dtype, len(stage_1_conditionings),
            )
            self._cleanup()

        # --- 两阶段去噪（DiffusionStage 管理 transformer 生命周期） ---
        self._logger.info("=== 阶段 1+2：transformer 去噪 ===")
        stage = DiffusionStage(
            GGUFTransformerBuilder(
                self.gguf_path, self.dtype, self.gemma_path, self.device, self._logger,
            ),
            self.dtype,
            self.device,
        )
        upsampler = VideoUpsampler(str(self.video_vae_path), str(self.upsampler_path), self.dtype, self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)

        t0 = time.time()
        # Stage 1：半分辨率 + ancestral 采样器
        video_state, audio_state = stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=width // 2,
            height=height // 2,
            frames=num_frames,
            fps=fps,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
            stepper=EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE),
            loop=partial(
                euler_ancestral_denoising_loop,
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
                model_dtype=self.dtype,
            ),
        )
        self._logger.info("Stage 1 完成（%.0fs），latent: %s", time.time() - t0, tuple(video_state.latent.shape))
        self._cleanup()

        # Stage 2：上采样 + 精修（确定性采样器）
        upscaled_video_latent = upsampler(video_state.latent[:1])

        # 多图条件：Stage 2 全分辨率重新编码（FP32 编码 + cast 到模型 dtype，同 Stage 1）
        # 与 Stage 1 的半分辨率条件独立；encoder 构建后释放
        if image_paths and image_conditioner is not None:
            stage_2_conditionings = image_conditioner(
                lambda enc: self._cast_conditionings_dtype(
                    combined_image_conditionings(
                        images=i2v_images,
                        height=height,
                        width=width,
                        video_encoder=enc,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    self.dtype,
                )
            )
            self._logger.info(
                "Stage 2 多图条件编码完成（全分辨率，已 cast 到 %s）：%d 个 conditioning",
                self.dtype, len(stage_2_conditionings),
            )
            self._cleanup()

        video_state, audio_state = stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=fps,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )
        self._logger.info("Stage 2 完成（总耗时 %.0fs）", time.time() - t0)
        del stage, upsampler, upscaled_video_latent
        self._cleanup()

        # --- 解码（VAE blocks 分时构建） ---
        self._logger.info("=== 阶段 3：VAE 解码 ===")
        video_decoder = VideoDecoder(str(self.video_vae_path), self.dtype, self.device)
        tiling_config = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(video_decoder.checkpoint_path),
            video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=fps),
            vae_checkpoint_path=video_decoder.checkpoint_path,
            diffvae_optimization=video_decoder.diffvae_optimization,
            device=self.device,
        )
        # 原型把惰性 chunk 迭代器直接交给 encode_video；这里需同时供帧提取与
        # 可选 mp4 导出使用（迭代器只能消费一次），逐 chunk 移 CPU 后物化为
        # 完整 (T, H, W, C) float [0,1] 张量（encode_video 内部自带 .to("cpu")）
        decoded_video = torch.cat(
            [chunk.detach().cpu() for chunk in video_decoder(video_state.latent, tiling_config, generator)],
            dim=0,
        )

        audio_decoder = AudioDecoder(str(self.audio_vae_path), self.dtype, self.device)
        decoded_audio = audio_decoder(audio_state.latent)

        # --- 音频提取（T3.2）：Audio{waveform, sampling_rate} → CPU numpy ---
        # waveform 形状 (channels, samples) 或 (samples,)，float [-1, 1]；
        # 上层 video_utils.save_video 负责转置/归一化/WAV 写盘/ffmpeg 混流
        audio_data: Optional[np.ndarray] = None
        audio_sample_rate = 48000
        try:
            audio_data = decoded_audio.waveform.detach().cpu().float().numpy()
            audio_sample_rate = int(decoded_audio.sampling_rate)
            self._logger.info(
                "音频解码完成：shape=%s, sr=%dHz",
                audio_data.shape, audio_sample_rate,
            )
        except Exception as e:
            # 音频提取失败不阻断视频产出（降级为无音轨）
            self._logger.warning("音频提取失败，降级为无音轨视频: %s", e)
            audio_data = None

        # --- 帧序列提取：(T, H, W, C) float [0,1] → 连续/clamp/255 → uint8 ---
        arr = decoded_video.detach().float().contiguous().clamp(0.0, 1.0).numpy()
        video_frames = (arr * 255).astype(np.uint8)
        self._logger.info("解码完成：%s uint8 帧", video_frames.shape)

        # --- 可选 mp4 导出 ---
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encode_video(
                video=decoded_video,
                fps=fps,
                audio=decoded_audio,
                output_path=str(output_path),
                video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
            )
            self._logger.info("输出完成：%s", output_path)

        infer_time = time.time() - infer_start
        # === mode 精细分类 ===
        if not image_paths:
            mode_label = "t2v"
        else:
            n_imgs = len(image_paths)
            has_first = any(ip["frame_idx"] == 0 for ip in image_paths)
            has_last = any(ip["frame_idx"] >= num_frames - 2 for ip in image_paths)
            if n_imgs == 1 and has_first:
                mode_label = "i2v"
            elif n_imgs == 2 and has_first and has_last:
                mode_label = "first_last_frame_v"
            elif n_imgs >= 2:
                mode_label = "multi_keyframe_v"
            else:
                mode_label = "i2v"  # fallback
        result_images = image_paths or []
        return {
            "video_frames": video_frames,
            "audio_data": audio_data,  # T3.2：音视频同步生成（None = 提取失败降级）
            "metadata": {
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "infer_time": infer_time,
                "mode": mode_label,
                "num_conditioning_images": len(result_images),
                "conditioning_frame_idxs": [ip["frame_idx"] for ip in result_images],
                "audio_sample_rate": audio_sample_rate if audio_data is not None else 0,
                "has_audio": audio_data is not None,
                "auto_duration": predicted_duration,
            },
        }

    # ===== 卸载 =====

    def unload(self) -> None:
        """释放持有引用并清理显存。

        blocks 分时加载架构下模型不常驻（generate 内部各阶段已即时释放），
        这里做兜底清理：GC + CUDA 缓存清空。
        """
        self._cleanup()
        self._logger.info("LTXVNativeRunner 卸载完成（blocks 分时加载架构，无常驻权重）")
