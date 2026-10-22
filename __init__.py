"""LTX-2.5 视频生成引擎"""
__version__ = "0.1.0"

# 强制离线模式：必须在引擎任何模块导入 diffusers / transformers / huggingface_hub 之前设置
# 权重已下载到本地，每次启动都连 huggingface.co 做 HEAD 检查会因网络问题超时
import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"

# huggingface_hub.constants 在首次导入时读取环境变量并缓存为模块属性
# 仅设 os.environ 不够（constants.HF_HUB_OFFLINE 可能已是 False），必须 patch 模块属性
try:
    import huggingface_hub.constants as _hf_constants
    _hf_constants.HF_HUB_OFFLINE = True
    _hf_constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
    # ENDPOINT 置空避免任何对 huggingface.co 的请求
    _hf_constants.ENDPOINT = ""
except ImportError:
    pass

# transformers 同理：transformers.utils.hub 模块也缓存了常量
try:
    import transformers.utils.hub as _tf_hub
    if hasattr(_tf_hub, "HF_HUB_OFFLINE"):
        _tf_hub.HF_HUB_OFFLINE = True
except ImportError:
    pass
