"""方案状态持久化存储。

把用户选择的默认方案持久化到 engine_state.json，下次启动自动使用。

存储路径：
    {data_root}/engine_state/ltx25_video_b9c4d2e1.json

存储格式：
    {
        "default_scheme": "I8",
        "history": [
            {
                "scheme": "I8",
                "variant": "Lightricks/LTX-2.5-Distilled",
                "timestamp": "2026-08-26T10:30:00",
                "success": true
            }
        ]
    }

设计原则（遵循 project-rules.md）：
    - 文件读写用 try/except 保护，避免文件损坏导致引擎无法启动
    - history 只保留最近 20 条，避免无限增长
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class SchemeStateStore:
    """方案状态持久化存储。

    用法：
        store = SchemeStateStore(data_root=Path("data"), engine_id="ltx25_video_b9c4d2e1")
        store.set_default_scheme("I8")
        default = store.get_default_scheme()  # "I8"
        store.record_history(scheme="I8", variant="Lightricks/LTX-2.5-Distilled", success=True)
    """

    def __init__(
        self,
        data_root: Path,
        engine_id: str = "ltx25_video_b9c4d2e1",
        logger: Optional[logging.Logger] = None,
    ):
        """初始化状态存储。

        Args:
            data_root: 数据根目录（如 data/）
            engine_id: 引擎 ID（用于构造状态文件名）
            logger: 可选的日志器
        """
        self._logger = logger or logging.getLogger(__name__)
        # 状态文件路径：{data_root}/engine_state/{engine_id}.json
        self._state_dir = data_root / "engine_state"
        self._state_file = self._state_dir / f"{engine_id}.json"
        self._state: dict = {"default_scheme": None, "history": []}

        # 启动时加载
        self._load()

    def _load(self) -> None:
        """从磁盘加载状态文件。"""
        try:
            if self._state_file.exists():
                with open(self._state_file, encoding="utf-8") as f:
                    self._state = json.load(f)
                self._logger.info(
                    "方案状态加载成功: file=%s, default_scheme=%s, history=%d",
                    self._state_file, self._state.get("default_scheme"),
                    len(self._state.get("history", [])),
                )
            else:
                self._logger.debug("方案状态文件不存在，使用空状态: %s", self._state_file)
        except Exception as e:
            self._logger.warning("方案状态加载失败（使用空状态）: %s, error=%s", self._state_file, e)
            self._state = {"default_scheme": None, "history": []}

    def _save(self) -> None:
        """保存状态到磁盘。"""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            self._logger.debug("方案状态保存成功: %s", self._state_file)
        except Exception as e:
            self._logger.warning("方案状态保存失败: %s, error=%s", self._state_file, e)

    def get_default_scheme(self) -> Optional[str]:
        """获取当前默认方案 ID。

        Returns:
            方案 ID（如 "I8"），未设置返回 None
        """
        return self._state.get("default_scheme")

    def set_default_scheme(self, scheme_id: str) -> None:
        """设置默认方案 ID。

        Args:
            scheme_id: 方案 ID（F16 / I8 / G8）
        """
        self._state["default_scheme"] = scheme_id
        self._save()
        self._logger.info("默认方案已设置: %s", scheme_id)

    def record_history(
        self,
        scheme: str,
        variant: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """记录方案使用历史。

        保留最近 20 条记录。

        Args:
            scheme: 方案 ID
            variant: variant 名称
            success: 是否成功
            error: 失败时的错误信息
        """
        history: list = self._state.setdefault("history", [])
        entry = {
            "scheme": scheme,
            "variant": variant,
            "timestamp": datetime.now().isoformat(),
            "success": success,
        }
        if error:
            entry["error"] = error

        history.insert(0, entry)
        # 只保留最近 20 条
        history[:] = history[:20]
        self._save()

    def get_history(self, limit: int = 20) -> list[dict]:
        """获取方案使用历史。

        Args:
            limit: 最多返回多少条

        Returns:
            历史记录列表（按时间倒序）
        """
        history: list = self._state.get("history", [])
        return history[:limit]
