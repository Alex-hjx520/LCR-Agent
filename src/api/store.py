"""审查报告的存储层。

当前实现是进程内的内存仓库，用于脚手架阶段跑通链路；接口刻意设计成与
「持久化实现」一致（save / get / list / delete），后续替换为 SQLite、Postgres
或 Redis 时调用方无需改动。
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from schemas.review import ReviewReport

logger = logging.getLogger(__name__)


class InMemoryReportStore:
    """线程安全、有容量上限的 LRU 报告仓库。"""

    def __init__(self, max_items: int = 200) -> None:
        self._max_items = max_items
        self._lock = threading.RLock()
        self._reports: OrderedDict[str, ReviewReport] = OrderedDict()

    # --------------------------------------------------------------- 写入
    def save(self, report: ReviewReport) -> ReviewReport:
        with self._lock:
            self._reports[report.report_id] = report
            self._reports.move_to_end(report.report_id)
            while len(self._reports) > self._max_items:
                evicted, _ = self._reports.popitem(last=False)
                logger.debug("报告 %s 因超出容量被淘汰", evicted)
        return report

    # --------------------------------------------------------------- 读取
    def get(self, report_id: str) -> ReviewReport | None:
        with self._lock:
            report = self._reports.get(report_id)
            if report is not None:
                self._reports.move_to_end(report_id)
            return report

    def list(self, *, limit: int = 50, offset: int = 0) -> list[ReviewReport]:
        with self._lock:
            items = list(self._reports.values())
        items.sort(key=lambda r: r.created_at, reverse=True)
        return items[offset : offset + limit]

    # --------------------------------------------------------------- 删除
    def delete(self, report_id: str) -> bool:
        with self._lock:
            return self._reports.pop(report_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._reports)


__all__ = ["InMemoryReportStore"]
