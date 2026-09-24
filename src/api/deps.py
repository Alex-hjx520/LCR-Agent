"""FastAPI 依赖注入。

把「图」「检索器」「报告仓库」这类重量级、可复用的对象收敛到这里，
路由层只声明依赖，不关心构造过程。所有 getter 都带缓存，等价于单例。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends

from api.store import InMemoryReportStore
from config import Settings, get_settings
from graph.builder import get_graph
from knowledge import get_retriever


@lru_cache(maxsize=1)
def get_store() -> InMemoryReportStore:
    """报告仓库单例（进程内；生产环境可替换为 Redis / 数据库实现）。"""
    return InMemoryReportStore()


def settings_dep() -> Settings:
    """当前运行时配置。"""
    return get_settings()


def graph_dep() -> Any:
    """已编译的 LangGraph 图（单例）。"""
    return get_graph()


def retriever_dep() -> Any:
    """混合检索器（单例）。"""
    return get_retriever()


SettingsDep = Annotated[Settings, Depends(settings_dep)]
GraphDep = Annotated[Any, Depends(graph_dep)]
RetrieverDep = Annotated[Any, Depends(retriever_dep)]
StoreDep = Annotated[InMemoryReportStore, Depends(get_store)]


def reset_dependency_caches() -> None:
    """清空依赖缓存（测试用）。"""
    get_store.cache_clear()


__all__ = [
    "GraphDep",
    "RetrieverDep",
    "SettingsDep",
    "StoreDep",
    "get_store",
    "graph_dep",
    "reset_dependency_caches",
    "retriever_dep",
    "settings_dep",
]
