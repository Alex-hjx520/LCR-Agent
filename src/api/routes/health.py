"""健康检查与就绪探针。"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from agents.llm import describe_llm
from api.deps import RetrieverDep, SettingsDep, StoreDep
from schemas.api import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])

VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse, summary="存活探针")
def health(settings: SettingsDep, retriever: RetrieverDep, store: StoreDep) -> HealthResponse:
    """返回进程存活状态与关键依赖摘要（不访问外部服务）。"""
    index_size = getattr(retriever, "size", 0)
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        version=VERSION,
        llm_configured=settings.has_llm,
        index_ready=bool(index_size),
        details={
            "llm": describe_llm(),
            "retriever_size": index_size,
            "retrieval_mode": "hybrid" if getattr(retriever, "vector_store", None) else "bm25",
            "reports_cached": store.size,
            "embedding_model": settings.embedding_model,
        },
    )


@router.get("/ready", response_model=HealthResponse, summary="就绪探针")
def ready(settings: SettingsDep, retriever: RetrieverDep, store: StoreDep) -> HealthResponse:
    """就绪探针：索引未构建时返回 `degraded`，供 K8s readinessProbe 使用。"""
    index_size = getattr(retriever, "size", 0)
    return HealthResponse(
        status="ready" if index_size else "degraded",
        app_env=settings.app_env,
        version=VERSION,
        llm_configured=settings.has_llm,
        index_ready=bool(index_size),
        details={"hint": "执行 `make index` 可构建检索索引", "reports_cached": store.size},
    )


__all__ = ["router"]
