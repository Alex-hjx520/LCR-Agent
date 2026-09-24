"""知识库与检索包。

对外主要入口：

>>> from knowledge import get_retriever
>>> retriever = get_retriever()
>>> passages = retriever.search("单方解除权的触发条件")
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from config import get_settings
from knowledge.base import PassageRecord, Retriever
from knowledge.bm25 import BM25Index, tokenize
from knowledge.cuad_loader import (
    CUAD_CATEGORY_MAP,
    contract_stats,
    load_cuad_records,
    map_category,
    resolve_cuad_path,
)
from knowledge.embeddings import Embedder, get_embedder
from knowledge.retriever import BM25_FILENAME, HybridRetriever, build_retriever
from schemas.retrieval import Corpus, RetrievedPassage

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_retriever(*, with_vector: bool = True, load_from_disk: bool = True) -> HybridRetriever:
    """返回（并缓存）全局检索器实例。

    首次调用时会尝试从 `LCR_CHROMA_DIR` 的上级目录加载已持久化的 BM25 索引，
    加载失败则返回空索引（应用仍可启动，只是检索结果为空）。

    Args:
        with_vector: 是否启用 Chroma 稠密检索。设为 False 可跳过模型加载，
            显著加快测试与冷启动速度。
        load_from_disk: 是否尝试加载已持久化的 BM25 索引。
    """
    settings = get_settings()
    retriever = build_retriever(with_vector=with_vector)
    if load_from_disk and retriever.bm25.size == 0:
        bm25_path = Path(settings.chroma_dir).parent / BM25_FILENAME
        if bm25_path.exists():
            try:
                retriever.bm25 = BM25Index.load(bm25_path)
            except Exception:  # noqa: BLE001
                logger.exception("BM25 索引加载失败，将使用空索引: %s", bm25_path)
        else:
            logger.warning("未找到 BM25 索引 %s，检索结果将为空（请先执行 make index）", bm25_path)
    return retriever


def reset_retriever_cache() -> None:
    """清空检索器缓存（测试 / 重建索引后调用）。"""
    get_retriever.cache_clear()


__all__ = [
    "BM25_FILENAME",
    "BM25Index",
    "CUAD_CATEGORY_MAP",
    "Corpus",
    "Embedder",
    "HybridRetriever",
    "PassageRecord",
    "RetrievedPassage",
    "Retriever",
    "build_retriever",
    "contract_stats",
    "get_embedder",
    "get_retriever",
    "load_cuad_records",
    "map_category",
    "reset_retriever_cache",
    "resolve_cuad_path",
    "tokenize",
]
