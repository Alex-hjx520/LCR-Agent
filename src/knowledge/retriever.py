"""混合检索器：BM25（稀疏）+ Chroma（稠密），用 RRF 融合排序。

RRF（Reciprocal Rank Fusion）不依赖两路分数的量纲，工程上比线性加权更稳健：

    score(d) = Σ_r  w_r / (k + rank_r(d))

其中 `k` 默认 60（Cormack et al., 2009）。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from config import get_settings
from knowledge.base import PassageRecord
from knowledge.bm25 import BM25Index
from schemas.retrieval import Corpus, RetrievedPassage

logger = logging.getLogger(__name__)

#: 索引文件名约定
BM25_FILENAME = "bm25_index.pkl"


class HybridRetriever:
    """同时管理稀疏与稠密索引的混合检索器。"""

    def __init__(
        self,
        *,
        bm25: BM25Index | None = None,
        vector_store=None,
        rrf_k: int | None = None,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        settings = get_settings()
        self.bm25 = bm25 if bm25 is not None else BM25Index()
        self.vector_store = vector_store
        self.rrf_k = rrf_k or settings.rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    # --------------------------------------------------------------- 写入
    def add(self, records: list[PassageRecord]) -> int:
        """写入两路索引。"""
        count = self.bm25.add(records)
        if self.vector_store is not None:
            self.vector_store.add(records)
        return count

    # --------------------------------------------------------------- 查询
    def search_bm25(
        self, query: str, *, top_k: int | None = None, corpora: list[Corpus] | None = None
    ) -> list[RetrievedPassage]:
        return self.bm25.search(query, top_k=top_k, corpora=corpora)

    def search_vector(
        self, query: str, *, top_k: int | None = None, corpora: list[Corpus] | None = None
    ) -> list[RetrievedPassage]:
        if self.vector_store is None:
            return []
        return self.vector_store.search(query, top_k=top_k, corpora=corpora)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        corpora: list[Corpus] | None = None,
        use_vector: bool = True,
    ) -> list[RetrievedPassage]:
        """混合检索，返回 RRF 融合后的结果。"""
        settings = get_settings()
        top_k = top_k or settings.top_k

        sparse = self.search_bm25(query, top_k=settings.top_k_bm25, corpora=corpora)
        dense = (
            self.search_vector(query, top_k=settings.top_k_vector, corpora=corpora)
            if use_vector
            else []
        )

        if not dense:
            return sparse[:top_k]
        if not sparse:
            return dense[:top_k]

        fused = self._rrf_fuse(sparse, dense)
        return [p for p in fused if p.score >= settings.min_score][:top_k] or fused[:top_k]

    def _rrf_fuse(
        self, sparse: list[RetrievedPassage], dense: list[RetrievedPassage]
    ) -> list[RetrievedPassage]:
        scores: dict[str, float] = defaultdict(float)
        pool: dict[str, RetrievedPassage] = {}

        for weight, items, attr in (
            (self.bm25_weight, sparse, "bm25_rank"),
            (self.vector_weight, dense, "vector_rank"),
        ):
            for rank, passage in enumerate(items):
                scores[passage.passage_id] += weight / (self.rrf_k + rank + 1)
                if passage.passage_id not in pool:
                    pool[passage.passage_id] = passage
                else:
                    # 合并两路的 rank 信息与元数据
                    existing = pool[passage.passage_id]
                    if getattr(existing, attr) is None:
                        setattr(existing, attr, getattr(passage, attr))
                    existing.metadata.setdefault("label", passage.metadata.get("label"))

        fused: list[RetrievedPassage] = []
        for pid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            passage = pool[pid]
            passage.score = round(score, 6)
            if passage.vector_rank is None:
                passage.vector_rank = _find_rank(dense, pid)
            if passage.bm25_rank is None:
                passage.bm25_rank = _find_rank(sparse, pid)
            fused.append(passage)
        return fused

    # --------------------------------------------------------------- 持久化
    def save(self, index_dir: str | Path | None = None) -> Path:
        settings = get_settings()
        index_dir = Path(index_dir or settings.chroma_dir).parent
        return self.bm25.save(index_dir / BM25_FILENAME)

    @property
    def size(self) -> int:
        return self.bm25.size

    @property
    def ready(self) -> bool:
        return self.bm25.size > 0


def _find_rank(items: list[RetrievedPassage], passage_id: str) -> int | None:
    for rank, item in enumerate(items):
        if item.passage_id == passage_id:
            return rank
    return None


def build_retriever(*, with_vector: bool = True, persist_dir: str | Path | None = None) -> HybridRetriever:
    """构造检索器；`with_vector=False` 时只启用 BM25（无需下载向量模型）。"""
    if not with_vector:
        return HybridRetriever(vector_store=None)

    from knowledge.vectorstore import ChromaVectorStore

    try:
        store = ChromaVectorStore(persist_dir=persist_dir)
    except Exception:  # noqa: BLE001 - chromadb 缺失或目录不可写时降级
        logger.exception("向量库初始化失败，降级为纯 BM25 检索")
        return HybridRetriever(vector_store=None)
    return HybridRetriever(vector_store=store)


__all__ = ["BM25_FILENAME", "HybridRetriever", "build_retriever"]
