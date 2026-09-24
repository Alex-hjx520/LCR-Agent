"""Chroma 向量库封装。

设计要点：
- 只存「向量 + 文本 + 扁平化元数据」，Chroma 的 metadata 仅支持标量值，
  因此写入前统一做扁平化处理；
- 支持按 corpus 过滤，便于把 CUAD 先例、企业 playbook、法条混库存储；
- 索引持久化在 `LCR_CHROMA_DIR`，可直接随项目物化到服务器。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import get_settings
from knowledge.base import PassageRecord
from knowledge.embeddings import Embedder, get_embedder
from schemas.retrieval import Corpus, RetrievedPassage

logger = logging.getLogger(__name__)


def _flatten_metadata(record: PassageRecord) -> dict[str, Any]:
    """把元数据压成 Chroma 可接受的标量字典。"""
    meta: dict[str, Any] = {
        "corpus": record.corpus.value,
        "label": record.label or record.passage_id,
        "source": record.source,
    }
    if record.clause_type:
        meta["clause_type"] = record.clause_type
    for key, value in record.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        elif value is None:
            continue
        else:
            meta[key] = str(value)
    return meta


class ChromaVectorStore:
    """基于 Chroma 的稠密检索实现。"""

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        import chromadb

        settings = get_settings()
        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.collection_name = collection or settings.chroma_collection
        self.embedder = embedder or get_embedder()

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma 就绪：dir=%s collection=%s size=%d",
            self.persist_dir,
            self.collection_name,
            self.count,
        )

    # --------------------------------------------------------------- 写入
    def add(self, records: list[PassageRecord], *, batch_size: int = 64) -> int:
        """批量写入记录（自动分块，避免单次请求过大）。"""
        if not records:
            return 0
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            vectors = self.embedder.encode([r.text for r in batch])
            self._collection.upsert(
                ids=[r.passage_id for r in batch],
                documents=[r.text for r in batch],
                metadatas=[_flatten_metadata(r) for r in batch],
                embeddings=vectors.tolist(),
            )
        logger.info("已写入 %d 条记录到 %s", len(records), self.collection_name)
        return len(records)

    # --------------------------------------------------------------- 查询
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        corpora: list[Corpus] | None = None,
    ) -> list[RetrievedPassage]:
        """向量相似度检索（含 vector_rank）。"""
        settings = get_settings()
        top_k = top_k or settings.top_k_vector
        if self.count == 0:
            return []

        where: dict[str, Any] | None = None
        if corpora:
            values = [c.value for c in corpora]
            where = {"corpus": values[0]} if len(values) == 1 else {"corpus": {"$in": values}}

        vector = self.embedder.encode_one(query)
        raw = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, max(self.count, 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        results: list[RetrievedPassage] = []
        for rank, (pid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists, strict=False)):
            meta = dict(meta or {})
            corpus = _safe_corpus(meta.pop("corpus", None))
            label = str(meta.pop("label", pid))
            # cosine 距离 -> 相似度
            score = round(max(0.0, 1.0 - float(dist)), 6)
            results.append(
                RetrievedPassage(
                    passage_id=pid,
                    text=text or "",
                    corpus=corpus,
                    score=score,
                    vector_rank=rank,
                    metadata={"label": label, **meta},
                )
            )
        return results

    # --------------------------------------------------------------- 维护
    @property
    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:  # noqa: BLE001
            return 0

    def reset(self, *, collection: str | None = None) -> None:
        """删除并重建 collection（重建索引用）。"""
        name = collection or self.collection_name
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001 - 不存在时忽略
            logger.debug("collection %s 不存在，跳过删除", name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("collection %s 已重置", name)

    def peek(self, limit: int = 5) -> list[dict[str, Any]]:
        data = self._collection.peek(limit)
        return list(zip(data.get("ids", []), data.get("documents", []), strict=False))


def _safe_corpus(value: Any) -> Corpus:
    try:
        return Corpus(value)
    except ValueError:
        return Corpus.CUAD


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


__all__ = ["ChromaVectorStore", "cosine_similarity"]
