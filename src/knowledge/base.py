"""知识库层的公共抽象：索引记录与检索器协议。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from schemas.retrieval import Corpus, RetrievedPassage


class PassageRecord(BaseModel):
    """写入索引的最小记录单元。"""

    passage_id: str
    text: str
    corpus: Corpus = Corpus.CUAD
    label: str = ""
    clause_type: str | None = None
    source: str = ""
    metadata: dict = Field(default_factory=dict)

    def to_passage(self, *, score: float = 0.0) -> RetrievedPassage:
        return RetrievedPassage(
            passage_id=self.passage_id,
            text=self.text,
            corpus=self.corpus,
            score=score,
            metadata={"label": self.label or self.passage_id, **self.metadata},
        )


@runtime_checkable
class Retriever(Protocol):
    """检索器统一接口（BM25 / 向量 / 混合检索均实现该协议）。"""

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        corpora: list[Corpus] | None = None,
    ) -> list[RetrievedPassage]:
        """返回按相关性降序排列的结果。"""
        ...

    def add(self, records: list[PassageRecord]) -> int:
        """写入记录，返回写入条数。"""
        ...


__all__ = ["PassageRecord", "Retriever"]
