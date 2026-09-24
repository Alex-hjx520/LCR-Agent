"""检索相关的数据模型。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Corpus(str, Enum):
    """检索语料的来源。"""

    CUAD = "cuad"          # CUAD 数据集中的先例条款
    PLAYBOOK = "playbook"  # 企业审查标准 / 红线清单
    STATUTE = "statute"    # 法律法规条文
    CONTRACT = "contract"  # 当前被审查的合同自身


class RetrievedPassage(BaseModel):
    """一条检索结果。"""

    passage_id: str
    text: str
    corpus: Corpus = Corpus.CUAD
    score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def label(self) -> str:
        """用于提示词中的简短引用标识。"""
        return str(self.metadata.get("label") or self.passage_id)


class RetrievalQuery(BaseModel):
    """一次检索请求。"""

    text: str
    top_k: int | None = None
    corpora: list[Corpus] | None = None
    filters: dict | None = None


__all__ = ["Corpus", "RetrievalQuery", "RetrievedPassage"]
