"""BM25 稀疏检索（rank_bm25）。

中文合同没有天然空格分词，这里用「jieba（若已安装）→ 否则 字 + 二元字符组」
的混合分词策略，保证在任何环境下都有可用的召回基线。
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import numpy as np

from knowledge.base import PassageRecord
from schemas.retrieval import Corpus, RetrievedPassage

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _jieba_available() -> bool:
    try:
        import jieba  # noqa: F401

        return True
    except ImportError:
        return False


def tokenize(text: str) -> list[str]:
    """中英文混合分词。

    - 英文/数字：按词切分并小写化；
    - 中文：优先 jieba 分词，否则退化为「单字 + 相邻二元组」。
    """
    text = text.lower()
    if _jieba_available() and _CJK_RE.search(text):
        import jieba

        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        return [t for t in tokens if t not in {"，", "。", "、", "；", "：", "的", "了"}]

    tokens = _TOKEN_RE.findall(text)
    bigrams = [f"{a}{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


class BM25Index:
    """rank_bm25 的轻量封装，支持持久化到磁盘。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.records: list[PassageRecord] = []
        self._tokens: list[list[str]] = []
        self._bm25 = None

    # --------------------------------------------------------------- 写入
    def add(self, records: list[PassageRecord]) -> int:
        if not records:
            return 0
        self.records.extend(records)
        self._tokens.extend(tokenize(r.text) for r in records)
        self._build()
        return len(records)

    def _build(self) -> None:
        from rank_bm25 import BM25Okapi

        corpus = self._tokens or [["_empty_"]]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    # --------------------------------------------------------------- 查询
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        corpora: list[Corpus] | None = None,
    ) -> list[RetrievedPassage]:
        """返回稀疏检索结果（含 bm25_rank）。"""
        from config import get_settings

        if self._bm25 is None or not self.records:
            return []
        top_k = top_k or get_settings().top_k_bm25

        scores: np.ndarray = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        max_score = float(scores.max()) if scores.size else 0.0
        norm = max_score if max_score > 0 else 1.0

        results: list[RetrievedPassage] = []
        for rank, idx in enumerate(order):
            record = self.records[int(idx)]
            if corpora and record.corpus not in corpora:
                continue
            # rank_bm25 在小语料上可能给出负分，统一裁剪到 [0, 1] 便于跨检索器融合
            normalized = max(0.0, float(scores[idx])) / norm
            passage = record.to_passage(score=round(min(1.0, normalized), 6))
            passage.bm25_rank = rank
            results.append(passage)
            if len(results) >= top_k:
                break
        return results

    # --------------------------------------------------------------- 持久化
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "records": [r.model_dump(mode="json") for r in self.records],
            "tokens": self._tokens,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        logger.info("BM25 索引已保存至 %s（%d 条）", path, len(self.records))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - 仅加载本工程自产索引
        index = cls(k1=payload["k1"], b=payload["b"])
        index.records = [PassageRecord.model_validate(r) for r in payload["records"]]
        index._tokens = payload["tokens"]
        index._build()
        logger.info("已从 %s 加载 BM25 索引（%d 条）", path, len(index.records))
        return index

    @property
    def size(self) -> int:
        return len(self.records)


__all__ = ["BM25Index", "tokenize"]
