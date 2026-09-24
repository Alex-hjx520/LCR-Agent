"""向量化后端。

默认使用 `sentence-transformers`（本地推理，合同数据不出内网）；
测试或离线环境下可切换为 `hashing` 后端，无需下载模型即可跑通全流程。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """向量化接口。"""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """返回形状为 (n, dim) 的 float32 矩阵，且已完成 L2 归一化。"""
        ...

    def encode_one(self, text: str) -> np.ndarray:
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的本地向量模型。"""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        logger.info("加载本地向量模型 %s (device=%s)", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return _l2_normalize(vectors) if self.normalize else vectors

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder:
    """无依赖的确定性哈希向量后端（仅用于测试 / 离线兜底）。

    通过字符 n-gram 哈希投影实现，质量远低于真实模型，**不要用于生产检索**。
    """

    def __init__(self, dim: int = 256, ngram: int = 2) -> None:
        self.dim = dim
        self.ngram = ngram

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split() or [text.lower()]
        for token in tokens:
            grams = [token[i : i + self.ngram] for i in range(max(1, len(token) - self.ngram + 1))]
            for gram in grams:
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._vector(t) for t in texts]))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(backend: str | None = None, model_name: str | None = None) -> Embedder:
    """按配置返回（并缓存）向量化后端。"""
    settings = get_settings()
    backend = backend or settings.embedding_backend

    if backend == "hashing":
        key = f"hashing:{settings.embedding_dim}"
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = HashingEmbedder(dim=settings.embedding_dim)
        return _EMBEDDER_CACHE[key]

    model_name = model_name or settings.embedding_model
    key = f"st:{model_name}:{settings.embedding_device}"
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = SentenceTransformerEmbedder(
            model_name,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    return _EMBEDDER_CACHE[key]


def clear_embedder_cache() -> None:
    _EMBEDDER_CACHE.clear()


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "clear_embedder_cache",
    "get_embedder",
]
