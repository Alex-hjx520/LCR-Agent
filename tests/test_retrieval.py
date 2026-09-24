"""检索层测试：分词、BM25、RRF 融合、CUAD 类目映射。"""

from __future__ import annotations

import pytest

from knowledge.base import PassageRecord
from knowledge.bm25 import BM25Index, tokenize
from knowledge.cuad_loader import CUAD_CATEGORY_MAP, map_category
from knowledge.retriever import HybridRetriever
from schemas.clause import ClauseType
from schemas.retrieval import Corpus, RetrievedPassage


def _records() -> list[PassageRecord]:
    return [
        PassageRecord(
            passage_id="p1",
            text="任何一方违反保密义务的，应赔偿对方因此遭受的全部损失。",
            corpus=Corpus.CUAD,
            label="保密条款先例",
            clause_type=ClauseType.CONFIDENTIALITY.value,
        ),
        PassageRecord(
            passage_id="p2",
            text="乙方的赔偿责任总额不超过甲方已支付的合同价款。",
            corpus=Corpus.CUAD,
            label="责任限制先例",
            clause_type=ClauseType.LIABILITY.value,
        ),
        PassageRecord(
            passage_id="p3",
            text="本合同适用中华人民共和国法律并据此解释。",
            corpus=Corpus.PLAYBOOK,
            label="适用法律红线",
            clause_type=ClauseType.GOVERNING_LAW.value,
        ),
    ]


class TestTokenizer:
    def test_tokenize_handles_mixed_language(self):
        tokens = tokenize("Payment 期限为 30 日")
        assert tokens
        assert any("payment" in t for t in tokens)
        assert any("期限" in t or "期" in t for t in tokens)

    def test_tokenize_empty(self):
        assert tokenize("") == [] or all(t.strip() == "" for t in tokenize(""))


class TestBM25Index:
    def test_search_returns_relevant_first(self):
        index = BM25Index()
        index.add(_records())
        results = index.search("赔偿责任上限", top_k=3)

        assert results
        assert results[0].passage_id == "p2"
        assert results[0].bm25_rank == 0
        assert 0.0 <= results[0].score <= 1.0

    def test_search_filters_by_corpus(self):
        index = BM25Index()
        index.add(_records())
        results = index.search("法律适用", top_k=5, corpora=[Corpus.PLAYBOOK])
        assert [r.passage_id for r in results] == ["p3"]

    def test_search_on_empty_index(self):
        assert BM25Index().search("任意查询") == []

    def test_add_empty_is_noop(self):
        index = BM25Index()
        assert index.add([]) == 0
        assert index.size == 0

    def test_roundtrip_persistence(self, tmp_path):
        index = BM25Index()
        index.add(_records())
        path = index.save(tmp_path / "bm25.pkl")

        loaded = BM25Index.load(path)
        assert loaded.size == 3
        assert loaded.search("保密义务", top_k=1)[0].passage_id == "p1"


class FakeVectorStore:
    """记录调用的向量库桩，用于验证 RRF 融合逻辑。"""

    def __init__(self, results: list[RetrievedPassage]) -> None:
        self.results = results
        self.added: list[PassageRecord] = []

    def add(self, records: list[PassageRecord]) -> int:
        self.added.extend(records)
        return len(records)

    def search(self, query: str, *, top_k: int | None = None, corpora=None):
        items = self.results
        if corpora:
            items = [r for r in items if r.corpus in corpora]
        return items[: (top_k or len(items))]

    @property
    def count(self) -> int:
        return len(self.results)


class TestHybridRetriever:
    def test_rrf_fuses_two_rankings(self):
        sparse = [
            RetrievedPassage(passage_id="a", text="A", corpus=Corpus.CUAD, score=1.0, bm25_rank=0),
            RetrievedPassage(passage_id="b", text="B", corpus=Corpus.CUAD, score=0.5, bm25_rank=1),
        ]
        dense = [
            RetrievedPassage(passage_id="b", text="B", corpus=Corpus.CUAD, score=0.9, vector_rank=0),
            RetrievedPassage(passage_id="c", text="C", corpus=Corpus.CUAD, score=0.8, vector_rank=1),
        ]
        retriever = HybridRetriever(vector_store=FakeVectorStore(dense))
        fused = retriever._rrf_fuse(sparse, dense)  # noqa: SLF001

        # b 同时出现在两路，得分应最高
        assert fused[0].passage_id == "b"
        assert {p.passage_id for p in fused} == {"a", "b", "c"}
        assert fused[0].bm25_rank == 1
        assert fused[0].vector_rank == 0

    def test_search_falls_back_to_bm25_without_vector_store(self):
        retriever = HybridRetriever(vector_store=None)
        retriever.add(_records())
        results = retriever.search("保密义务", top_k=2)
        assert results
        assert results[0].passage_id == "p1"

    def test_add_writes_both_indexes(self):
        store = FakeVectorStore([])
        retriever = HybridRetriever(vector_store=store)
        assert retriever.add(_records()) == 3
        assert retriever.size == 3
        assert len(store.added) == 3
        assert retriever.ready is True


class TestCuadMapping:
    def test_known_categories_map_to_clause_types(self):
        assert map_category("Governing Law") is ClauseType.GOVERNING_LAW
        assert map_category("Non-Compete") is ClauseType.NON_COMPETE
        assert map_category("Cap On Liability") is ClauseType.LIABILITY
        assert map_category("  exclusivity  ") is ClauseType.EXCLUSIVITY

    def test_unknown_category_falls_back_to_other(self):
        assert map_category("Totally Unknown Category") is ClauseType.OTHER

    def test_mapping_table_is_consistent(self):
        assert len(CUAD_CATEGORY_MAP) >= 40
        assert all(isinstance(v, ClauseType) for v in CUAD_CATEGORY_MAP.values())


@pytest.mark.integration
def test_chroma_vector_store_roundtrip(tmp_path):
    """需要 chromadb 的真实向量库往返测试（使用 hashing 后端，不下载模型）。"""
    from knowledge.embeddings import get_embedder
    from knowledge.vectorstore import ChromaVectorStore

    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma", collection="test_collection", embedder=get_embedder()
    )
    store.add(_records())
    assert store.count == 3

    results = store.search("赔偿责任", top_k=2)
    assert results
    assert all(r.vector_rank is not None for r in results)
