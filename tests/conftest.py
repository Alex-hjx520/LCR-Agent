"""pytest 全局夹具。

测试环境的关键约束：
1. **不联网、不调用真实 LLM**：通过 `LCR_EMBEDDING_BACKEND=hashing` 与脚本化的
   `ScriptedChatModel` 实现确定性测试；
2. **必须先设置环境变量再导入 `config`**：`get_settings()` 有 `lru_cache`，
   因此这里在导入任何项目模块之前就把 env 写好；
3. **不触发 FastAPI lifespan**：`TestClient` 不在 `with` 块中使用，从而跳过
   启动时的图/检索器预热。
"""

from __future__ import annotations

import os

# ---- 必须在导入项目模块之前完成环境准备 -------------------------------------
os.environ.setdefault("LCR_APP_ENV", "test")
os.environ.setdefault("LCR_EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("LCR_EMBEDDING_DIM", "128")
os.environ.setdefault("LCR_LOG_LEVEL", "WARNING")
os.environ["OPENAI_API_KEY"] = ""  # 强制走 stub，避免误连真实端点
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from config import reload_settings  # noqa: E402
from tests.stubs import SAMPLE_CONTRACT, FakeCompiledGraph, ScriptedChatModel  # noqa: E402


class _DummyRetriever:
    """轻量检索器替身，避免健康检查触发真实索引 / 向量模型加载。"""

    size = 0
    vector_store = None

    def search(self, query, *, top_k=None, corpora=None):
        return []

    def add(self, records):
        return len(records)


# 测试数据（SAMPLE_CONTRACT）与测试替身（ScriptedChatModel / FakeCompiledGraph）
# 统一定义在 tests/stubs.py，避免 conftest 被重复导入时产生类身份不一致。


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session", autouse=True)
def _reset_settings_cache() -> None:
    """重置配置缓存，确保测试进程使用上面设置的环境变量。"""
    reload_settings()


@pytest.fixture
def settings():
    return reload_settings()


@pytest.fixture
def sample_contract() -> str:
    return SAMPLE_CONTRACT


@pytest.fixture
def tmp_contract_txt(tmp_path, sample_contract):
    path = tmp_path / "contract.txt"
    path.write_text(sample_contract, encoding="utf-8")
    return path


@pytest.fixture
def stub_llm() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def fake_graph() -> FakeCompiledGraph:
    return FakeCompiledGraph()


@pytest.fixture
def api_client(fake_graph):
    """FastAPI 测试客户端；图与检索器均被替换为替身，且不进入 `with` 块以跳过 lifespan。"""
    from fastapi.testclient import TestClient

    from api.deps import get_store, graph_dep, retriever_dep
    from api.main import app

    app.dependency_overrides[graph_dep] = lambda: fake_graph
    app.dependency_overrides[retriever_dep] = lambda: _DummyRetriever()

    store = get_store()
    store.clear()

    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        store.clear()
