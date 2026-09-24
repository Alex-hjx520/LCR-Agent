"""FastAPI 接口层。

>>> from api.main import app          # ASGI 应用（uvicorn api.main:app）
>>> from api.main import create_app   # 应用工厂（测试用）
"""

from __future__ import annotations

from api.deps import get_store, graph_dep, retriever_dep, settings_dep
from api.main import API_PREFIX, VERSION, app, create_app

__all__ = [
    "API_PREFIX",
    "VERSION",
    "app",
    "create_app",
    "get_store",
    "graph_dep",
    "retriever_dep",
    "settings_dep",
]
