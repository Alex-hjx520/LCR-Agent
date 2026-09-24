"""LangGraph 编排层。

对外入口：

>>> from graph import run_review
>>> report = run_review(raw_text=contract_text, filename="NDA.pdf")
>>> report.overall_risk, len(report.issues)
"""

from __future__ import annotations

from graph.builder import build_review_graph, get_graph, reset_graph_cache, run_review
from graph.nodes import (
    make_agent_node,
    make_chunk_node,
    make_finalize_node,
    make_ingest_node,
    route_after_extract,
    route_after_ingest,
)
from graph.state import ReviewState, initial_state

__all__ = [
    "ReviewState",
    "build_review_graph",
    "get_graph",
    "initial_state",
    "make_agent_node",
    "make_chunk_node",
    "make_finalize_node",
    "make_ingest_node",
    "reset_graph_cache",
    "route_after_extract",
    "route_after_ingest",
    "run_review",
]
