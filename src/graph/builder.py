"""LangGraph 编排：把各 Agent 组装成可执行、可观测的审查工作流。

拓扑::

    START → ingest → chunk → extract ─┬─→ review → compliance → summarize → finalize → END
                │                     └────────────────────────→ summarize
                └─(解析失败)→ fail ────────────────────────────→ finalize

通过 `build_review_graph()` 显式注入依赖（LLM / 检索器 / playbook / checkpointer），
方便在测试中替换为 stub，也方便在服务端以单例形式复用已编译的图。
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from agents.clause_extractor import ClauseExtractorAgent
from agents.compliance_checker import ComplianceCheckerAgent, Playbook
from agents.risk_reviewer import RiskReviewerAgent
from agents.summarizer import SummarizerAgent
from config import Settings, get_settings
from graph.nodes import (
    make_agent_node,
    make_chunk_node,
    make_fail_node,
    make_finalize_node,
    make_ingest_node,
    route_after_extract,
    route_after_ingest,
)
from graph.state import ReviewState, initial_state
from knowledge.base import Retriever
from schemas.review import ReviewReport

logger = logging.getLogger(__name__)


def build_review_graph(
    *,
    llm: Any | None = None,
    retriever: Retriever | None = None,
    settings: Settings | None = None,
    playbook: Playbook | None = None,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
):
    """构建并编译合同审查图。

    Args:
        llm: LangChain ChatModel；None 时按配置自动创建（无密钥则走 stub）。
        retriever: 混合检索器；None 表示不做外部证据检索。
        settings: 运行时配置。
        playbook: 合规红线，默认从 `configs/playbook.yaml` 加载。
        checkpointer: LangGraph 持久化器（如 `MemorySaver`），用于断点续跑 / 人工审核。
        interrupt_before: 在这些节点前中断（HITL：人工确认后再继续）。

    Returns:
        已编译的 LangGraph `CompiledStateGraph`。
    """
    from langgraph.graph import END, START, StateGraph

    settings = settings or get_settings()

    extractor = ClauseExtractorAgent(llm=llm, retriever=retriever, settings=settings)
    reviewer = RiskReviewerAgent(llm=llm, retriever=retriever, settings=settings)
    checker = ComplianceCheckerAgent(llm=llm, retriever=retriever, settings=settings, playbook=playbook)
    summarizer = SummarizerAgent(llm=llm, retriever=retriever, settings=settings)

    builder = StateGraph(ReviewState)
    builder.add_node("ingest", make_ingest_node(settings))
    builder.add_node("chunk", make_chunk_node(settings))
    builder.add_node("extract", make_agent_node(extractor))
    builder.add_node("review", make_agent_node(reviewer))
    builder.add_node("compliance", make_agent_node(checker))
    builder.add_node("summarize", make_agent_node(summarizer))
    builder.add_node("finalize", make_finalize_node())
    builder.add_node("fail", make_fail_node())

    builder.add_edge(START, "ingest")
    builder.add_conditional_edges(
        "ingest", route_after_ingest, {"chunk": "chunk", "fail": "fail"}
    )
    builder.add_edge("chunk", "extract")
    builder.add_conditional_edges(
        "extract", route_after_extract, {"review": "review", "summarize": "summarize"}
    )
    builder.add_edge("review", "compliance")
    builder.add_edge("compliance", "summarize")
    builder.add_edge("summarize", "finalize")
    builder.add_edge("fail", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or None,
    )
    logger.info("审查图编译完成，节点: %s", sorted(builder.nodes))
    return graph


@lru_cache(maxsize=1)
def get_graph():
    """返回服务端复用的图单例（延迟初始化检索器）。"""
    from knowledge import get_retriever

    retriever: Retriever | None
    try:
        retriever = get_retriever()
    except Exception:  # noqa: BLE001 - 索引不可用时仍允许无检索运行
        logger.exception("检索器初始化失败，将以无检索模式运行")
        retriever = None
    return build_review_graph(retriever=retriever)


def reset_graph_cache() -> None:
    """清空图单例（重建索引或测试时调用）。"""
    get_graph.cache_clear()


def run_review(
    *,
    raw_text: str = "",
    source_path: str = "",
    filename: str = "",
    doc_id: str | None = None,
    use_retrieval: bool = True,
    top_k: int | None = None,
    max_clauses: int | None = None,
    graph: Any | None = None,
    thread_id: str | None = None,
    **extra: Any,
) -> ReviewReport:
    """端到端执行一次审查，返回 `ReviewReport`。

    这是 graph 层对外的唯一入口，API 与脚本都调用它，保证行为一致。
    """
    compiled = graph or get_graph()
    state = initial_state(
        raw_text=raw_text,
        filename=filename,
        source_path=source_path,
        doc_id=doc_id or f"doc_{uuid.uuid4().hex[:8]}",
        use_retrieval=use_retrieval,
        top_k=top_k,
        max_clauses=max_clauses,
        **extra,
    )
    config = {"configurable": {"thread_id": thread_id or state["doc_id"]}}
    result = compiled.invoke(state, config=config)

    report = result.get("report")
    if report is None:
        # 兜底：图未能走到 finalize（例如异常中断）时补一个最小报告
        report = ReviewReport(
            report_id=f"rpt_{state['doc_id']}",
            doc_id=state["doc_id"],
            filename=filename,
            summary=result.get("summary", ""),
            clauses=result.get("clauses", []),
            issues=result.get("issues", []),
            trace=result.get("trace", []),
            stats={**(result.get("stats") or {}), "errors": result.get("errors", [])},
        )
    return report


__all__ = ["build_review_graph", "get_graph", "reset_graph_cache", "run_review"]
