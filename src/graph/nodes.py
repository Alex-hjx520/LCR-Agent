"""图节点实现。

每个节点都是一个纯函数 `(state) -> partial_state`，便于单独测试；
需要注入依赖（Agent / 检索器 / 配置）的节点通过工厂函数构造，
这样既满足 LangGraph 的节点签名要求，又避免了全局可变状态。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from agents.base import BaseAgent
from config import Settings, get_settings
from graph.state import ReviewState
from parsing import chunk_document, parse_document, parse_text
from schemas.clause import ClauseCoverage
from schemas.review import ComplianceResult, ReviewReport, RiskIssue, Severity

logger = logging.getLogger(__name__)

Node = Callable[[ReviewState], dict[str, Any]]


def _timed(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """执行节点体，并把本节点耗时写入 trace。"""
    started = time.perf_counter()
    update = fn()
    elapsed = int((time.perf_counter() - started) * 1000)
    trace = list(update.get("trace") or [])
    if trace and trace[-1].get("node") == name:
        trace[-1]["latency_ms"] = elapsed
    else:
        trace.append({"node": name, "latency_ms": elapsed})
    update["trace"] = trace
    return update


# --------------------------------------------------------------------- ingest
def make_ingest_node(settings: Settings | None = None) -> Node:
    """把原始输入（文件路径或纯文本）规范化为 `ParsedDocument` + `raw_text`。"""

    def ingest_node(state: ReviewState) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            settings_ = settings or get_settings()
            source_path = state.get("source_path") or ""
            raw_text = state.get("raw_text") or ""
            filename = state.get("filename") or ""

            if source_path:
                document = parse_document(source_path, doc_id=state.get("doc_id") or None)
            elif raw_text:
                document = parse_text(
                    raw_text,
                    filename=filename or "inline.txt",
                    doc_id=state.get("doc_id") or None,
                )
            else:
                msg = "输入为空：必须提供 raw_text 或 source_path"
                logger.error(msg)
                return {"errors": [msg], "raw_text": "", "issues": []}

            return {
                "document": document,
                "doc_id": document.doc_id,
                "filename": document.filename,
                "raw_text": document.raw_text,
                "trace": [
                    {
                        "node": "ingest",
                        "blocks": document.num_blocks,
                        "chars": document.char_count,
                        "source_type": document.source_type.value,
                        "chunk_size": settings_.chunk_size,
                    }
                ],
            }

        return _timed("ingest", _run)

    return ingest_node


# ---------------------------------------------------------------------- chunk
def make_chunk_node(settings: Settings | None = None) -> Node:
    """对解析结果做语义切分，供检索与长合同分块审查使用。"""

    def chunk_node(state: ReviewState) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            document = state.get("document")
            if document is None:
                return {"chunks": [], "errors": ["chunk 节点缺少 document"]}
            chunks = chunk_document(document, chunk_size=(settings or get_settings()).chunk_size)
            return {"chunks": chunks, "trace": [{"node": "chunk", "chunks": len(chunks)}]}

        return _timed("chunk", _run)

    return chunk_node


# -------------------------------------------------------------------- agents
def make_agent_node(agent: BaseAgent, *, node_name: str | None = None) -> Node:
    """把任意 Agent 包装为图节点，并统一处理异常。"""
    name = node_name or agent.name

    def agent_node(state: ReviewState) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            try:
                return agent.run(dict(state))
            except Exception as exc:  # noqa: BLE001 - 单节点失败不应中断整图
                logger.exception("[%s] 节点执行失败", name)
                return {"errors": [f"{name}: {exc}"]}

        return _timed(name, _run)

    return agent_node


# -------------------------------------------------------------------- finalize
def make_finalize_node() -> Node:
    """汇总状态生成最终 `ReviewReport`。"""

    def finalize_node(state: ReviewState) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            clauses = list(state.get("clauses") or [])
            issues: list[RiskIssue] = list(state.get("issues") or [])
            compliance = state.get("compliance") or ComplianceResult()
            coverage = state.get("coverage") or ClauseCoverage()
            overall = state.get("overall_risk") or Severity.INFO

            report = ReviewReport(
                report_id=f"rpt_{state.get('doc_id', 'unknown')}",
                doc_id=state.get("doc_id", "unknown"),
                filename=state.get("filename", ""),
                contract_type=state.get("contract_type", "unknown"),
                summary=state.get("summary", ""),
                clauses=clauses,
                coverage=coverage,
                issues=sorted(issues, key=lambda i: (-i.severity.weight, i.clause_id or "")),
                compliance=compliance,
                overall_risk=overall,
                trace=list(state.get("trace") or []),
                stats=state.get("stats") or {},
            )
            return {"report": report, "trace": [{"node": "finalize"}]}

        return _timed("finalize", _run)

    return finalize_node


# --------------------------------------------------------------------- 路由
def route_after_ingest(state: ReviewState) -> str:
    """解析失败（无文本）时直接终止，避免后续节点空跑。"""
    return "fail" if state.get("errors") else "chunk"


def route_after_extract(state: ReviewState) -> str:
    """抽不到条款时跳过风险审查与合规检查，直接生成摘要。"""
    if not state.get("clauses"):
        return "summarize"
    return "review"


def make_fail_node() -> Node:
    """统一失败出口：写入错误并生成最小可用报告。"""

    def fail_node(state: ReviewState) -> dict[str, Any]:
        errors = list(state.get("errors") or ["未知错误"])
        logger.error("审查流程失败: %s", errors)
        return {
            "summary": "文档解析失败，无法完成审查。",
            "overall_risk": Severity.INFO,
            "stats": {"num_clauses": 0, "num_issues": 0, "errors": errors},
            "trace": [{"node": "fail", "errors": errors}],
        }

    return fail_node


__all__ = [
    "Node",
    "make_agent_node",
    "make_chunk_node",
    "make_fail_node",
    "make_finalize_node",
    "make_ingest_node",
    "route_after_extract",
    "route_after_ingest",
]
