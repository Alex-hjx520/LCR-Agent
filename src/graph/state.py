"""LangGraph 图状态定义。

状态字段分三类：
- **输入**：调用方在图执行前传入；
- **中间态**：由节点逐步填充（clauses / chunks / compliance ...）；
- **输出**：`report` 为最终产物。

`issues` / `trace` / `errors` 使用 `operator.add` reducer，因此节点只需返回
增量列表，LangGraph 会自动合并（这也是后续支持并行审查节点的基础）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from schemas.clause import Clause, ClauseCoverage
from schemas.document import Chunk, ParsedDocument
from schemas.review import ComplianceResult, ReviewReport, RiskIssue, Severity


class ReviewState(TypedDict, total=False):
    """合同审查流水线的共享状态。"""

    # ------------------------------------------------------------- 输入
    doc_id: str
    filename: str
    source_path: str
    raw_text: str
    contract_type: str
    use_retrieval: bool
    top_k: int
    max_clauses: int
    config: dict[str, Any]

    # ----------------------------------------------------------- 中间态
    document: ParsedDocument
    chunks: list[Chunk]
    clauses: list[Clause]
    coverage: ClauseCoverage
    compliance: ComplianceResult
    issues: Annotated[list[RiskIssue], operator.add]

    # ------------------------------------------------------------- 输出
    summary: str
    overall_risk: Severity
    stats: dict[str, Any]
    report: ReviewReport

    # --------------------------------------------------------- 可观测性
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


def initial_state(
    *,
    raw_text: str = "",
    filename: str = "",
    source_path: str = "",
    doc_id: str = "",
    use_retrieval: bool = True,
    top_k: int | None = None,
    max_clauses: int | None = None,
    **extra: Any,
) -> ReviewState:
    """构造一份干净的初始状态。

    `issues` / `errors` / `trace` 必须显式初始化为空列表，否则
    `operator.add` reducer 在首次合并时会与上一次运行的结果串味
    （同一 checkpointer 复用 thread 时尤其明显）。
    """
    state: ReviewState = {
        "raw_text": raw_text,
        "filename": filename,
        "source_path": source_path,
        "use_retrieval": use_retrieval,
        "issues": [],
        "errors": [],
        "trace": [],
    }
    if doc_id:
        state["doc_id"] = doc_id
    if top_k is not None:
        state["top_k"] = top_k
    if max_clauses is not None:
        state["max_clauses"] = max_clauses
    if extra:
        state["config"] = extra
    return state


__all__ = ["ReviewState", "initial_state"]
