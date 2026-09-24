"""Pydantic 数据模型包：全项目共享的契约层。

该包只依赖 pydantic，不依赖任何业务实现，因此可被 parsing / knowledge /
agents / graph / api 各层自由导入，不会产生循环依赖。
"""

from schemas.api import (
    ErrorResponse,
    HealthResponse,
    ReportListResponse,
    ReviewResponse,
    ReviewSummary,
    ReviewTextRequest,
    TaskState,
)
from schemas.clause import CRITICAL_CLAUSE_TYPES, Clause, ClauseCoverage, ClauseType
from schemas.document import Chunk, ParsedDocument, SourceType, TextBlock
from schemas.retrieval import Corpus, RetrievalQuery, RetrievedPassage
from schemas.review import (
    Citation,
    ComplianceFinding,
    ComplianceResult,
    IssueCategory,
    ReviewReport,
    RiskIssue,
    Severity,
)

__all__ = [
    "CRITICAL_CLAUSE_TYPES",
    "Chunk",
    "Citation",
    "Clause",
    "ClauseCoverage",
    "ClauseType",
    "ComplianceFinding",
    "ComplianceResult",
    "Corpus",
    "ErrorResponse",
    "HealthResponse",
    "IssueCategory",
    "ParsedDocument",
    "ReportListResponse",
    "RetrievalQuery",
    "RetrievedPassage",
    "ReviewReport",
    "ReviewResponse",
    "ReviewSummary",
    "ReviewTextRequest",
    "RiskIssue",
    "Severity",
    "SourceType",
    "TaskState",
    "TextBlock",
]
