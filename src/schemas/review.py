"""审查结果相关的数据模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from schemas.clause import Clause, ClauseCoverage, ClauseType
from schemas.retrieval import Corpus


class Severity(str, Enum):
    """风险等级。"""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        return {
            Severity.INFO: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4,
        }[self]

    @classmethod
    def worst(cls, items: list["Severity"]) -> "Severity":
        if not items:
            return cls.INFO
        return max(items, key=lambda s: s.weight)


class IssueCategory(str, Enum):
    """问题类别。"""

    MISSING_CLAUSE = "missing_clause"
    UNFAVORABLE_TERM = "unfavorable_term"
    AMBIGUITY = "ambiguity"
    LIABILITY_EXPOSURE = "liability_exposure"
    COMPLIANCE_RISK = "compliance_risk"
    INTERNAL_CONFLICT = "internal_conflict"
    UNUSUAL_TERM = "unusual_term"


class Citation(BaseModel):
    """问题引用的证据来源。"""

    corpus: Corpus = Corpus.CUAD
    ref: str = ""
    text: str = ""
    score: float = 0.0


class RiskIssue(BaseModel):
    """单条审查发现。"""

    issue_id: str
    doc_id: str
    clause_id: str | None = None
    clause_type: ClauseType = ClauseType.OTHER
    category: IssueCategory = IssueCategory.UNFAVORABLE_TERM
    severity: Severity = Severity.MEDIUM
    title: str
    description: str = ""
    suggestion: str = ""
    citations: list[Citation] = Field(default_factory=list)
    reviewer: str = "risk_reviewer"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def is_blocking(self) -> bool:
        return self.severity in {Severity.HIGH, Severity.CRITICAL}


class ComplianceFinding(BaseModel):
    """合规性检查结论。"""

    rule_id: str
    requirement: str
    passed: bool
    severity: Severity = Severity.MEDIUM
    detail: str = ""
    related_clause_type: ClauseType | None = None


class ComplianceResult(BaseModel):
    findings: list[ComplianceFinding] = Field(default_factory=list)

    @property
    def failed(self) -> list[ComplianceFinding]:
        return [f for f in self.findings if not f.passed]

    @property
    def pass_ratio(self) -> float:
        if not self.findings:
            return 1.0
        return (len(self.findings) - len(self.failed)) / len(self.findings)


class ReviewReport(BaseModel):
    """端到端审查报告，也是 API 的响应体。"""

    report_id: str
    doc_id: str
    filename: str = ""
    contract_type: str = "unknown"
    summary: str = ""
    clauses: list[Clause] = Field(default_factory=list)
    coverage: ClauseCoverage = Field(default_factory=ClauseCoverage)
    issues: list[RiskIssue] = Field(default_factory=list)
    compliance: ComplianceResult = Field(default_factory=ComplianceResult)
    overall_risk: Severity = Severity.INFO
    trace: list[dict] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def blocking_issues(self) -> list[RiskIssue]:
        return [i for i in self.issues if i.is_blocking]

    def severity_histogram(self) -> dict[str, int]:
        hist = {s.value: 0 for s in Severity}
        for issue in self.issues:
            hist[issue.severity.value] += 1
        return hist


__all__ = [
    "Citation",
    "ComplianceFinding",
    "ComplianceResult",
    "IssueCategory",
    "ReviewReport",
    "RiskIssue",
    "Severity",
]
