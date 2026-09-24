"""FastAPI 请求 / 响应模型。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from schemas.review import ReviewReport, Severity


class HealthResponse(BaseModel):
    status: str = "ok"
    app_env: str = "dev"
    version: str = "0.1.0"
    llm_configured: bool = False
    index_ready: bool = False
    details: dict = Field(default_factory=dict)


class ReviewTextRequest(BaseModel):
    """直接提交合同纯文本进行审查。"""

    text: str = Field(..., min_length=1, description="合同全文纯文本")
    filename: str = "inline.txt"
    contract_type: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    use_retrieval: bool = True
    max_clauses: int | None = Field(default=None, ge=1, le=200)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            msg = "text 不能为空白内容"
            raise ValueError(msg)
        return v


class ReviewResponse(BaseModel):
    report_id: str
    status: str = "succeeded"
    overall_risk: Severity = Severity.INFO
    report: ReviewReport | None = None
    error: str | None = None


class ReviewSummary(BaseModel):
    """列表接口用的轻量摘要。"""

    report_id: str
    doc_id: str
    filename: str
    overall_risk: Severity
    issue_count: int
    created_at: str


class ReportListResponse(BaseModel):
    total: int
    items: list[ReviewSummary] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    code: str = "internal_error"


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "ReportListResponse",
    "ReviewResponse",
    "ReviewSummary",
    "ReviewTextRequest",
    "TaskState",
]
