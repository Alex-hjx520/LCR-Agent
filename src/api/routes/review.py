"""合同审查接口。

- `POST /reviews`          提交纯文本合同，同步返回完整报告
- `POST /reviews/upload`   上传 PDF / DOCX / TXT 文件
- `GET  /reviews`          分页列出历史报告摘要
- `GET  /reviews/{id}`     获取单份报告
- `DELETE /reviews/{id}`   删除报告
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from api.deps import GraphDep, StoreDep
from graph.builder import run_review
from schemas.api import (
    ErrorResponse,
    ReportListResponse,
    ReviewResponse,
    ReviewSummary,
    ReviewTextRequest,
)
from schemas.review import ReviewReport, Severity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reviews", tags=["review"])

#: 上传文件大小上限（10 MB）
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@router.post(
    "",
    response_model=ReviewResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="审查纯文本合同",
)
async def create_review(payload: ReviewTextRequest, graph: GraphDep, store: StoreDep) -> ReviewResponse:
    """对提交的合同文本执行完整审查流水线。"""
    report = await _execute(
        graph,
        raw_text=payload.text,
        filename=payload.filename,
        use_retrieval=payload.use_retrieval,
        top_k=payload.top_k,
        max_clauses=payload.max_clauses,
        contract_type=payload.contract_type,
    )
    store.save(report)
    return ReviewResponse(
        report_id=report.report_id, overall_risk=report.overall_risk, report=report
    )


@router.post(
    "/upload",
    response_model=ReviewResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="上传文件并审查（PDF / DOCX / TXT / MD）",
)
async def upload_review(
    graph: GraphDep,
    store: StoreDep,
    file: UploadFile = File(..., description="合同文件"),
    use_retrieval: bool = Form(default=True),
    top_k: int | None = Form(default=None),
) -> ReviewResponse:
    """接收上传文件，落临时文件后交给解析器处理。"""
    filename = file.filename or "upload.bin"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型 {suffix}，仅支持 {sorted(ALLOWED_SUFFIXES)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大（{len(content)} 字节），上限 {MAX_UPLOAD_BYTES} 字节",
        )

    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        report = await _execute(
            graph,
            source_path=str(tmp_path),
            filename=filename,
            use_retrieval=use_retrieval,
            top_k=top_k,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    store.save(report)
    return ReviewResponse(
        report_id=report.report_id, overall_risk=report.overall_risk, report=report
    )


@router.get("", response_model=ReportListResponse, summary="列出历史报告")
def list_reviews(
    store: StoreDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ReportListResponse:
    """按创建时间倒序返回报告摘要。"""
    reports = store.list(limit=limit, offset=offset)
    return ReportListResponse(
        total=store.size,
        items=[
            ReviewSummary(
                report_id=r.report_id,
                doc_id=r.doc_id,
                filename=r.filename,
                overall_risk=r.overall_risk,
                issue_count=len(r.issues),
                created_at=r.created_at.isoformat(),
            )
            for r in reports
        ],
    )


@router.get(
    "/{report_id}",
    response_model=ReviewReport,
    responses={404: {"model": ErrorResponse}},
    summary="获取单份报告",
)
def get_review(report_id: str, store: StoreDep) -> ReviewReport:
    report = store.get(report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"报告不存在: {report_id}"
        )
    return report


@router.delete(
    "/{report_id}",
    responses={404: {"model": ErrorResponse}},
    summary="删除报告",
)
def delete_review(report_id: str, store: StoreDep) -> dict:
    if not store.delete(report_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"报告不存在: {report_id}"
        )
    return {"deleted": report_id}


async def _execute(graph, **kwargs) -> ReviewReport:
    """在线程池中执行阻塞的 LangGraph 调用，避免阻塞事件循环。"""
    try:
        return await run_in_threadpool(lambda: run_review(graph=graph, **kwargs))
    except Exception as exc:  # noqa: BLE001
        logger.exception("审查执行失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"审查执行失败: {exc}",
        ) from exc


__all__ = ["MAX_UPLOAD_BYTES", "router", "Severity"]
