"""FastAPI 应用入口。

本地启动::

    make run
    # 等价于
    poetry run uvicorn api.main:app --app-dir src --reload

文档地址：http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.llm import describe_llm
from api.deps import get_store
from api.routes import health_router, review_router
from config import get_settings

logger = logging.getLogger(__name__)

VERSION = "0.1.0"
API_PREFIX = "/api/v1"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热重依赖（图 + 检索器），避免首个请求超时。"""
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    logger.info("LCR-Agent 启动中 (env=%s)…", settings.app_env)
    logger.info("LLM: %s", describe_llm())

    try:
        from graph.builder import get_graph

        get_graph()
        logger.info("审查图预热完成")
    except Exception:  # noqa: BLE001 - 预热失败不应阻止服务启动
        logger.exception("审查图预热失败，首个请求将承担初始化开销")

    logger.info("报告仓库就绪，容量 %d", get_store()._max_items)  # noqa: SLF001
    yield
    logger.info("LCR-Agent 已停止")


def create_app() -> FastAPI:
    """应用工厂：便于测试中以自定义配置创建独立实例。"""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="LCR-Agent · 法律合同审查 API",
        description=(
            "基于 LangGraph 的合同审查智能体：文档解析 → 条款抽取 → 风险审查 → "
            "合规校验 → 摘要报告。"
        ),
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=API_PREFIX)
    app.include_router(review_router, prefix=API_PREFIX)

    @app.get("/", tags=["system"], summary="服务元信息")
    def root() -> dict:
        return {
            "name": "LCR-Agent",
            "version": VERSION,
            "docs": "/docs",
            "api_prefix": API_PREFIX,
            "endpoints": [
                f"{API_PREFIX}/health",
                f"{API_PREFIX}/reviews",
                f"{API_PREFIX}/reviews/upload",
            ],
        }

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        logger.warning("请求参数错误: %s", exc)
        return JSONResponse(status_code=400, content={"detail": str(exc), "code": "bad_request"})

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=_settings.api_host,
        port=_settings.api_port,
        reload=_settings.is_dev,
    )
