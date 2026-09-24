"""API 路由集合。"""

from __future__ import annotations

from api.routes.health import router as health_router
from api.routes.review import router as review_router

__all__ = ["health_router", "review_router"]
