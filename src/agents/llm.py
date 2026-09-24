"""LLM 工厂。

三种模式：
1. `openai`：配置了 `OPENAI_API_KEY` 时使用 `ChatOpenAI`（兼容 DeepSeek / Qwen /
   vLLM 等任意 OpenAI-兼容端点，只需设置 `OPENAI_BASE_URL`）；
2. `stub`：未配置密钥时返回 `FakeListChatModel`，让整条链路（含单元测试）仍可跑通；
3. 注入：测试可直接传入自定义 chat model。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)

#: stub 模式下返回的占位响应（结构化输出解析失败时会被上层兜底处理）
_STUB_RESPONSES = [
    '{"clauses": [], "issues": [], "summary": "未配置 LLM，返回占位结果。"}',
]


def get_chat_model(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    force_stub: bool = False,
    **kwargs: Any,
):
    """返回一个 LangChain ChatModel 实例。

    Args:
        model: 覆盖默认模型名。
        temperature: 覆盖默认温度。
        max_tokens: 覆盖最大输出 token 数。
        force_stub: 强制使用离线 stub（测试用）。
        **kwargs: 透传给 `ChatOpenAI` 的额外参数。
    """
    settings = get_settings()

    if force_stub or not settings.has_llm:
        if not force_stub:
            logger.warning(
                "未检测到 OPENAI_API_KEY，使用离线 stub 模型（结果不可用于真实审查）"
            )
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=_STUB_RESPONSES)

    from langchain_openai import ChatOpenAI

    # 让 langchain 内部其他组件也能读到凭据
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")
    if settings.openai_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)

    return ChatOpenAI(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        **kwargs,
    )


def describe_llm() -> dict[str, Any]:
    """返回当前 LLM 配置摘要（用于 /health 与日志）。"""
    settings = get_settings()
    return {
        "mode": "openai" if settings.has_llm else "stub",
        "model": settings.llm_model,
        "base_url": settings.openai_base_url or "https://api.openai.com/v1",
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }


__all__ = ["describe_llm", "get_chat_model"]
