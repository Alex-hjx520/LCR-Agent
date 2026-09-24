"""Agent 基础设施：提示词、结构化输出、检索增强与容错调用。"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from agents.llm import get_chat_model
from config import Settings, get_settings
from knowledge.base import Retriever
from schemas.retrieval import Corpus, RetrievedPassage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class AgentResult(BaseModel):
    """Agent 统一执行结果包装（便于在 graph 中记录 trace）。"""

    agent: str
    ok: bool = True
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    extra: dict = {}


class BaseAgent(ABC, Generic[T]):
    """所有审查 Agent 的基类。

    子类需定义：
    - `name` / `description`：用于注册与日志；
    - `system_prompt`：角色设定；
    - `run(state)`：读取图状态、返回增量更新。
    """

    name: str = "base_agent"
    description: str = ""

    #: 检索时使用的语料范围（None 表示不过滤）
    retrieval_corpora: list[Corpus] | None = None

    def __init__(
        self,
        *,
        llm: Any | None = None,
        retriever: Retriever | None = None,
        settings: Settings | None = None,
        top_k: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else get_chat_model()
        self.retriever = retriever
        self.top_k = top_k or self.settings.top_k

    # ------------------------------------------------------------- 提示词
    @property
    def system_prompt(self) -> str:
        return f"你是 {self.name}，一名严谨的法律合同审查助手。"

    def build_user_prompt(self, **kwargs: Any) -> str:
        """子类覆盖：构造用户提示词。"""
        raise NotImplementedError

    # ---------------------------------------------------------- 检索增强
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        corpora: list[Corpus] | None = None,
    ) -> list[RetrievedPassage]:
        """检索相关先例 / 红线条款；检索器不可用时返回空列表。"""
        if self.retriever is None:
            return []
        try:
            return self.retriever.search(
                query,
                top_k=top_k or self.top_k,
                corpora=corpora if corpora is not None else self.retrieval_corpora,
            )
        except Exception:  # noqa: BLE001 - 检索失败不应阻断整条审查链路
            logger.exception("[%s] 检索失败，降级为无外部证据模式", self.name)
            return []

    @staticmethod
    def format_evidence(passages: list[RetrievedPassage], *, max_items: int = 5) -> str:
        """把检索结果格式化为提示词中的证据块。"""
        if not passages:
            return "（无可用先例）"
        lines = []
        for i, p in enumerate(passages[:max_items], start=1):
            text = p.text.strip().replace("\n", " ")
            lines.append(f"[{i}] ({p.corpus.value}, score={p.score:.3f}) {p.label}: {text}")
        return "\n".join(lines)

    # ---------------------------------------------------------- LLM 调用
    def invoke_structured(
        self,
        schema: type[T],
        *,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[T | None, AgentResult]:
        """调用 LLM 并解析为 Pydantic 对象。

        先尝试原生 structured output，失败则回退到「纯文本 + JSON 提取」，
        两次都失败时返回 `(None, result)` 而不抛异常，交由节点决定降级策略。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        started = time.perf_counter()
        messages = [
            SystemMessage(content=system_prompt or self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        parsed: T | None = None
        error: str | None = None

        try:
            structured_llm = self.llm.with_structured_output(schema)
            parsed = structured_llm.invoke(messages)
            if isinstance(parsed, dict):  # 部分模型返回 dict
                parsed = schema.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] structured output 失败，回退 JSON 解析: %s", self.name, exc)
            error = str(exc)
            try:
                raw = self.llm.invoke(messages)
                parsed = _parse_json_payload(getattr(raw, "content", str(raw)), schema)
                error = None if parsed is not None else error
            except Exception as exc2:  # noqa: BLE001
                logger.exception("[%s] LLM 调用彻底失败", self.name)
                error = str(exc2)

        latency = int((time.perf_counter() - started) * 1000)
        usage = _usage_from(self.llm)
        result = AgentResult(
            agent=self.name,
            ok=parsed is not None,
            latency_ms=latency,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            error=error,
        )
        return parsed, result

    def invoke_text(
        self, *, user_prompt: str, system_prompt: str | None = None
    ) -> tuple[str, AgentResult]:
        """调用 LLM 返回纯文本（用于摘要等自由文本任务）。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        started = time.perf_counter()
        try:
            raw = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt or self.system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = str(getattr(raw, "content", raw))
            usage = _usage_from(self.llm)
            return text, AgentResult(
                agent=self.name,
                latency_ms=int((time.perf_counter() - started) * 1000),
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] 文本生成失败", self.name)
            return "", AgentResult(
                agent=self.name,
                ok=False,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    # --------------------------------------------------------------- 执行
    @abstractmethod
    def run(self, state: dict) -> dict:
        """执行 Agent，返回需要写回图状态的增量字段。"""
        raise NotImplementedError


def _parse_json_payload(text: str, schema: type[T]) -> T | None:
    """从自由文本中提取第一个合法 JSON 并校验为 schema。"""
    candidates: list[str] = []
    fenced = _JSON_FENCE_RE.findall(text)
    candidates.extend(fenced)
    candidates.append(text.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            logger.debug("JSON 校验失败: %s", exc)
            continue
    return None


def _usage_from(llm: Any) -> tuple[int, int]:
    """尽力从 chat model 上读取最近一次调用的 token 用量。"""
    usage = getattr(llm, "last_usage", None)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return 0, 0


__all__ = ["AgentResult", "BaseAgent"]
