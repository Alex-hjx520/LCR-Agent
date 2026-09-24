"""Agent 层：每个 Agent 负责审查流水线中的一个明确职责。

设计约定：
- Agent 无状态，输入 / 输出都通过图状态（dict）传递，便于 LangGraph 并行与重放；
- 所有 Agent 继承 `BaseAgent`，统一获得「检索增强 + 结构化输出 + 容错降级」能力；
- `ClauseExtractorAgent` 与 `RiskReviewerAgent` 依赖 LLM；
  `ComplianceCheckerAgent` 是确定性规则引擎，不消耗 token。
"""

from __future__ import annotations

from typing import Type

from agents.base import AgentResult, BaseAgent
from agents.clause_extractor import ClauseExtraction, ClauseExtractorAgent, ExtractedClause
from agents.compliance_checker import (
    ComplianceCheckerAgent,
    Playbook,
    PlaybookRule,
    get_playbook,
    load_playbook,
)
from agents.llm import describe_llm, get_chat_model
from agents.risk_reviewer import ClauseRisk, ClauseRiskBatch, RiskReviewerAgent
from agents.summarizer import SummarizerAgent

#: Agent 注册表（名称 -> 类），供图编排与实验配置按名构建
AGENT_REGISTRY: dict[str, Type[BaseAgent]] = {
    ClauseExtractorAgent.name: ClauseExtractorAgent,
    RiskReviewerAgent.name: RiskReviewerAgent,
    ComplianceCheckerAgent.name: ComplianceCheckerAgent,
    SummarizerAgent.name: SummarizerAgent,
}


def get_agent_class(name: str) -> Type[BaseAgent]:
    """按名称取得 Agent 类。"""
    try:
        return AGENT_REGISTRY[name]
    except KeyError:
        msg = f"未知 Agent: {name!r}，可选: {', '.join(sorted(AGENT_REGISTRY))}"
        raise KeyError(msg) from None


__all__ = [
    "AGENT_REGISTRY",
    "AgentResult",
    "BaseAgent",
    "ClauseExtraction",
    "ClauseExtractorAgent",
    "ClauseRisk",
    "ClauseRiskBatch",
    "ComplianceCheckerAgent",
    "ExtractedClause",
    "Playbook",
    "PlaybookRule",
    "RiskReviewerAgent",
    "SummarizerAgent",
    "describe_llm",
    "get_agent_class",
    "get_chat_model",
    "get_playbook",
    "load_playbook",
]
