"""合规检查 Agent。

与风险审查不同，合规检查是**确定性**的：依据 `configs/playbook.yaml` 中声明的
红线规则（必备条款 / 禁止表述 / 必备表述）对条款集合做规则匹配，结果可复现、
可审计，适合作为流水线中的"硬门槛"。

同时支持把「缺失的必备条款」转化为高优先级风险项，交给报告统一呈现。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from agents.base import BaseAgent
from config import get_settings
from schemas.clause import Clause, ClauseCoverage, ClauseType
from schemas.review import (
    ComplianceFinding,
    ComplianceResult,
    IssueCategory,
    RiskIssue,
    Severity,
)

logger = logging.getLogger(__name__)

DEFAULT_PLAYBOOK = "playbook.yaml"


# --------------------------------------------------------------------- 模型
class PlaybookRule(BaseModel):
    """一条合规红线规则。"""

    id: str
    requirement: str
    severity: Severity = Severity.MEDIUM
    required_clause: ClauseType | None = None
    forbidden_patterns: list[str] = Field(default_factory=list)
    required_patterns: list[str] = Field(default_factory=list)
    applies_to: list[ClauseType] | None = None
    message: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _upper_severity(cls, v):
        return v.lower() if isinstance(v, str) else v


class Playbook(BaseModel):
    """企业审查标准（playbook）。"""

    name: str = "default"
    version: str = "0.1.0"
    description: str = ""
    counterparty_bias: str = "buyer"
    required_clauses: list[ClauseType] = Field(default_factory=list)
    rules: list[PlaybookRule] = Field(default_factory=list)

    @field_validator("required_clauses", mode="before")
    @classmethod
    def _coerce_clause_types(cls, value):
        if not value:
            return []
        out = []
        for item in value:
            if isinstance(item, ClauseType):
                out.append(item)
            else:
                try:
                    out.append(ClauseType(str(item).lower()))
                except ValueError:
                    logger.warning("playbook 中出现未知条款类型: %s（已忽略）", item)
        return out


def load_playbook(path: str | Path | None = None) -> Playbook:
    """从 YAML 加载 playbook；文件不存在时返回内置默认值。"""
    settings = get_settings()
    candidate = Path(path) if path else Path(settings.configs_dir) / DEFAULT_PLAYBOOK

    if not candidate.exists():
        logger.warning("未找到 playbook %s，使用内置默认规则", candidate)
        return Playbook(
            name="builtin",
            description="内置兜底规则：仅校验核心条款是否齐备",
            required_clauses=[
                ClauseType.PAYMENT,
                ClauseType.TERM,
                ClauseType.TERMINATION,
                ClauseType.CONFIDENTIALITY,
                ClauseType.LIABILITY,
                ClauseType.GOVERNING_LAW,
                ClauseType.DISPUTE,
            ],
        )

    raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    playbook = Playbook.model_validate(raw)
    logger.info("已加载 playbook: %s v%s（%d 条规则）", playbook.name, playbook.version, len(playbook.rules))
    return playbook


@lru_cache(maxsize=4)
def get_playbook(path: str | None = None) -> Playbook:
    return load_playbook(path)


# -------------------------------------------------------------------- Agent
class ComplianceCheckerAgent(BaseAgent):
    """基于 playbook 的确定性合规校验。"""

    name = "compliance_checker"
    description = "按企业 playbook 校验必备条款与红线表述"

    def __init__(self, *args, playbook: Playbook | None = None, **kwargs) -> None:
        # 合规检查不依赖 LLM，因此不强制初始化 chat model
        super().__init__(*args, **kwargs)
        self.playbook = playbook or get_playbook()

    def run(self, state: dict) -> dict:
        clauses: list[Clause] = list(state.get("clauses") or [])
        coverage = self.check_coverage(clauses)

        findings: list[ComplianceFinding] = []
        issues: list[RiskIssue] = []
        doc_id: str = state.get("doc_id") or "unknown"

        # 1) 必备条款缺失
        for clause_type in self.playbook.required_clauses:
            passed = clause_type in coverage.present
            findings.append(
                ComplianceFinding(
                    rule_id=f"required:{clause_type.value}",
                    requirement=f"合同应包含「{clause_type.value}」条款",
                    passed=passed,
                    severity=Severity.HIGH if not passed else Severity.INFO,
                    detail="" if passed else "未在合同中识别到该条款，建议补充。",
                    related_clause_type=clause_type,
                )
            )
            if not passed:
                issues.append(
                    RiskIssue(
                        issue_id=f"iss_missing_{clause_type.value}",
                        doc_id=doc_id,
                        clause_type=clause_type,
                        category=IssueCategory.MISSING_CLAUSE,
                        severity=Severity.HIGH,
                        title=f"缺失必备条款：{clause_type.value}",
                        description=f"playbook `{self.playbook.name}` 要求合同必须包含该条款。",
                        suggestion=f"补充「{clause_type.value}」条款并明确其内容。",
                        reviewer=self.name,
                        confidence=1.0,
                    )
                )

        # 2) 禁止 / 必备表述
        for rule in self.playbook.rules:
            finding = self._evaluate_rule(rule, clauses)
            findings.append(finding)
            if not finding.passed:
                issues.append(
                    RiskIssue(
                        issue_id=f"iss_rule_{rule.id}",
                        doc_id=doc_id,
                        clause_type=rule.required_clause or ClauseType.OTHER,
                        category=IssueCategory.COMPLIANCE_RISK,
                        severity=rule.severity,
                        title=f"合规规则未通过：{rule.id}",
                        description=finding.detail or rule.requirement,
                        suggestion=rule.message or "请按 playbook 要求调整条款措辞。",
                        reviewer=self.name,
                        confidence=1.0,
                    )
                )

        result = ComplianceResult(findings=findings)
        logger.info(
            "[%s] 规则 %d 条，未通过 %d 条；条款覆盖率 %.2f",
            self.name,
            len(findings),
            len(result.failed),
            coverage.coverage_ratio,
        )
        return {
            "compliance": result,
            "coverage": coverage,
            "issues": issues,
            "trace": [
                {
                    "node": "check_compliance",
                    "agent": self.name,
                    "rules": len(findings),
                    "failed": len(result.failed),
                    "coverage": coverage.coverage_ratio,
                }
            ],
        }

    # ------------------------------------------------------------ 覆盖度
    def check_coverage(self, clauses: list[Clause]) -> ClauseCoverage:
        """统计 playbook 要求条款的覆盖情况。"""
        present = {c.type for c in clauses}
        required = list(self.playbook.required_clauses) or list(
            t for t in present if t != ClauseType.OTHER
        )
        found = [t for t in required if t in present]
        missing = [t for t in required if t not in present]
        return ClauseCoverage(
            present=found,
            missing=missing,
            coverage_ratio=round(len(found) / len(required), 4) if required else 1.0,
        )

    # ------------------------------------------------------------ 规则求值
    def _evaluate_rule(self, rule: PlaybookRule, clauses: list[Clause]) -> ComplianceFinding:
        scope = clauses
        if rule.applies_to:
            scope = [c for c in clauses if c.type in rule.applies_to]

        haystack = "\n".join(c.text for c in scope)

        for pattern in rule.forbidden_patterns:
            if not scope:
                detail = "目标条款缺失，无法校验该规则。"
                return ComplianceFinding(
                    rule_id=rule.id,
                    requirement=rule.requirement,
                    passed=False,
                    severity=rule.severity,
                    detail=detail,
                    related_clause_type=rule.required_clause,
                )
            if _search(pattern, haystack):
                return ComplianceFinding(
                    rule_id=rule.id,
                    requirement=rule.requirement,
                    passed=False,
                    severity=rule.severity,
                    detail=f"命中禁止表述：`{pattern}`",
                    related_clause_type=rule.required_clause,
                )

        for pattern in rule.required_patterns:
            if not _search(pattern, haystack):
                return ComplianceFinding(
                    rule_id=rule.id,
                    requirement=rule.requirement,
                    passed=False,
                    severity=rule.severity,
                    detail=f"缺失必备表述：`{pattern}`",
                    related_clause_type=rule.required_clause,
                )

        return ComplianceFinding(
            rule_id=rule.id,
            requirement=rule.requirement,
            passed=True,
            severity=Severity.INFO,
            detail="通过",
            related_clause_type=rule.required_clause,
        )


def _search(pattern: str, haystack: str) -> bool:
    """先用正则，正则非法时退化为子串匹配。"""
    try:
        return re.search(pattern, haystack, re.IGNORECASE | re.MULTILINE) is not None
    except re.error:
        return pattern.lower() in haystack.lower()


__all__ = [
    "ComplianceCheckerAgent",
    "DEFAULT_PLAYBOOK",
    "Playbook",
    "PlaybookRule",
    "get_playbook",
    "load_playbook",
]
