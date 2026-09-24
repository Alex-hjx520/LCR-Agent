"""风险审查 Agent。

对每条（或重点）条款执行「检索先例 → LLM 比对 → 输出风险项」的三步流程。
检索到的 CUAD 先例与 playbook 红线会作为证据注入提示词，并要求模型在
`citations` 中回指证据编号，从而抑制幻觉、让结论可核验。
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from schemas.clause import Clause, ClauseType
from schemas.retrieval import Corpus, RetrievedPassage
from schemas.review import Citation, IssueCategory, RiskIssue, Severity

logger = logging.getLogger(__name__)

#: 默认不送审的条款类型（信息性条款，风险低且数量多）
SKIP_TYPES: frozenset[ClauseType] = frozenset(
    {ClauseType.DEFINE, ClauseType.PARTIES, ClauseType.NOTICE, ClauseType.OTHER}
)


class ClauseRisk(BaseModel):
    """LLM 对单条条款给出的风险判断。"""

    severity: Severity = Field(default=Severity.INFO)
    category: IssueCategory = Field(default=IssueCategory.UNFAVORABLE_TERM)
    title: str = ""
    description: str = ""
    suggestion: str = ""
    evidence_refs: list[int] = Field(default_factory=list, description="引用的证据编号，如 [1,3]")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ClauseRiskBatch(BaseModel):
    findings: list[ClauseRisk] = Field(default_factory=list)


class RiskReviewerAgent(BaseAgent):
    """逐条款识别法律/商业风险并给出修改建议。"""

    name = "risk_reviewer"
    description = "识别条款中的不利约定、责任暴露与歧义表述"

    max_clauses: int = 40
    min_severity: Severity = Severity.LOW

    retrieval_corpora = [Corpus.CUAD, Corpus.PLAYBOOK]

    @property
    def system_prompt(self) -> str:
        return (
            "你是负责合同风险审查的资深律师，代表委托方（通常是付款方/采购方）审阅合同。\n"
            "请针对给定条款输出风险发现，要求：\n"
            "1. 只依据条款原文与提供的先例证据判断，不要臆测未写明的内容；\n"
            "2. severity 取值：info/low/medium/high/critical。"
            "单方不利、无限责任、无条件单方解除等属 high 或 critical；\n"
            "3. category 取值：missing_clause / unfavorable_term / ambiguity / "
            "liability_exposure / compliance_risk / internal_conflict / unusual_term；\n"
            "4. suggestion 必须是可落地的修改建议（替换措辞或补充条款），不要写空话；\n"
            "5. evidence_refs 只能引用提示词中实际出现的证据编号；无证据时留空数组；\n"
            "6. 若条款表述公允、无明显风险，返回空 findings 数组，不要为凑数而编造问题。"
        )

    # ------------------------------------------------------------ 主流程
    def run(self, state: dict) -> dict:
        clauses: list[Clause] = list(state.get("clauses") or [])
        doc_id: str = state.get("doc_id") or "unknown"
        use_retrieval: bool = bool(state.get("use_retrieval", True))

        targets = self.select_clauses(clauses)
        if not targets:
            return {"issues": [], "trace": [self._trace("review_risks", reviewed=0, issues=0)]}

        issues: list[RiskIssue] = []
        total_prompt = 0
        total_completion = 0

        for clause in targets:
            passages = (
                self.retrieve(f"{clause.title or ''}\n{clause.text[:600]}") if use_retrieval else []
            )
            parsed, result = self.invoke_structured(
                ClauseRiskBatch,
                user_prompt=self._build_prompt(clause, passages),
                system_prompt=self.system_prompt,
            )
            total_prompt += result.prompt_tokens
            total_completion += result.completion_tokens

            if parsed is None:
                logger.warning("[%s] 条款 %s 审查失败，跳过", self.name, clause.clause_id)
                continue

            for risk in parsed.findings:
                if risk.severity.weight < self.min_severity.weight:
                    continue
                issues.append(self._to_issue(doc_id, clause, risk, passages))

        logger.info("[%s] 审查 %d 条条款，产出 %d 条风险", self.name, len(targets), len(issues))
        return {
            "issues": issues,
            "trace": [
                self._trace(
                    "review_risks",
                    reviewed=len(targets),
                    issues=len(issues),
                    prompt_tokens=total_prompt,
                    completion_tokens=total_completion,
                )
            ],
        }

    # ------------------------------------------------------------ 条款筛选
    def select_clauses(self, clauses: list[Clause]) -> list[Clause]:
        """按风险优先级排序并截断，避免无上限的 LLM 调用。"""
        relevant = [c for c in clauses if c.type not in SKIP_TYPES]
        relevant.sort(key=lambda c: (not c.is_critical, -c.confidence, c.order))
        return relevant[: self.max_clauses]

    # -------------------------------------------------------------- 提示词
    def _build_prompt(self, clause: Clause, passages: list[RetrievedPassage]) -> str:
        return "\n".join(
            [
                f"【条款类型】{clause.type.value}",
                f"【条款标题】{clause.title or '(无)'}",
                "【条款原文】",
                clause.text.strip()[:3000],
                "",
                "【可参考的先例 / 红线条款】",
                self.format_evidence(passages, max_items=5),
                "",
                "请判断该条款存在的风险，并按 ClauseRiskBatch 结构输出 findings 数组。",
                "每条 finding 必须包含 severity / category / title / description / suggestion / "
                "evidence_refs / confidence。",
            ]
        )

    # ---------------------------------------------------------------- 转换
    def _to_issue(
        self,
        doc_id: str,
        clause: Clause,
        risk: ClauseRisk,
        passages: list[RetrievedPassage],
    ) -> RiskIssue:
        citations = [
            Citation(
                corpus=passages[i - 1].corpus,
                ref=passages[i - 1].label,
                text=passages[i - 1].text[:300],
                score=passages[i - 1].score,
            )
            for i in risk.evidence_refs
            if 1 <= i <= len(passages)
        ]
        return RiskIssue(
            issue_id=f"iss_{uuid.uuid4().hex[:10]}",
            doc_id=doc_id,
            clause_id=clause.clause_id,
            clause_type=clause.type,
            category=risk.category,
            severity=risk.severity,
            title=risk.title or f"{clause.title or clause.type.value} 存在风险",
            description=risk.description,
            suggestion=risk.suggestion,
            citations=citations,
            reviewer=self.name,
            confidence=risk.confidence,
        )

    def _trace(self, node: str, **extra) -> dict:
        return {"node": node, "agent": self.name, **extra}


__all__ = ["ClauseRisk", "ClauseRiskBatch", "RiskReviewerAgent", "SKIP_TYPES"]
