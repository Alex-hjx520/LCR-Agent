"""摘要与总体结论 Agent。

职责：
1. 汇总条款覆盖度、风险分布，给出总体风险等级（由规则计算，不交给 LLM 猜测）；
2. 让 LLM 生成面向业务方的中文执行摘要（Executive Summary）；
3. LLM 不可用时退化为模板化摘要，保证接口输出结构稳定。
"""

from __future__ import annotations

import logging
from collections import Counter

from agents.base import BaseAgent
from schemas.clause import Clause, ClauseCoverage
from schemas.review import ComplianceResult, RiskIssue, Severity

logger = logging.getLogger(__name__)

_SEVERITY_ZH = {
    Severity.CRITICAL: "严重",
    Severity.HIGH: "高",
    Severity.MEDIUM: "中",
    Severity.LOW: "低",
    Severity.INFO: "提示",
}


class SummarizerAgent(BaseAgent):
    """生成审查摘要与总体风险等级。"""

    name = "summarizer"
    description = "汇总审查结论并生成执行摘要"

    #: 摘要提示词中最多列举的风险条数
    max_issues_in_prompt: int = 20

    @property
    def system_prompt(self) -> str:
        return (
            "你是向管理层汇报合同审查结论的法务负责人。\n"
            "请用中文撰写 3~6 句的执行摘要，结构为：\n"
            "1) 合同概况与整体风险判断；2) 最需要关注的 2~3 个问题及其商业影响；\n"
            "3) 建议的下一步动作（修改、谈判或可直接签署）。\n"
            "要求：语言精炼、避免法律术语堆砌，不要重复罗列明细，不要编造未提供的信息。"
        )

    def run(self, state: dict) -> dict:
        clauses: list[Clause] = list(state.get("clauses") or [])
        issues: list[RiskIssue] = list(state.get("issues") or [])
        compliance: ComplianceResult = state.get("compliance") or ComplianceResult()
        coverage: ClauseCoverage = state.get("coverage") or self._coverage(clauses)

        overall = self.compute_overall_risk(issues, compliance, coverage)
        stats = self.build_stats(clauses, issues, compliance, coverage)

        summary, result = self.invoke_text(
            user_prompt=self._build_prompt(state, issues, compliance, coverage, overall, stats)
        )
        if not result.ok or not summary.strip():
            summary = self.fallback_summary(issues, compliance, coverage, overall, stats)

        return {
            "summary": summary.strip(),
            "overall_risk": overall,
            "coverage": coverage,
            "stats": stats,
            "trace": [
                {
                    "node": "summarize",
                    "agent": self.name,
                    "overall_risk": overall.value,
                    "llm_ok": result.ok,
                    "latency_ms": result.latency_ms,
                }
            ],
        }

    # -------------------------------------------------------------- 计算
    @staticmethod
    def compute_overall_risk(
        issues: list[RiskIssue],
        compliance: ComplianceResult,
        coverage: ClauseCoverage,
    ) -> Severity:
        """总体风险 = 风险项最高等级，并按合规失败 / 覆盖度缺口向上修正。"""
        severities = [i.severity for i in issues]
        severities.extend(f.severity for f in compliance.failed if f.severity != Severity.INFO)
        overall = Severity.worst(severities)

        if overall.weight < Severity.MEDIUM.weight and coverage.coverage_ratio < 0.6:
            overall = Severity.MEDIUM
        return overall

    @staticmethod
    def build_stats(
        clauses: list[Clause],
        issues: list[RiskIssue],
        compliance: ComplianceResult,
        coverage: ClauseCoverage,
    ) -> dict:
        hist = Counter(i.severity.value for i in issues)
        return {
            "num_clauses": len(clauses),
            "num_issues": len(issues),
            "severity_histogram": {s.value: hist.get(s.value, 0) for s in Severity},
            "blocking_issues": sum(1 for i in issues if i.is_blocking),
            "compliance_failed": len(compliance.failed),
            "compliance_pass_ratio": compliance.pass_ratio,
            "coverage_ratio": coverage.coverage_ratio,
            "missing_clauses": [c.value for c in coverage.missing],
            "clause_type_histogram": dict(Counter(c.type.value for c in clauses)),
        }

    @staticmethod
    def _coverage(clauses: list[Clause]) -> ClauseCoverage:
        present = sorted({c.type for c in clauses}, key=lambda t: t.value)
        return ClauseCoverage(present=present, missing=[], coverage_ratio=1.0 if present else 0.0)

    # -------------------------------------------------------------- 提示词
    def _build_prompt(
        self,
        state: dict,
        issues: list[RiskIssue],
        compliance: ComplianceResult,
        coverage: ClauseCoverage,
        overall: Severity,
        stats: dict,
    ) -> str:
        lines = [
            f"【文档】{state.get('filename') or state.get('doc_id', 'unknown')}",
            f"【合同类型】{state.get('contract_type', 'unknown')}",
            f"【条款数量】{stats['num_clauses']}",
            f"【条款覆盖度】{coverage.coverage_ratio:.0%}"
            + (f"，缺失：{', '.join(stats['missing_clauses'])}" if coverage.missing else ""),
            f"【合规检查】未通过 {stats['compliance_failed']} 项，通过率 {compliance.pass_ratio:.0%}",
            f"【风险分布】{stats['severity_histogram']}",
            f"【系统判定的总体风险】{_SEVERITY_ZH[overall]}({overall.value})",
            "",
            "【重点风险清单】",
        ]
        ranked = sorted(issues, key=lambda i: -i.severity.weight)[: self.max_issues_in_prompt]
        if not ranked:
            lines.append("（未发现风险项）")
        for i, issue in enumerate(ranked, start=1):
            lines.append(
                f"{i}. [{_SEVERITY_ZH[issue.severity]}] ({issue.clause_type.value}) "
                f"{issue.title} —— {issue.description[:120]}"
            )
        lines.append("")
        lines.append("请给出执行摘要（纯文本，不要 Markdown 标题，不要 JSON）。")
        return "\n".join(lines)

    @staticmethod
    def fallback_summary(
        issues: list[RiskIssue],
        compliance: ComplianceResult,
        coverage: ClauseCoverage,
        overall: Severity,
        stats: dict,
    ) -> str:
        blocking = [i for i in issues if i.is_blocking]
        parts = [
            f"本次共识别条款 {stats['num_clauses']} 条，合规规则通过率 {compliance.pass_ratio:.0%}，"
            f"条款覆盖度 {coverage.coverage_ratio:.0%}。",
            f"共发现 {stats['num_issues']} 项风险，其中高风险及以上 {len(blocking)} 项，"
            f"总体风险等级判定为「{_SEVERITY_ZH[overall]}」。",
        ]
        if blocking:
            top = "；".join(i.title for i in blocking[:3])
            parts.append(f"需优先处理的条款：{top}。建议完成修改后再行签署。")
        else:
            parts.append("未发现阻断性风险，可在完成常规商务确认后签署。")
        if coverage.missing:
            parts.append("同时建议补充缺失条款：" + "、".join(c.value for c in coverage.missing) + "。")
        return " ".join(parts)


__all__ = ["SummarizerAgent"]
