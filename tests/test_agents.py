"""Agent 层测试。

全部使用 `ScriptedChatModel` / 规则兜底路径，不产生任何网络调用。
"""

from __future__ import annotations

import pytest

from agents.clause_extractor import ClauseExtraction, ClauseExtractorAgent, ExtractedClause
from agents.compliance_checker import ComplianceCheckerAgent, Playbook, PlaybookRule, load_playbook
from agents.risk_reviewer import ClauseRisk, ClauseRiskBatch, RiskReviewerAgent
from agents.summarizer import SummarizerAgent
from parsing import parse_text
from schemas.clause import Clause, ClauseCoverage, ClauseType
from schemas.review import (
    ComplianceFinding,
    ComplianceResult,
    IssueCategory,
    RiskIssue,
    Severity,
)
from tests.stubs import ExplodingChatModel, ScriptedChatModel


def _clauses(sample_contract: str, llm: ScriptedChatModel | None = None) -> list[Clause]:
    """用规则路径抽取条款，作为其它 Agent 的输入。"""
    agent = ClauseExtractorAgent(llm=llm or ScriptedChatModel())
    return agent.run({"doc_id": "doc_test", "raw_text": sample_contract})["clauses"]


# ------------------------------------------------------------------ 条款抽取
class TestClauseExtractor:
    def test_rule_path_extracts_expected_types(self, sample_contract, stub_llm):
        agent = ClauseExtractorAgent(llm=stub_llm)
        result = agent.run({"doc_id": "doc_test", "raw_text": sample_contract})
        clauses = result["clauses"]

        assert len(clauses) >= 9
        found = {c.type for c in clauses}
        for expected in (
            ClauseType.PAYMENT,
            ClauseType.CONFIDENTIALITY,
            ClauseType.LIABILITY,
            ClauseType.IP,
            ClauseType.GOVERNING_LAW,
            ClauseType.DISPUTE,
        ):
            assert expected in found, f"未识别出 {expected}"

        assert all(c.clause_id.startswith("doc_test-cl") for c in clauses)
        assert all(c.char_end > c.char_start for c in clauses)
        assert all(c.extraction_method == "rule" for c in clauses)

    def test_llm_result_overrides_rule_classification(self, sample_contract):
        llm = ScriptedChatModel(
            structured={
                ClauseExtraction: ClauseExtraction(
                    contract_type="技术服务合同",
                    clauses=[ExtractedClause(index=0, type=ClauseType.SCOPE, confidence=0.93)],
                )
            }
        )
        agent = ClauseExtractorAgent(llm=llm)
        result = agent.run({"doc_id": "doc_x", "raw_text": sample_contract})

        clauses = result["clauses"]
        assert clauses[0].type is ClauseType.SCOPE
        assert clauses[0].confidence == pytest.approx(0.93)
        assert clauses[0].extraction_method == "hybrid"
        assert result["contract_type"] == "技术服务合同"
        # 其余条款仍走规则兜底
        assert any(c.extraction_method == "rule" for c in clauses)

    def test_empty_text_returns_no_clauses(self, stub_llm):
        agent = ClauseExtractorAgent(llm=stub_llm)
        result = agent.run({"doc_id": "d", "raw_text": "   "})
        assert result["clauses"] == []

    def test_segment_without_numbering_returns_single_candidate(self, stub_llm):
        agent = ClauseExtractorAgent(llm=stub_llm)
        candidates = agent.segment("这是一份没有条款编号的简单协议正文。")
        assert len(candidates) == 1
        assert candidates[0]["title"] is None

    def test_segment_collects_preamble(self, stub_llm):
        agent = ClauseExtractorAgent(llm=stub_llm)
        candidates = agent.segment("甲方向乙方采购设备，双方就以下条款达成一致：\n\n第一条 价款\n总价 100 万元。")
        assert candidates[0]["title"] == "前言"
        assert candidates[1]["title"].startswith("第一条")


# ------------------------------------------------------------------ 风险审查
class TestRiskReviewer:
    def test_llm_findings_become_issues(self, sample_contract):
        llm = ScriptedChatModel(
            structured={
                ClauseRiskBatch: ClauseRiskBatch(
                    findings=[
                        ClauseRisk(
                            severity=Severity.CRITICAL,
                            category=IssueCategory.LIABILITY_EXPOSURE,
                            title="完全免责条款",
                            description="乙方不承担任何责任，风险完全由甲方承担。",
                            suggestion="删除完全免责表述，改为赔偿直接损失并设置合理责任上限。",
                            evidence_refs=[1],
                            confidence=0.95,
                        )
                    ]
                )
            }
        )
        agent = RiskReviewerAgent(llm=llm)
        clauses = _clauses(sample_contract)
        result = agent.run({"doc_id": "doc_r", "clauses": clauses, "use_retrieval": False})

        issues = result["issues"]
        assert issues
        assert all(i.severity is Severity.CRITICAL for i in issues)
        assert all(i.reviewer == "risk_reviewer" for i in issues)
        assert issues[0].suggestion

    def test_info_severity_is_filtered(self, sample_contract):
        llm = ScriptedChatModel(
            structured={
                ClauseRiskBatch: ClauseRiskBatch(
                    findings=[ClauseRisk(severity=Severity.INFO, title="无风险")]
                )
            }
        )
        agent = RiskReviewerAgent(llm=llm)
        clauses = _clauses(sample_contract)
        result = agent.run({"doc_id": "doc_r", "clauses": clauses, "use_retrieval": False})
        assert result["issues"] == []

    def test_select_clauses_skips_informational_types(self):
        agent = RiskReviewerAgent(llm=None)
        clauses = [
            Clause(doc_id="d", clause_id="c1", type=ClauseType.DEFINE, text="定义"),
            Clause(doc_id="d", clause_id="c2", type=ClauseType.LIABILITY, text="责任"),
        ]
        selected = agent.select_clauses(clauses)
        assert [c.clause_id for c in selected] == ["c2"]

    def test_no_clauses_yields_no_issues(self, stub_llm):
        agent = RiskReviewerAgent(llm=stub_llm)
        assert agent.run({"doc_id": "d", "clauses": []})["issues"] == []


# ------------------------------------------------------------------ 合规检查
class TestComplianceChecker:
    def test_missing_required_clauses_reported(self, sample_contract):
        checker = ComplianceCheckerAgent(llm=None)
        clauses = _clauses(sample_contract)
        result = checker.run({"doc_id": "doc_c", "clauses": clauses})

        coverage: ClauseCoverage = result["coverage"]
        assert ClauseType.FORCE_MAJEURE in coverage.missing
        assert 0.0 < coverage.coverage_ratio < 1.0

        missing_issues = [i for i in result["issues"] if i.category is IssueCategory.MISSING_CLAUSE]
        assert any(i.clause_type is ClauseType.FORCE_MAJEURE for i in missing_issues)

    def test_forbidden_pattern_flagged(self, sample_contract):
        checker = ComplianceCheckerAgent(llm=None)
        clauses = _clauses(sample_contract)
        result = checker.run({"doc_id": "doc_c", "clauses": clauses})

        findings = {f.rule_id: f for f in result["compliance"].findings}
        assert findings["PAY-002"].passed is False  # 全额预付
        assert findings["IPR-001"].passed is False  # 知识产权归乙方

    def test_inline_playbook_is_honored(self):
        playbook = Playbook(
            name="test",
            required_clauses=[ClauseType.FORCE_MAJEURE],
            rules=[
                PlaybookRule(
                    id="T-1",
                    requirement="必须约定不可抗力",
                    severity=Severity.MEDIUM,
                    required_patterns=["不可抗力"],
                )
            ],
        )
        checker = ComplianceCheckerAgent(llm=None, playbook=playbook)
        clauses = [Clause(doc_id="d", clause_id="c1", type=ClauseType.TERM, text="合同期限两年。")]
        result = checker.run({"doc_id": "d", "clauses": clauses})

        assert result["coverage"].coverage_ratio == 0.0
        assert all(not f.passed for f in result["compliance"].findings)

    def test_default_playbook_loads_from_configs(self):
        playbook = load_playbook()
        assert playbook.rules
        assert ClauseType.PAYMENT in playbook.required_clauses


# -------------------------------------------------------------------- 摘要
class TestSummarizer:
    def test_overall_risk_follows_worst_issue(self):
        assert (
            SummarizerAgent.compute_overall_risk(
                [RiskIssue(issue_id="1", doc_id="d", severity=Severity.CRITICAL, title="x")],
                ComplianceResult(),
                ClauseCoverage(coverage_ratio=1.0),
            )
            is Severity.CRITICAL
        )

    def test_low_coverage_lifts_overall_risk(self):
        overall = SummarizerAgent.compute_overall_risk(
            [], ComplianceResult(), ClauseCoverage(coverage_ratio=0.3)
        )
        assert overall is Severity.MEDIUM

    def test_failed_compliance_raises_risk(self):
        compliance = ComplianceResult(
            findings=[
                ComplianceFinding(
                    rule_id="r1",
                    requirement="x",
                    passed=False,
                    severity=Severity.HIGH,
                    detail="未通过",
                )
            ]
        )
        assert (
            SummarizerAgent.compute_overall_risk([], compliance, ClauseCoverage(coverage_ratio=1.0))
            is Severity.HIGH
        )

    def test_summary_generated_with_llm_text(self, stub_llm):
        agent = SummarizerAgent(llm=stub_llm)
        result = agent.run(
            {
                "doc_id": "doc_s",
                "filename": "c.txt",
                "clauses": [],
                "issues": [],
                "compliance": ComplianceResult(),
            }
        )
        assert result["summary"]
        assert result["overall_risk"] in set(Severity)
        assert "num_clauses" in result["stats"]

    def test_fallback_summary_when_llm_fails(self):
        agent = SummarizerAgent(llm=ExplodingChatModel())
        result = agent.run(
            {"doc_id": "d", "clauses": [], "issues": [], "compliance": ComplianceResult()}
        )
        assert "总体风险等级" in result["summary"]


class TestSchemaSanity:
    def test_severity_worst(self):
        assert Severity.worst([Severity.LOW, Severity.HIGH, Severity.INFO]) is Severity.HIGH
        assert Severity.worst([]) is Severity.INFO

    def test_parse_text_used_by_agents(self, sample_contract):
        doc = parse_text(sample_contract)
        assert doc.num_blocks > 0
