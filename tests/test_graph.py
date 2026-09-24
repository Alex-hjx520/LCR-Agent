"""LangGraph 编排的端到端测试（使用桩模型，无外部依赖）。"""

from __future__ import annotations

import pytest

from graph import build_review_graph, initial_state, run_review
from graph.nodes import route_after_extract, route_after_ingest
from parsing import parse_text
from schemas.clause import ClauseType
from schemas.review import IssueCategory, Severity
from tests.stubs import ScriptedChatModel


@pytest.fixture
def graph(settings, stub_llm):
    """使用桩模型、关闭检索器的编译图。"""
    return build_review_graph(llm=stub_llm, retriever=None, settings=settings)


class TestInitialState:
    def test_reducer_lists_are_initialized(self):
        state = initial_state(raw_text="x", filename="f.txt")
        assert state["issues"] == []
        assert state["errors"] == []
        assert state["trace"] == []
        assert state["use_retrieval"] is True

    def test_optional_fields_are_omitted(self):
        state = initial_state(raw_text="x")
        assert "top_k" not in state
        assert "max_clauses" not in state

    def test_extra_kwargs_go_into_config(self):
        state = initial_state(raw_text="x", contract_type="采购合同")
        assert state["config"] == {"contract_type": "采购合同"}


class TestRouting:
    def test_route_after_ingest_goes_to_fail_on_error(self):
        assert route_after_ingest({"errors": ["boom"]}) == "fail"
        assert route_after_ingest({"errors": []}) == "chunk"
        assert route_after_ingest({}) == "chunk"

    def test_route_after_extract_skips_review_without_clauses(self):
        assert route_after_extract({"clauses": []}) == "summarize"
        assert route_after_extract({"clauses": [object()]}) == "review"


class TestEndToEnd:
    def test_full_pipeline_produces_report(self, graph, sample_contract):
        report = run_review(
            graph=graph,
            raw_text=sample_contract,
            filename="contract.txt",
            use_retrieval=False,
        )

        assert report.clauses, "应抽取到条款"
        assert report.filename == "contract.txt"
        assert report.summary
        assert report.stats["num_clauses"] == len(report.clauses)
        assert report.stats["num_issues"] == len(report.issues)

    def test_compliance_findings_reach_the_report(self, graph, sample_contract):
        report = run_review(graph=graph, raw_text=sample_contract, use_retrieval=False)

        assert report.coverage.missing, "样例合同缺少部分必备条款"
        assert ClauseType.FORCE_MAJEURE in report.coverage.missing
        assert report.compliance.findings

        missing = [i for i in report.issues if i.category is IssueCategory.MISSING_CLAUSE]
        assert missing, "缺失必备条款应转化为风险项"

    def test_overall_risk_reflects_compliance_failures(self, graph, sample_contract):
        report = run_review(graph=graph, raw_text=sample_contract, use_retrieval=False)
        # 样例合同含有「完全免责」「全额预付」「知识产权归乙方」等红线问题
        assert report.overall_risk.weight >= Severity.HIGH.weight

    def test_trace_records_every_node(self, graph, sample_contract):
        report = run_review(graph=graph, raw_text=sample_contract, use_retrieval=False)
        nodes = {entry["node"] for entry in report.trace}
        for expected in ("ingest", "chunk", "extract", "review", "compliance", "summarize", "finalize"):
            assert expected in nodes, f"trace 缺少节点 {expected}"

    def test_graph_does_not_mutate_caller_state(self, graph, sample_contract):
        state = initial_state(raw_text=sample_contract, filename="c.txt", use_retrieval=False)
        graph.invoke(state, config={"configurable": {"thread_id": "t1"}})
        assert state["issues"] == []  # 原始 state 不应被就地修改

    def test_repeated_runs_do_not_accumulate_issues(self, graph, sample_contract):
        first = run_review(graph=graph, raw_text=sample_contract, use_retrieval=False)
        second = run_review(graph=graph, raw_text=sample_contract, use_retrieval=False)
        assert len(first.issues) == len(second.issues)


class TestFailurePath:
    def test_empty_input_routes_to_fail_and_finalize(self, graph):
        report = run_review(graph=graph, raw_text="", use_retrieval=False)
        assert "解析失败" in report.summary
        assert report.clauses == []

    def test_no_clauses_skips_review_node(self, settings, stub_llm):
        compiled = build_review_graph(llm=stub_llm, retriever=None, settings=settings)
        # 一段没有条款编号、也不含任何关键词的文本
        report = run_review(
            graph=compiled,
            raw_text="感谢您选择我们的产品，祝您使用愉快。",
            use_retrieval=False,
        )
        nodes = {entry["node"] for entry in report.trace}
        assert "summarize" in nodes
        assert "review" not in nodes or any(
            entry.get("reviewed") == 0 for entry in report.trace if entry["node"] == "review"
        )


class TestGraphWiring:
    def test_build_is_deterministic(self, settings, stub_llm):
        g1 = build_review_graph(llm=stub_llm, retriever=None, settings=settings)
        g2 = build_review_graph(llm=stub_llm, retriever=None, settings=settings)
        assert type(g1) is type(g2)

    def test_parse_text_feeds_graph(self, graph, sample_contract):
        doc = parse_text(sample_contract)
        report = run_review(graph=graph, raw_text=doc.raw_text, use_retrieval=False)
        assert report.clauses
