"""测试替身（test doubles）。

放在独立模块而不是 `conftest.py` 中，便于在测试文件中直接 `import`，
同时避免 `conftest` 被重复导入导致的类身份不一致问题。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage

from schemas.review import ComplianceResult, ReviewReport, RiskIssue, Severity

SAMPLE_CONTRACT = """技术服务合同

甲方：三花智控股份有限公司
乙方：某某科技有限责任公司

第一条 服务范围
乙方应为甲方提供电子膨胀阀控制算法的开发与调试服务，交付物包括源代码、
设计文档及测试报告。具体服务内容以附件一《工作说明书》为准。

第二条 合同价款与付款
本合同总价款为人民币 100 万元（含税）。甲方应在合同签订后 10 日内支付
全额预付款，乙方收款后开始履行。

第三条 合同期限
本合同自双方签署之日起生效，有效期两年。

第四条 保密
双方应对履行本合同过程中知悉的对方商业秘密承担保密义务，保密期限为 1 年。

第五条 责任限制
无论因何种原因，乙方不承担任何责任，赔偿上限不超过合同总金额的 10%。

第六条 知识产权
基于本合同产生的定制开发成果，其知识产权均归乙方所有。

第七条 终止
甲方可提前 90 日书面通知解除本合同。

第八条 适用法律
本合同的订立、效力、解释与争议解决均适用香港特别行政区法律。

第九条 争议解决
因本合同产生的争议，提交 ICC 国际仲裁院在新加坡仲裁解决。
"""


class StructuredRunnable:
    """模拟 `llm.with_structured_output(schema)` 返回的 Runnable。"""

    def __init__(self, model: "ScriptedChatModel", schema: Any) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._model.resolve_structured(self._schema, messages)


class ScriptedChatModel:
    """确定性 ChatModel 替身。

    Args:
        structured: `schema -> 实例 | callable(messages) -> 实例`。
            未配置的 schema 会以零参构造函数返回同名 Pydantic 模型实例。
        texts: 纯文本调用的返回值队列（用尽后回落到 `default_text`）。
        default_text: 默认文本回复。
    """

    def __init__(
        self,
        *,
        structured: dict[Any, Any] | None = None,
        texts: list[str] | None = None,
        default_text: str = "这是一份合同的执行摘要。",
    ) -> None:
        self.structured = structured or {}
        self.texts = list(texts or [])
        self.default_text = default_text
        self.calls: list[Any] = []

    def resolve_structured(self, schema: Any, messages: Any) -> Any:
        self.calls.append(messages)
        value = self.structured.get(schema)
        if value is None:
            return schema()
        return value(messages) if callable(value) else value

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredRunnable:
        return StructuredRunnable(self, schema)

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.texts.pop(0) if self.texts else self.default_text)


class ExplodingChatModel(ScriptedChatModel):
    """任何调用都会抛异常的模型，用于验证降级路径。"""

    def invoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        raise RuntimeError("boom: LLM 不可用")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredRunnable:
        class _Boom:
            def invoke(self, messages: Any, **kwargs: Any) -> Any:
                raise RuntimeError("boom: structured output 不可用")

        return _Boom()  # type: ignore[return-value]


class FakeCompiledGraph:
    """模拟 LangGraph 编译产物，让 API 层测试不依赖真实图执行。"""

    def __init__(self, report: ReviewReport | None = None) -> None:
        self.report = report
        self.invocations: list[dict] = []

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        self.invocations.append(state)
        if self.report is not None:
            return {**state, "report": self.report}
        doc_id = state.get("doc_id", "unknown")
        report = ReviewReport(
            report_id=f"rpt_{uuid.uuid4().hex[:8]}",
            doc_id=doc_id,
            filename=state.get("filename", ""),
            summary="桩图执行完成。",
            overall_risk=Severity.MEDIUM,
            issues=[
                RiskIssue(
                    issue_id="iss_stub",
                    doc_id=doc_id,
                    severity=Severity.MEDIUM,
                    title="桩风险项",
                    description="用于 API 层测试。",
                )
            ],
            compliance=ComplianceResult(),
            stats={"num_clauses": 0, "num_issues": 1},
        )
        return {**state, "report": report, "overall_risk": Severity.MEDIUM}


__all__ = [
    "SAMPLE_CONTRACT",
    "ExplodingChatModel",
    "FakeCompiledGraph",
    "ScriptedChatModel",
    "StructuredRunnable",
]
