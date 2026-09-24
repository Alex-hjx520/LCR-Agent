"""条款（Clause）相关的数据模型。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    """条款类型，参考 CUAD 的 41 类关键条款做归并。"""

    DEFINE = "definition"              # 定义
    PARTIES = "parties"                # 缔约方
    SCOPE = "scope_of_work"            # 工作范围
    PAYMENT = "payment"                # 付款 / 价款
    PRICE_ADJUST = "price_adjustment"  # 价格调整
    TERM = "term"                      # 合同期限
    RENEWAL = "renewal"                # 续约
    TERMINATION = "termination"        # 解除 / 终止
    CONFIDENTIALITY = "confidentiality"  # 保密
    LIABILITY = "liability"            # 责任限制
    INDEMNIFICATION = "indemnification"  # 赔偿
    IP = "ip_ownership"                # 知识产权
    WARRANTY = "warranty"              # 陈述与保证
    NON_COMPETE = "non_compete"        # 竞业限制
    EXCLUSIVITY = "exclusivity"        # 排他性
    ASSIGNMENT = "assignment"          # 权利义务转让
    CHANGE_OF_CONTROL = "change_of_control"
    FORCE_MAJEURE = "force_majeure"    # 不可抗力
    GOVERNING_LAW = "governing_law"    # 适用法律
    DISPUTE = "dispute_resolution"     # 争议解决
    NOTICE = "notice"                  # 通知
    AUDIT = "audit_rights"             # 审计权
    INSURANCE = "insurance"
    COMPLIANCE = "compliance"          # 合规 / 反腐败
    DATA_PROTECTION = "data_protection"
    OTHER = "other"


#: 关键条款类型（缺失即视为高风险，由 configs/playbook.yaml 覆盖）
CRITICAL_CLAUSE_TYPES: frozenset[ClauseType] = frozenset(
    {
        ClauseType.PAYMENT,
        ClauseType.TERM,
        ClauseType.TERMINATION,
        ClauseType.CONFIDENTIALITY,
        ClauseType.LIABILITY,
        ClauseType.GOVERNING_LAW,
        ClauseType.DISPUTE,
    }
)


class Clause(BaseModel):
    """从合同中抽取出的一条条款。"""

    clause_id: str
    doc_id: str
    type: ClauseType = ClauseType.OTHER
    title: str | None = None
    text: str
    order: int = 0
    char_start: int = 0
    char_end: int = 0
    page: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    extraction_method: str = "rule"  # rule | llm | hybrid

    @property
    def is_critical(self) -> bool:
        return self.type in CRITICAL_CLAUSE_TYPES

    @property
    def char_count(self) -> int:
        return len(self.text)


class ClauseCoverage(BaseModel):
    """条款覆盖度统计。"""

    present: list[ClauseType] = Field(default_factory=list)
    missing: list[ClauseType] = Field(default_factory=list)
    coverage_ratio: float = 0.0


__all__ = ["CRITICAL_CLAUSE_TYPES", "Clause", "ClauseCoverage", "ClauseType"]
