"""条款抽取 Agent。

两阶段策略（hybrid），兼顾召回率与成本：

1. **规则切分**：用正则识别 `第X条` / `1.2 xxx` / `ARTICLE 5` 等条款边界，
   把合同正文切成候选条款段，保证不遗漏、不越界、可回溯字符偏移；
2. **LLM 分类**：把候选条款标题 + 首段文本批量送入 LLM，判断 `ClauseType`
   并输出置信度；LLM 不可用或失败时退化为关键词规则分类。

这样既避免了「让 LLM 从 5 万字合同里直接抽取条款」带来的不稳定与高成本，
又保留了语义分类能力。
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Iterable

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from parsing.base import HEADING_RE
from schemas.clause import Clause, ClauseType

logger = logging.getLogger(__name__)

#: 条款起始行（比 HEADING_RE 更宽松，允许「第X条」后直接跟正文）
_CLAUSE_START_RE = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9一二三四五六七八九十百千]+\s*条"
    r"|第\s*[0-9一二三四五六七八九十百千]+\s*[章节]"
    r"|(?:ART(?:ICLE)?|Section|Clause)\s+[0-9IVXLC]+"
    r"|[0-9]{1,2}(?:\.[0-9]{1,2}){0,2}[\.、]\s*\S"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

#: 关键词 → 条款类型（LLM 不可用时的兜底分类器）
KEYWORD_RULES: tuple[tuple[ClauseType, tuple[str, ...]], ...] = (
    (ClauseType.GOVERNING_LAW, ("适用法律", "管辖法律", "governing law", "准据法")),
    (ClauseType.DISPUTE, ("争议解决", "仲裁", "arbitration", "诉讼", "jurisdiction")),
    (ClauseType.CONFIDENTIALITY, ("保密", "confidential", "nda")),
    (ClauseType.LIABILITY, ("责任限制", "责任上限", "limitation of liability", "liability cap")),
    (ClauseType.INDEMNIFICATION, ("赔偿", "补偿", "indemnif", "hold harmless")),
    (ClauseType.TERMINATION, ("解除", "终止", "termination", "terminate")),
    (ClauseType.TERM, ("合同期限", "有效期", "term of", "duration")),
    (ClauseType.RENEWAL, ("续约", "续期", "renewal", "renew")),
    (ClauseType.PAYMENT, ("付款", "价款", "支付", "payment", "fees", "invoice")),
    (ClauseType.PRICE_ADJUST, ("价格调整", "调价", "price adjustment")),
    (ClauseType.IP, ("知识产权", "著作权", "专利", "intellectual property", "ip ownership")),
    (ClauseType.CONFIDENTIALITY, ("商业秘密", "trade secret")),
    (ClauseType.NON_COMPETE, ("竞业", "不竞争", "non-compete", "non-solicit")),
    (ClauseType.EXCLUSIVITY, ("排他", "独家", "exclusivity", "exclusive")),
    (ClauseType.ASSIGNMENT, ("转让", "assignment", "权利义务的转让")),
    (ClauseType.CHANGE_OF_CONTROL, ("控制权变更", "change of control")),
    (ClauseType.FORCE_MAJEURE, ("不可抗力", "force majeure")),
    (ClauseType.WARRANTY, ("陈述与保证", "保证", "warranty", "representation")),
    (ClauseType.INSURANCE, ("保险", "insurance")),
    (ClauseType.AUDIT, ("审计", "检查权", "audit")),
    (ClauseType.COMPLIANCE, ("反腐败", "合规", "anti-bribery", "compliance")),
    (ClauseType.DATA_PROTECTION, ("个人信息", "数据保护", "data protection", "gdpr")),
    (ClauseType.NOTICE, ("通知", "notice")),
    (ClauseType.SCOPE, ("服务范围", "工作范围", "scope of work", "scope of services")),
    (ClauseType.PARTIES, ("甲方", "乙方", "缔约方", "parties")),
    (ClauseType.DEFINE, ("定义", "释义", "definition", "interpretation")),
)


# --------------------------------------------------------------------- 模型
class ExtractedClause(BaseModel):
    """LLM 输出的单条条款分类结果。"""

    index: int = Field(description="候选条款的序号，必须与输入一致")
    type: ClauseType = Field(default=ClauseType.OTHER, description="条款类型")
    title: str | None = Field(default=None, description="规范化后的条款标题")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ClauseExtraction(BaseModel):
    clauses: list[ExtractedClause] = Field(default_factory=list)
    contract_type: str = Field(default="unknown", description="合同类型，如 采购合同/保密协议")


class ClauseExtractorAgent(BaseAgent):
    """把合同文本切分并分类为结构化条款列表。"""

    name = "clause_extractor"
    description = "识别合同条款边界并归类到预定义的条款类型"

    #: 单次送入 LLM 的候选条款数量
    batch_size: int = 25
    #: 单条候选条款送入 LLM 的最大字符数（标题 + 摘要）
    preview_chars: int = 200

    @property
    def system_prompt(self) -> str:
        types = ", ".join(t.value for t in ClauseType)
        return (
            "你是资深法务，负责对合同条款进行结构化归类。\n"
            f"可选条款类型（必须严格从中选择）: {types}\n"
            "要求：\n"
            "1. 只做分类与标题规范化，不要改写或总结条款正文；\n"
            "2. 一条候选条款只能对应一个类型；无法判断时用 other；\n"
            "3. confidence 表示你的把握程度（0~1），含糊或多主题的条款应给出较低值；\n"
            "4. 同时判断合同类型（如 采购合同 / 保密协议 / 服务协议 / 劳动合同）。"
        )

    # ------------------------------------------------------------ 主流程
    def run(self, state: dict) -> dict:
        text: str = state.get("raw_text") or ""
        doc_id: str = state.get("doc_id") or f"doc_{uuid.uuid4().hex[:8]}"
        if not text.strip():
            return {"clauses": [], "trace": [self._trace("no_text")]}

        candidates = self.segment(text)
        logger.info("[%s] 规则切分得到 %d 条候选条款", self.name, len(candidates))

        classified, result = self._classify(candidates)
        clauses = [
            Clause(
                clause_id=f"{doc_id}-cl{index:04d}",
                doc_id=doc_id,
                type=item.type,
                title=item.title or candidates[index].get("title"),
                text=candidates[index]["text"],
                order=index,
                char_start=candidates[index]["char_start"],
                char_end=candidates[index]["char_end"],
                page=candidates[index].get("page"),
                confidence=item.confidence,
                extraction_method="hybrid" if result.ok else "rule",
            )
            for index, item in enumerate(classified)
            if index < len(candidates)
        ]

        return {
            "clauses": clauses,
            "contract_type": getattr(classified, "contract_type", "unknown") or "unknown",
            "trace": [
                self._trace(
                    "extract_clauses",
                    candidates=len(candidates),
                    clauses=len(clauses),
                    llm_ok=result.ok,
                )
            ],
        }

    # ------------------------------------------------------------ 规则切分
    def segment(self, text: str) -> list[dict]:
        """把合同正文按条款边界切分为候选段落。"""
        lines = text.splitlines()
        starts: list[tuple[int, str]] = []
        offset = 0
        line_offsets: list[int] = []

        for line in lines:
            line_offsets.append(offset)
            if _CLAUSE_START_RE.match(line):
                starts.append((offset, line.strip()))
            offset += len(line) + 1

        if not starts:
            # 无显式条款编号：整段正文作为一条「条款」
            return [self._make_candidate(text, 0)] if text.strip() else []

        candidates: list[dict] = []
        # 编号条款之前的前言部分（当事人、签署页信息等）
        if starts[0][0] > 0:
            preamble = text[: starts[0][0]].strip()
            if len(preamble) > 30:
                candidates.append(self._make_candidate(preamble, 0, "前言"))

        for i, (start, title) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            body = text[start:end].strip()
            if len(body) < 5:
                continue
            candidates.append(self._make_candidate(body, start, title, char_end=end))
        return candidates

    @staticmethod
    def _make_candidate(
        body: str, char_start: int, title: str | None = None, char_end: int | None = None
    ) -> dict:
        return {
            "text": body,
            "title": _clean_title(title) if title else None,
            "char_start": char_start,
            "char_end": char_end if char_end is not None else char_start + len(body),
            "page": None,
        }

    # ------------------------------------------------------------ LLM 分类
    def _classify(self, candidates: list[dict]) -> tuple[list[ExtractedClause], object]:
        results: list[ExtractedClause] = []
        last_result = None

        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            parsed, last_result = self.invoke_structured(
                ClauseExtraction, user_prompt=self._build_batch_prompt(batch, start)
            )
            if parsed is None:
                results.extend(self._rule_classify_batch(batch, start))
                continue

            by_index = {item.index: item for item in parsed.clauses}
            for offset in range(len(batch)):
                index = start + offset
                results.append(by_index.get(index) or self._rule_classify(batch[offset], index))

        return results, last_result or type("R", (), {"ok": False})()

    def _build_batch_prompt(self, batch: list[dict], start_index: int) -> str:
        lines = ["以下是合同的候选条款清单，请逐条分类：", ""]
        for offset, item in enumerate(batch):
            preview = re.sub(r"\s+", " ", item["text"])[: self.preview_chars]
            lines.append(f"[{start_index + offset}] 标题: {item['title'] or '(无)'}")
            lines.append(f"      正文: {preview}")
        lines.append("")
        lines.append(f"请输出长度为 {len(batch)} 的 clauses 数组，index 必须与上面的序号一一对应。")
        return "\n".join(lines)

    # ------------------------------------------------------------ 规则兜底
    def _rule_classify_batch(self, batch: list[dict], start: int) -> list[ExtractedClause]:
        return [self._rule_classify(item, start + i) for i, item in enumerate(batch)]

    def _rule_classify(self, candidate: dict, index: int) -> ExtractedClause:
        haystack = f"{candidate.get('title') or ''} {candidate['text'][:300]}".lower()
        for clause_type, keywords in KEYWORD_RULES:
            if any(kw.lower() in haystack for kw in keywords):
                return ExtractedClause(
                    index=index, type=clause_type, title=candidate.get("title"), confidence=0.4
                )
        return ExtractedClause(
            index=index, type=ClauseType.OTHER, title=candidate.get("title"), confidence=0.2
        )

    # ----------------------------------------------------------------- 工具
    def _trace(self, node: str, **extra) -> dict:
        return {"node": node, "agent": self.name, **extra}


def _clean_title(title: str | None) -> str | None:
    if not title:
        return None
    title = re.sub(r"\s+", " ", title).strip(" .。:：、")
    return title[:120] or None


def iter_clause_types() -> Iterable[str]:
    return (t.value for t in ClauseType)


__all__ = [
    "KEYWORD_RULES",
    "ClauseExtraction",
    "ClauseExtractorAgent",
    "ExtractedClause",
    "HEADING_RE",
    "iter_clause_types",
]
