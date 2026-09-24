"""CUAD 数据集加载器。

CUAD（Contract Understanding Attainment Dataset）包含 510 份合同、13k+ 条
由律师标注的关键条款，是合同审查类任务最常用的公开先例语料。

原始 JSON 结构（SQuAD 风格）::

    {"data": [{"title": "合约名", "paragraphs": [
        {"context": "合同全文", "qas": [
            {"question": "... related to \"Governing Law\" ...",
             "answers": [{"text": "德克萨斯州法律", "answer_start": 1234}],
             "is_impossible": false}]}]}]}

本模块把它展平成 `PassageRecord`，并把 CUAD 的 41 个类目映射到本项目的
`ClauseType`，以便与规则抽取 / LLM 抽取结果对齐。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

from config import get_settings
from knowledge.base import PassageRecord
from schemas.clause import ClauseType
from schemas.retrieval import Corpus

logger = logging.getLogger(__name__)

_QUESTION_LABEL_RE = re.compile(r'related to\s+"([^"]+)"', re.IGNORECASE)

#: CUAD 原始类目 -> 本项目 ClauseType
CUAD_CATEGORY_MAP: dict[str, ClauseType] = {
    "document name": ClauseType.OTHER,
    "parties": ClauseType.PARTIES,
    "agreement date": ClauseType.OTHER,
    "effective date": ClauseType.OTHER,
    "expiration date": ClauseType.TERM,
    "renewal term": ClauseType.RENEWAL,
    "notice period to terminate renewal": ClauseType.RENEWAL,
    "governing law": ClauseType.GOVERNING_LAW,
    "most favored nation": ClauseType.OTHER,
    "non-compete": ClauseType.NON_COMPETE,
    "exclusivity": ClauseType.EXCLUSIVITY,
    "no-solicit of customers": ClauseType.NON_COMPETE,
    "competitive restriction exception": ClauseType.NON_COMPETE,
    "no-solicit of employees": ClauseType.NON_COMPETE,
    "non-disparagement": ClauseType.OTHER,
    "termination for convenience": ClauseType.TERMINATION,
    "rofr/rofo/rofn": ClauseType.OTHER,
    "change of control": ClauseType.CHANGE_OF_CONTROL,
    "anti-assignment": ClauseType.ASSIGNMENT,
    "revenue/profit sharing": ClauseType.PAYMENT,
    "price restrictions": ClauseType.PRICE_ADJUST,
    "minimum commitment": ClauseType.OTHER,
    "volume restriction": ClauseType.OTHER,
    "ip ownership assignment": ClauseType.IP,
    "joint ip ownership": ClauseType.IP,
    "license grant": ClauseType.IP,
    "non-transferable license": ClauseType.IP,
    "affiliate license-licensor or licensee": ClauseType.IP,
    "unlimited/all-you-can-eat-license": ClauseType.IP,
    "irrevocable or perpetual license": ClauseType.IP,
    "source code escrow": ClauseType.IP,
    "post-termination services": ClauseType.TERMINATION,
    "audit rights": ClauseType.AUDIT,
    "uncapped liability": ClauseType.LIABILITY,
    "cap on liability": ClauseType.LIABILITY,
    "liquidated damages": ClauseType.LIABILITY,
    "warranty duration": ClauseType.WARRANTY,
    "insurance": ClauseType.INSURANCE,
    "covenant not to sue": ClauseType.OTHER,
    "third party beneficiary": ClauseType.OTHER,
    "indemnification": ClauseType.INDEMNIFICATION,
    "limitation of liability": ClauseType.LIABILITY,
    "confidentiality": ClauseType.CONFIDENTIALITY,
    "data protection": ClauseType.DATA_PROTECTION,
    "force majeure": ClauseType.FORCE_MAJEURE,
    "dispute resolution": ClauseType.DISPUTE,
}


def map_category(label: str) -> ClauseType:
    """把 CUAD 类目名映射为 `ClauseType`（未知类目归为 OTHER）。"""
    return CUAD_CATEGORY_MAP.get(label.strip().lower(), ClauseType.OTHER)


def resolve_cuad_path(path: str | Path | None = None) -> Path:
    settings = get_settings()
    candidate = Path(path) if path else settings.cuad_path
    if not candidate.exists():
        msg = (
            f"未找到 CUAD 数据文件: {candidate}\n"
            "请先运行 `python scripts/download_cuad.py` 下载数据集。"
        )
        raise FileNotFoundError(msg)
    return candidate


def load_cuad_records(
    path: str | Path | None = None,
    *,
    limit_contracts: int | None = None,
    min_answer_chars: int = 20,
    context_window: int = 400,
) -> list[PassageRecord]:
    """加载 CUAD 并展平为检索记录。

    Args:
        path: CUAD JSON 路径，默认 `data/cuad/CUAD_v1.json`。
        limit_contracts: 只加载前 N 份合同（调试 / 快速建索引用）。
        min_answer_chars: 过滤过短（无信息量）的答案。
        context_window: 以 `answer_start` 为中心截取的上下文窗口长度。

    Returns:
        `PassageRecord` 列表。
    """
    return list(
        iter_cuad_records(
            path,
            limit_contracts=limit_contracts,
            min_answer_chars=min_answer_chars,
            context_window=context_window,
        )
    )


def iter_cuad_records(
    path: str | Path | None = None,
    *,
    limit_contracts: int | None = None,
    min_answer_chars: int = 20,
    context_window: int = 400,
) -> Iterator[PassageRecord]:
    """流式产出 CUAD 记录，避免一次性把全量数据读入内存。"""
    resolved = resolve_cuad_path(path)
    logger.info("加载 CUAD：%s", resolved)
    with resolved.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    contracts = payload.get("data", [])
    total = 0
    for c_index, contract in enumerate(contracts):
        if limit_contracts is not None and c_index >= limit_contracts:
            break
        title = contract.get("title") or f"contract_{c_index:04d}"
        for p_index, paragraph in enumerate(contract.get("paragraphs", [])):
            context = paragraph.get("context", "") or ""
            for q_index, qa in enumerate(paragraph.get("qas", [])):
                if qa.get("is_impossible"):
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    continue
                label = _extract_label(qa.get("question", ""))
                for a_index, answer in enumerate(answers):
                    text = (answer.get("text") or "").strip()
                    if len(text) < min_answer_chars:
                        continue
                    start = int(answer.get("answer_start") or 0)
                    snippet = _window(context, start, len(text), context_window)
                    total += 1
                    yield PassageRecord(
                        passage_id=f"cuad-{c_index:04d}-{p_index}-{q_index}-{a_index}",
                        text=snippet,
                        corpus=Corpus.CUAD,
                        label=f"{title} · {label or 'clause'}",
                        clause_type=map_category(label).value,
                        source="cuad",
                        metadata={
                            "contract": title,
                            "category": label,
                            "answer_text": text,
                            "answer_start": start,
                        },
                    )
    logger.info("CUAD 加载完成：%d 条条款记录", total)


def _extract_label(question: str) -> str:
    match = _QUESTION_LABEL_RE.search(question or "")
    return match.group(1).strip() if match else ""


def _window(context: str, start: int, length: int, window: int) -> str:
    """以答案为中心取一段上下文，保留条款语境的完整性。"""
    if not context:
        return ""
    half = max(0, (window - length) // 2)
    left = max(0, start - half)
    right = min(len(context), start + length + half)
    snippet = context[left:right].strip()
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(context) else ""
    return f"{prefix}{snippet}{suffix}"


def contract_stats(path: str | Path | None = None) -> dict:
    """返回数据集的概览统计（用于健康检查 / 数据审计）。"""
    resolved = resolve_cuad_path(path)
    with resolved.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    contracts = payload.get("data", [])
    questions = sum(len(p.get("qas", [])) for c in contracts for p in c.get("paragraphs", []))
    return {
        "file": str(resolved),
        "num_contracts": len(contracts),
        "num_qas": questions,
        "categories": len(CUAD_CATEGORY_MAP),
    }


__all__ = [
    "CUAD_CATEGORY_MAP",
    "contract_stats",
    "iter_cuad_records",
    "load_cuad_records",
    "map_category",
    "resolve_cuad_path",
]
