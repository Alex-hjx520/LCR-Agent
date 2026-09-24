"""PDF 解析器：基于 pdfplumber，逐页提取文本与表格。"""

from __future__ import annotations

import logging
from pathlib import Path

from parsing.base import BaseParser, RawBlock
from schemas.document import SourceType

logger = logging.getLogger(__name__)


class PdfParser(BaseParser):
    """从 PDF 中提取段落级文本块。

    策略：
    - 优先使用逐行 `extract_text` 结果按版面 y 坐标排序，尽量保持阅读顺序；
    - 表格单独提取并转换为 Markdown 表格，附加到该页的文本流末尾；
    - 需要扫描件（图片型 PDF）时，请在外层接入 OCR，本解析器不做 OCR。
    """

    source_type = SourceType.PDF
    suffixes = (".pdf",)

    def __init__(self, *, extract_tables: bool = True, table_settings: dict | None = None) -> None:
        self.extract_tables = extract_tables
        self.table_settings = table_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
        }

    def _iter_blocks(self, path: Path) -> list[RawBlock]:
        import pdfplumber  # 延迟导入，缩短应用冷启动时间

        blocks: list[RawBlock] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                # ---- 正文：按 top 坐标排序，恢复阅读顺序 ----
                words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
                lines = _group_words_into_lines(words)
                for text, top in lines:
                    blocks.append((text, page_no, None, "body"))

                # ---- 表格 ----
                if self.extract_tables:
                    for tbl in page.extract_tables(self.table_settings) or []:
                        md = _table_to_markdown(tbl)
                        if md:
                            blocks.append((md, page_no, None, "table"))

                logger.debug("PDF 第 %d 页解析完成，累计 %d 块", page_no, len(blocks))
        return blocks


def _group_words_into_lines(words: list[dict], *, y_tol: float = 3.0) -> list[tuple[str, float]]:
    """把 pdfplumber 的 word 列表按 y 坐标聚合成行。"""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["top"] / y_tol), w["x0"]))
    lines: list[tuple[str, float]] = []
    bucket: list[dict] = []
    current_top = ordered[0]["top"]

    def flush() -> None:
        if bucket:
            text = " ".join(w["text"] for w in bucket)
            lines.append((text, current_top))

    for word in ordered:
        if abs(word["top"] - current_top) > y_tol:
            flush()
            bucket = []
            current_top = word["top"]
        bucket.append(word)
    flush()
    return lines


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """把二维表格转成 Markdown，便于 LLM 理解结构化数据。"""
    rows = [[(c or "").strip().replace("\n", " ") for c in row] for row in table if row]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


__all__ = ["PdfParser"]
