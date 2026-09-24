"""纯文本 / Markdown 解析器：按空行切段，识别 Markdown 标题。"""

from __future__ import annotations

import re
from pathlib import Path

from parsing.base import BaseParser, RawBlock
from schemas.document import SourceType

_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


class TextParser(BaseParser):
    """处理 `.txt` / `.md`。

    - 以空行作为段落分隔；
    - Markdown 标题行单独成块并标记为 heading 样式；
    - 不解析内联 Markdown 语法（保留原文，避免丢失合同要件措辞）。
    """

    source_type = SourceType.TXT
    suffixes = (".txt", ".md", ".markdown")

    def _iter_blocks(self, path: Path) -> list[RawBlock]:
        raw = _read_text(path)
        blocks: list[RawBlock] = []

        for para in re.split(r"\n\s*\n", raw):
            para = para.strip()
            if not para:
                continue
            match = _MD_HEADING_RE.match(para)
            if match:
                blocks.append((match.group(2).strip(), None, None, "heading"))
            else:
                blocks.append((para, None, None, "body"))
        return blocks


def _read_text(path: Path) -> str:
    """按常见编码依次尝试读取，兼容国内合同文档的 GBK 编码。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


class MarkdownParser(TextParser):
    source_type = SourceType.MARKDOWN
    suffixes = (".md", ".markdown")


__all__ = ["MarkdownParser", "TextParser"]
