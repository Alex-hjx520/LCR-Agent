"""文档解析结果相关的数据模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    """支持的输入文档类型。"""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"


class TextBlock(BaseModel):
    """解析后的最小文本单元。

    保留页码 / 版面框 / 样式等信息，便于后续把风险点回溯到原文位置。
    """

    model_config = ConfigDict(frozen=True)

    block_id: str
    text: str
    order: int = 0
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    style: str | None = None
    is_heading: bool = False


class ParsedDocument(BaseModel):
    """一份文档的完整解析产物。"""

    doc_id: str
    filename: str
    source_type: SourceType
    blocks: list[TextBlock] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def raw_text(self) -> str:
        """按阅读顺序拼接的纯文本。"""
        return "\n".join(b.text for b in sorted(self.blocks, key=lambda b: b.order))

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    def slice_blocks(self, start: int, end: int) -> list[TextBlock]:
        """按 order 区间切片（左闭右开）。"""
        ordered = sorted(self.blocks, key=lambda b: b.order)
        return ordered[start:end]


class Chunk(BaseModel):
    """切分后的检索单元，是索引与引用的基本粒度。"""

    chunk_id: str
    doc_id: str
    text: str
    start_block: int = 0
    end_block: int = 0
    page_from: int | None = None
    page_to: int | None = None
    token_count: int = 0
    section: str | None = None
    clause_hint: str | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def char_span(self) -> tuple[int, int]:
        return (self.start_block, self.end_block)


__all__ = ["Chunk", "ParsedDocument", "SourceType", "TextBlock"]
