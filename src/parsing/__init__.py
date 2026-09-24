"""文档解析包。

对外主要入口：

>>> from parsing import parse_document, chunk_document, parse_and_chunk, parse_bytes
>>> doc = parse_document("contract.pdf")
>>> chunks = chunk_document(doc)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from parsing.base import BaseParser, is_heading, make_doc_id, normalize_text
from parsing.chunker import chunk_document, count_tokens
from parsing.docx_parser import DocxParser
from parsing.pdf_parser import PdfParser
from parsing.text_parser import MarkdownParser, TextParser
from schemas.document import Chunk, ParsedDocument, SourceType, TextBlock

logger = logging.getLogger(__name__)

#: 解析器注册表（顺序即匹配优先级）
_PARSERS: tuple[type[BaseParser], ...] = (
    PdfParser,
    DocxParser,
    MarkdownParser,
    TextParser,
)


class UnsupportedDocumentError(ValueError):
    """没有匹配到可用解析器时抛出。"""


def get_parser(path: str | Path) -> BaseParser:
    """按文件后缀返回解析器实例。"""
    for parser_cls in _PARSERS:
        if parser_cls.supports(path):
            return parser_cls()
    supported = sorted({s for p in _PARSERS for s in p.suffixes})
    msg = f"不支持的文件类型 {Path(path).suffix!r}，已支持: {', '.join(supported)}"
    raise UnsupportedDocumentError(msg)


def parse_document(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    """解析本地文件为 `ParsedDocument`。"""
    path = Path(path)
    parser = get_parser(path)
    document = parser.parse(path, doc_id=doc_id)
    logger.info("已解析 %s：%d blocks / %d 字符", path.name, document.num_blocks, document.char_count)
    return document


def parse_bytes(
    content: bytes,
    filename: str,
    *,
    doc_id: str | None = None,
) -> ParsedDocument:
    """解析内存中的文件内容（用于 API 的 UploadFile）。"""
    path = Path(filename)
    parser = get_parser(path)
    doc_id = doc_id or make_doc_id(filename)

    if parser.source_type in {SourceType.TXT, SourceType.MARKDOWN}:
        text = _decode_bytes(content)
        blocks = _blocks_from_text(text, doc_id)
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            source_type=parser.source_type,
            blocks=blocks,
            metadata={"source": "bytes"},
        )

    # pdfplumber / python-docx 均支持 file-like 对象；这里落临时文件最稳妥
    with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        return parser.parse(tmp_path, doc_id=doc_id)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_text(text: str, *, filename: str = "inline.txt", doc_id: str | None = None) -> ParsedDocument:
    """把一段纯文本包装成 `ParsedDocument`。"""
    doc_id = doc_id or make_doc_id(filename)
    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        source_type=SourceType.TXT,
        blocks=_blocks_from_text(text, doc_id),
        metadata={"source": "inline"},
    )


def parse_and_chunk(path: str | Path, **chunk_kwargs) -> tuple[ParsedDocument, list[Chunk]]:
    """一步完成「解析 + 切分」。"""
    document = parse_document(path)
    return document, chunk_document(document, **chunk_kwargs)


def _decode_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return content.decode("utf-8", errors="replace")


def _blocks_from_text(text: str, doc_id: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    order = 0
    for raw in text.split("\n\n"):
        cleaned = normalize_text(raw)
        if not cleaned:
            continue
        blocks.append(
            TextBlock(
                block_id=f"{doc_id}-b{order:04d}",
                text=cleaned,
                order=order,
                style="heading" if is_heading(cleaned) else "body",
                is_heading=is_heading(cleaned),
            )
        )
        order += 1
    return blocks


__all__ = [
    "BaseParser",
    "Chunk",
    "DocxParser",
    "MarkdownParser",
    "ParsedDocument",
    "PdfParser",
    "SourceType",
    "TextBlock",
    "TextParser",
    "UnsupportedDocumentError",
    "chunk_document",
    "count_tokens",
    "get_parser",
    "is_heading",
    "make_doc_id",
    "normalize_text",
    "parse_and_chunk",
    "parse_bytes",
    "parse_document",
    "parse_text",
]
