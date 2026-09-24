"""文本切分（chunking）。

目标：产出「语义完整 + 带定位信息」的检索单元。合同场景下，切分粒度错位
（例如把一条条款劈成两半）会直接导致条款抽取与风险判定失败，因此这里采用
「先按条款边界聚合，再按 token 上限兜底切分」的两级策略。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from config import get_settings
from parsing.base import HEADING_RE
from schemas.document import Chunk, ParsedDocument, TextBlock

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_encoder(encoding: str):
    """缓存 tiktoken 编码器；tiktoken 不可用时回退为字符估算。"""
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding)
    except Exception:  # noqa: BLE001 - 网络受限时无法下载编码文件
        logger.warning("tiktoken 编码 %s 不可用，回退为字符估算（≈ len/1.6）", encoding)
        return None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """估算文本 token 数。"""
    encoder = _get_encoder(encoding)
    if encoder is None:
        return max(1, int(len(text) / 1.6))
    return len(encoder.encode(text))


def _split_sentences(text: str) -> list[str]:
    """中英文混合的句子切分（中文标点 + 英文句点）。"""
    import re

    parts = re.split(r"(?<=[。；！？!?])\s*|(?<=\.)\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p and p.strip()]


def _hard_split(text: str, max_tokens: int, encoding: str) -> list[str]:
    """超长文本按句子累积；单句仍超限时按字符硬切。"""
    sentences = _split_sentences(text) or [text]
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent, encoding)
        if sent_tokens > max_tokens:
            if buf:
                out.append("".join(buf))
                buf, buf_tokens = [], 0
            # 单句超长：按字符硬切
            step = max(1, int(max_tokens * 1.6))
            out.extend(sent[i : i + step] for i in range(0, len(sent), step))
            continue
        if buf_tokens + sent_tokens > max_tokens and buf:
            out.append("".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += sent_tokens

    if buf:
        out.append("".join(buf))
    return out


def _tail_overlap(text: str, overlap_tokens: int, encoding: str) -> str:
    """取文本尾部约 `overlap_tokens` 个 token 作为重叠前缀。"""
    if overlap_tokens <= 0:
        return ""
    sentences = _split_sentences(text)
    acc: list[str] = []
    total = 0
    for sent in reversed(sentences):
        t = count_tokens(sent, encoding)
        if total + t > overlap_tokens and acc:
            break
        acc.insert(0, sent)
        total += t
    return "".join(acc)


def _group_by_section(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """按标题把块聚合成 section；无标题时整体视作一个 section。"""
    sections: list[list[TextBlock]] = []
    current: list[TextBlock] = []

    for block in blocks:
        if block.is_heading and current:
            sections.append(current)
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append(current)
    return sections


def chunk_document(
    document: ParsedDocument,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    encoding: str | None = None,
) -> list[Chunk]:
    """把 `ParsedDocument` 切分为 `Chunk` 列表。

    Args:
        document: 解析后的文档。
        chunk_size: 每个 chunk 的 token 上限，默认取配置 `LCR_CHUNK_SIZE`。
        chunk_overlap: 相邻 chunk 的重叠 token 数，默认取配置 `LCR_CHUNK_OVERLAP`。
        encoding: tiktoken 编码名。

    Returns:
        按文档顺序排列的 chunk 列表。
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    encoding = encoding or settings.chunk_encoding

    blocks = sorted(document.blocks, key=lambda b: b.order)
    if not blocks:
        return []

    chunks: list[Chunk] = []
    for section in _group_by_section(blocks):
        heading = section[0].text if section[0].is_heading else None
        merged = "\n".join(b.text for b in section)
        if count_tokens(merged, encoding) <= chunk_size:
            chunks.append(
                _make_chunk(document, section, merged, heading, encoding, len(chunks))
            )
            continue

        # 超长 section：按 token 上限拆分，块区间用近似比例映射回 block 范围
        pieces = _hard_split(merged, chunk_size, encoding)
        offset = 0
        for i, piece in enumerate(pieces):
            sub = _blocks_for_span(section, offset, len(piece))
            body = piece if i == 0 else _tail_overlap(pieces[i - 1], chunk_overlap, encoding) + piece
            chunks.append(
                _make_chunk(document, sub, body, heading, encoding, len(chunks))
            )
            offset += len(piece)

    logger.info(
        "文档 %s 切分完成：%d blocks -> %d chunks (size=%d, overlap=%d)",
        document.doc_id,
        len(blocks),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _blocks_for_span(blocks: list[TextBlock], start: int, length: int) -> list[TextBlock]:
    """把字符区间映射回 block 子集（近似，用于保留页码定位）。"""
    sub: list[TextBlock] = []
    pos = 0
    for block in blocks:
        end = pos + len(block.text) + 1
        if end > start and pos < start + length:
            sub.append(block)
        pos = end
    return sub or blocks


def _make_chunk(
    document: ParsedDocument,
    blocks: list[TextBlock],
    text: str,
    heading: str | None,
    encoding: str,
    index: int,
) -> Chunk:
    pages = [b.page for b in blocks if b.page is not None]
    first = blocks[0] if blocks else None
    last = blocks[-1] if blocks else None
    return Chunk(
        chunk_id=f"{document.doc_id}-c{index:04d}",
        doc_id=document.doc_id,
        text=text,
        start_block=first.order if first else 0,
        end_block=last.order if last else 0,
        page_from=min(pages) if pages else None,
        page_to=max(pages) if pages else None,
        token_count=count_tokens(text, encoding),
        section=heading,
        clause_hint=heading if heading and HEADING_RE.search(heading) else None,
        metadata={"source_type": document.source_type.value, "filename": document.filename},
    )


__all__ = ["chunk_document", "count_tokens"]
