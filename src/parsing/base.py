"""解析器抽象接口与公共工具。"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from schemas.document import ParsedDocument, SourceType, TextBlock

#: 解析器内部流式输出的原始块：(text, page, bbox, style)
RawBlock = tuple[Any, ...]

#: 常见标题形态：`第X条`、`第X章`、`1.2 xxx`、`ARTICLE 5`、`Section 3` 等
HEADING_RE = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9一二三四五六七八九十百]+\s*[条章节款项]"
    r"|(?:ART(?:ICLE)?|Section|Clause|Schedule|Appendix|Exhibit)\s+[0-9IVXLC]+"
    r"|[0-9]+(?:\.[0-9]+){1,3}\s+\S"
    r")\b",
    re.IGNORECASE,
)


def is_heading(text: str) -> bool:
    """启发式判断一段文本是否为条款标题。"""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    return bool(HEADING_RE.match(stripped))


def make_doc_id(path: str | Path) -> str:
    """基于文件名 + 路径生成稳定的文档 ID（同一路径重复解析结果一致）。"""
    raw = str(Path(path).as_posix()).encode("utf-8")
    return f"doc_{hashlib.sha1(raw).hexdigest()[:12]}"


def normalize_text(text: str) -> str:
    """行内归一化：折叠空白、去掉孤立的连字符换行，保留段落结构。"""
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # 断词连字符
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BaseParser(ABC):
    """所有文档解析器的基类。

    子类只需实现 `_iter_blocks()`，把文档拆解为 (文本, 元数据) 流，
    由基类统一生成 `TextBlock` / `ParsedDocument`。
    """

    source_type: SourceType
    suffixes: tuple[str, ...] = ()

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        return Path(path).suffix.lower() in cls.suffixes

    @abstractmethod
    def _iter_blocks(self, path: Path) -> list[RawBlock]:
        """返回按阅读顺序排列的 (text, page, bbox, style) 元组列表。"""
        raise NotImplementedError

    def parse(self, path: str | Path, doc_id: str | None = None) -> ParsedDocument:
        path = Path(path)
        if not path.exists():
            msg = f"文件不存在: {path}"
            raise FileNotFoundError(msg)

        raw_blocks = self._iter_blocks(path)
        blocks: list[TextBlock] = []
        order = 0
        for item in raw_blocks:
            text = normalize_text(item[0])
            if not text:
                continue
            blocks.append(
                TextBlock(
                    block_id=f"{doc_id or make_doc_id(path)}-b{order:04d}",
                    text=text,
                    order=order,
                    page=item[1] if len(item) > 1 else None,
                    bbox=item[2] if len(item) > 2 else None,
                    style=item[3] if len(item) > 3 else None,
                    is_heading=is_heading(text),
                )
            )
            order += 1

        return ParsedDocument(
            doc_id=doc_id or make_doc_id(path),
            filename=path.name,
            source_type=self.source_type,
            blocks=blocks,
            metadata={"path": str(path), "suffix": path.suffix.lower()},
        )


__all__ = [
    "BaseParser",
    "HEADING_RE",
    "RawBlock",
    "is_heading",
    "make_doc_id",
    "normalize_text",
]
