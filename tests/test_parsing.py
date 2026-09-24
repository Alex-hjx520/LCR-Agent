"""文档解析与切分测试。"""

from __future__ import annotations

import pytest

from parsing import (
    UnsupportedDocumentError,
    chunk_document,
    count_tokens,
    get_parser,
    is_heading,
    make_doc_id,
    parse_bytes,
    parse_document,
    parse_text,
)
from parsing.text_parser import TextParser
from schemas.document import SourceType


class TestParsingBasics:
    def test_is_heading_recognizes_chinese_clause(self):
        assert is_heading("第一条 服务范围")
        assert is_heading("第 12 条 违约责任")
        assert is_heading("ARTICLE 5 TERMINATION")
        assert is_heading("3.2 Payment Terms")

    def test_is_heading_rejects_body_text(self):
        assert not is_heading("乙方应在收到发票后三十日内支付全部价款，逾期视为违约。")
        assert not is_heading("")

    def test_make_doc_id_is_stable(self):
        assert make_doc_id("a/b/contract.pdf") == make_doc_id("a/b/contract.pdf")
        assert make_doc_id("a/contract.pdf") != make_doc_id("b/contract.pdf")

    def test_get_parser_dispatch(self):
        assert isinstance(get_parser("x.PDF"), type(get_parser("y.pdf")))
        assert get_parser("x.txt").source_type is SourceType.TXT
        assert get_parser("x.md").source_type is SourceType.MARKDOWN
        assert get_parser("x.docx").source_type is SourceType.DOCX

    def test_get_parser_raises_on_unknown_suffix(self):
        with pytest.raises(UnsupportedDocumentError, match="不支持的文件类型"):
            get_parser("weird.xyz")

    def test_count_tokens_positive(self):
        assert count_tokens("") <= 1
        assert count_tokens("合同价款为人民币一百万元。") > 0


class TestParseText:
    def test_parse_text_produces_blocks(self, sample_contract):
        doc = parse_text(sample_contract, filename="c.txt")
        assert doc.source_type is SourceType.TXT
        assert doc.num_blocks >= 9
        assert doc.raw_text.startswith("技术服务合同")
        assert all(b.block_id.startswith(doc.doc_id) for b in doc.blocks)

    def test_parse_text_marks_headings(self, sample_contract):
        doc = parse_text(sample_contract)
        headings = [b.text for b in doc.blocks if b.is_heading]
        assert any("第一条" in h for h in headings)
        assert any("保密" in h for h in headings)

    def test_parse_text_skips_blank_paragraphs(self):
        doc = parse_text("第一段。\n\n\n\n第二段。", filename="x.txt")
        assert doc.num_blocks == 2

    def test_parse_document_from_file(self, tmp_contract_txt):
        doc = parse_document(tmp_contract_txt)
        assert doc.filename == "contract.txt"
        assert doc.char_count > 100

    def test_parse_document_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_document(tmp_path / "nope.txt")

    def test_parse_bytes_roundtrip(self, sample_contract):
        doc = parse_bytes(sample_contract.encode("utf-8"), "inline.txt")
        assert doc.num_blocks >= 9
        assert "技术服务合同" in doc.raw_text


class TestChunking:
    def test_chunk_document_covers_content(self, sample_contract):
        doc = parse_text(sample_contract, filename="contract.txt")
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)

        assert chunks, "应当产生至少一个 chunk"
        assert all(c.doc_id == doc.doc_id for c in chunks)
        merged = "".join(c.text for c in chunks)
        for keyword in ("服务范围", "付款", "适用法律"):
            assert keyword in merged

    def test_chunk_ids_are_unique_and_ordered(self, sample_contract):
        doc = parse_text(sample_contract)
        chunks = chunk_document(doc, chunk_size=120, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)

    def test_chunk_respects_token_budget(self):
        long_text = "第一条 定义\n" + "双方就本合同的解释达成一致。" * 200
        doc = parse_text(long_text)
        chunks = chunk_document(doc, chunk_size=150, chunk_overlap=20)
        assert len(chunks) > 1
        # 允许句子边界带来的少量超出（重叠前缀），但不应翻倍
        assert max(c.token_count for c in chunks) < 150 * 2

    def test_empty_document_yields_no_chunks(self):
        doc = parse_text("   ", filename="empty.txt")
        assert chunk_document(doc) == []

    def test_section_is_tracked(self, sample_contract):
        doc = parse_text(sample_contract)
        chunks = chunk_document(doc, chunk_size=1000)
        sections = {c.section for c in chunks if c.section}
        assert any("第一条" in s for s in sections)


class TestParserContract:
    def test_text_parser_suffixes_declared(self):
        assert TextParser.suffixes == (".txt", ".md", ".markdown")

    def test_parser_registry_covers_expected_types(self):
        assert get_parser("a.pdf").source_type is SourceType.PDF
        assert get_parser("a.docx").source_type is SourceType.DOCX
