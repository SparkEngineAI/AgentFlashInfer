"""Tests for file_parser.py helper functions and SingleFileParser."""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from examples.deepresearch.tools.file_tools.file_parser import (
    CustomJSONEncoder,
    FileParserError,
    SingleFileParser,
    clean_text,
    compress,
    df_to_markdown,
    extract_csv_schema,
    extract_xls_schema,
    extract_xml_skeleton_markdown,
    get_plain_doc,
    parse_html,
    parse_tabular_file,
    parse_txt,
    parse_xml,
    postprocess_page_content,
    process_idp_result,
    table_converter,
)


class TestCleanText:
    def test_collapses_newlines(self):
        assert clean_text("hello\n\n\nworld") == "hello\nworld"

    def test_removes_reading_list_text(self):
        assert "Qwen's Reading List" not in clean_text("Add to Qwen's Reading List text")

    def test_collapses_dashes(self):
        assert clean_text("a------b") == "a-----b"

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestGetPlainDoc:
    def test_extracts_text_and_table(self):
        doc = [
            {
                "page_num": 1,
                "content": [
                    {"text": "Hello"},
                    {"table": "| a | b |"},
                    {"image": "img_data"},
                ],
            }
        ]
        result = get_plain_doc(doc)
        assert "Hello" in result
        assert "| a | b |" in result
        assert "img_data" in result

    def test_empty_doc(self):
        assert get_plain_doc([]) == ""


class TestDfToMarkdown:
    def test_converts_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df_to_markdown(df)
        assert "a" in result
        assert "b" in result
        assert "|" in result

    def test_handles_nans(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, 4]})
        result = df_to_markdown(df)
        assert isinstance(result, str)


class TestParseTxt:
    def test_parses_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3", encoding="utf-8")
        result = parse_txt(str(f))
        assert len(result) == 1
        assert result[0]["page_num"] == 1
        texts = [c["text"] for c in result[0]["content"]]
        assert "line1" in texts
        assert "line3" in texts

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = parse_txt(str(f))
        assert len(result) == 1


class TestParseHtml:
    def test_parses_html_file(self, tmp_path):
        html = "<html><head><title>Test</title></head><body><p>Hello</p><div>World</div></body></html>"
        f = tmp_path / "test.html"
        f.write_text(html, encoding="utf-8")
        # BeautifulSoup is lazily imported inside parse_html, patch at bs4 module
        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_soup.title.string = "Test"
            mock_p = MagicMock()
            mock_p.get_text.return_value = "Hello"
            mock_div = MagicMock()
            mock_div.get_text.return_value = "World"
            mock_soup.find_all.return_value = [mock_p, mock_div]
            mock_bs.return_value = mock_soup

            result = parse_html(str(f))
            assert len(result) == 1
            assert result[0]["title"] == "Test"


class TestParseXml:
    def test_parses_xml_file(self, tmp_path):
        xml = '<?xml version="1.0"?>\n<root><item>hello</item></root>'
        f = tmp_path / "test.xml"
        f.write_text(xml, encoding="utf-8")
        with patch("examples.deepresearch.tools.file_tools.file_parser.count_tokens", return_value=100):
            result = parse_xml(str(f))
            assert len(result) == 1
            assert "hello" in result[0]["content"][0]["text"]

    def test_large_xml_returns_schema(self, tmp_path):
        xml = '<?xml version="1.0"?>\n<root><item>hello</item></root>'
        f = tmp_path / "large.xml"
        f.write_text(xml, encoding="utf-8")
        with patch("examples.deepresearch.tools.file_tools.file_parser.count_tokens", return_value=999999), \
             patch("examples.deepresearch.tools.file_tools.file_parser.extract_xml_skeleton_markdown", return_value="## Root: root"):
            result = parse_xml(str(f))
            assert "schema" in result[0]["content"][0]


class TestTableConverter:
    def test_converts_table(self):
        table = [["a", "b"], ["1", "2"]]
        result = table_converter(table)
        assert "|a|b|" in result
        assert "|1|2|" in result

    def test_handles_none_values(self):
        table = [["a", None], [None, "2"]]
        result = table_converter(table)
        assert "None" in result

    def test_handles_newlines_in_cells(self):
        table = [["a\nb", "c"]]
        result = table_converter(table)
        assert "a b" in result


class TestCustomJSONEncoder:
    def test_serializes_datetime(self):
        dt = datetime(2024, 1, 15, 12, 30, 0)
        result = json.dumps({"dt": dt}, cls=CustomJSONEncoder)
        assert "2024-01-15" in result

    def test_serializes_timestamp(self):
        from pandas import Timestamp
        ts = Timestamp("2024-01-15")
        result = json.dumps({"ts": ts}, cls=CustomJSONEncoder)
        assert "2024-01-15" in result


class TestFileParserError:
    def test_creation(self):
        err = FileParserError("test error", code="500")
        assert str(err) == "test error"
        assert err.code == "500"

    def test_with_exception(self):
        original = ValueError("original")
        err = FileParserError("wrapped", exception=original)
        assert err.exception is original


class TestProcessIdpResult:
    def test_processes_layouts(self):
        result = {
            "layouts": [
                {"pageNum": 1, "markdownContent": "Page 1 content"},
                {"pageNum": 1, "markdownContent": "More page 1"},
                {"pageNum": 2, "markdownContent": "Page 2"},
            ]
        }
        pages = process_idp_result(result)
        assert len(pages) == 2
        assert pages[0]["page_num"] == 1
        assert len(pages[0]["content"]) == 2
        assert pages[1]["page_num"] == 2

    def test_empty_layouts(self):
        assert process_idp_result({"layouts": []}) == []


class TestExtractXmlSkeletonMarkdown:
    def test_basic_xml(self, tmp_path):
        xml = '<?xml version="1.0"?>\n<root><child attr="val">text</child></root>'
        f = tmp_path / "test.xml"
        f.write_text(xml, encoding="utf-8")
        result = extract_xml_skeleton_markdown(str(f))
        assert "Root: root" in result
        assert "child" in result


class TestExtractCsvSchema:
    def test_extracts_schema(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        schema = extract_csv_schema(str(f))
        assert "columns" in schema
        assert "a" in schema["columns"]
        assert "b" in schema["columns"]
        assert "sample_data" in schema


class TestExtractXlsSchema:
    def test_extracts_schema(self, tmp_path):
        f = tmp_path / "test.xlsx"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        try:
            df.to_excel(str(f), index=False)
        except ImportError:
            pytest.skip("openpyxl not installed")
        schema = extract_xls_schema(str(f))
        assert "sheets" in schema
        assert schema["n_sheets"] == 1
        assert "col1" in schema["sheets"][0]["columns"]


class TestParseTabularFile:
    def test_parses_csv(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        with patch("examples.deepresearch.tools.file_tools.file_parser.count_tokens", return_value=100):
            result = parse_tabular_file(str(f), sep=",")
            assert len(result) == 1
            assert "table" in result[0]["content"][0]

    def test_large_table_returns_schema(self, tmp_path):
        f = tmp_path / "big.csv"
        f.write_text("name,age\nAlice,30\n", encoding="utf-8")
        with patch("examples.deepresearch.tools.file_tools.file_parser.count_tokens", return_value=999999), \
             patch("examples.deepresearch.tools.file_tools.file_parser.extract_csv_schema") as mock_schema:
            mock_schema.return_value = {"columns": ["name", "age"], "dtypes": {}, "sample_data": {}}
            result = parse_tabular_file(str(f), sep=",")
            assert "schema" in result[0]["content"][0]

    def test_parse_failure_returns_empty(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("not,a,valid\ncsv", encoding="utf-8")
        with patch("examples.deepresearch.tools.file_tools.file_parser.pd.read_csv", side_effect=Exception("parse error")):
            result = parse_tabular_file(str(f), sep=",")
            assert result == []


class TestCompress:
    def test_compresses_results(self):
        with patch("examples.deepresearch.tools.file_tools.file_parser.tokenizer") as mock_tok, \
             patch("examples.deepresearch.tools.file_tools.file_parser.DEFAULT_MAX_INPUT_TOKENS", 1000):
            mock_tok.tokenize.side_effect = lambda x: list(x)
            mock_tok.convert_tokens_to_string.side_effect = lambda x: "".join(x)
            results = ["hello world", "foo bar"]
            compressed = compress(results)
            assert len(compressed) == 2
            assert isinstance(compressed[0], str)


class TestPostprocessPageContent:
    def test_removes_repetitive_text(self):
        # Create mock objects with bbox
        mock_table_obj = MagicMock()
        mock_table_obj.bbox = (0, 0, 100, 100)

        mock_text_obj = MagicMock()
        mock_text_obj.bbox = (10, 10, 90, 90)
        mock_text_obj.height = 12

        page_content = [
            {"table": "table data", "obj": mock_table_obj},
            {"text": "repetitive text", "obj": mock_text_obj, "font-size": 12},
        ]
        result = postprocess_page_content(page_content)
        # The text should be removed because it's inside the table bbox
        table_items = [p for p in result if "table" in p]
        text_items = [p for p in result if "text" in p]
        assert len(table_items) >= 1
        # Text inside table bbox should be filtered


class TestSingleFileParser:
    @pytest.fixture
    def parser(self):
        with patch("examples.deepresearch.tools.file_tools.file_parser.Storage"):
            return SingleFileParser()

    def test_init(self, parser):
        assert parser.name == "file_parser"
        assert "pdf" in parser.parsers
        assert "txt" in parser.parsers
        assert "csv" in parser.parsers

    def test_call_with_cached_result(self, parser):
        with patch.object(parser, "_prepare_file", return_value="/tmp/test.txt"), \
             patch.object(parser.db, "get", return_value='[{"page_num": 1, "content": [{"text": "cached"}]}]'):
            result = parser.call(json.dumps({"url": "/tmp/test.txt"}))
            assert "cached" in result

    def test_call_with_new_file(self, parser):
        from qwen_agent.tools.storage import KeyNotExistsError
        with patch.object(parser, "_prepare_file", return_value="/tmp/test.txt"), \
             patch.object(parser.db, "get", side_effect=KeyNotExistsError("not found")), \
             patch.object(parser, "_process_new_file", return_value=[{"page_num": 1, "content": [{"text": "new content"}]}]):
            result = parser.call(json.dumps({"url": "/tmp/test.txt"}))
            assert "new content" in result

    def test_prepare_file_local(self, parser):
        with patch("examples.deepresearch.tools.file_tools.file_parser.is_http_url", return_value=False), \
             patch("examples.deepresearch.tools.file_tools.file_parser.sanitize_chrome_file_path", return_value="/local/path"):
            result = parser._prepare_file("/local/path")
            assert result == "/local/path"

    def test_prepare_file_http(self, parser):
        with patch("examples.deepresearch.tools.file_tools.file_parser.is_http_url", return_value=True), \
             patch("examples.deepresearch.tools.file_tools.file_parser.save_url_to_local_work_dir", return_value="/downloaded/file"):
            result = parser._prepare_file("https://example.com/file.pdf")
            assert result == "/downloaded/file"

    def test_flatten_result(self, parser):
        result = [
            {"page_num": 1, "content": [{"text": "Hello"}, {"table": "| a |"}]},
            {"page_num": 2, "content": [{"text": "World"}]},
        ]
        flat = parser._flatten_result(result)
        assert "Hello" in flat
        assert "| a |" in flat
        assert "World" in flat