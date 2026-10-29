"""Tests for file_tools/utils.py."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from examples.deepresearch.tools.file_tools.utils import (
    contains_html_tags,
    extract_code,
    extract_markdown_urls,
    extract_urls,
    get_basename_from_url,
    get_content_type_by_head_request,
    get_file_type,
    get_local_ip,
    has_chinese_chars,
    has_chinese_messages,
    hash_sha256,
    is_http_url,
    is_image,
    json_dumps_compact,
    json_dumps_pretty,
    json_loads,
    merge_generate_cfgs,
    save_text_to_file,
    save_url_to_local_work_dir,
    sanitize_chrome_file_path,
    sanitize_windows_file_path,
)


class TestHashSha256:
    def test_returns_hex_string(self):
        result = hash_sha256("test")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self):
        assert hash_sha256("hello") == hash_sha256("hello")

    def test_different_inputs(self):
        assert hash_sha256("a") != hash_sha256("b")


class TestGetBasenameFromUrl:
    def test_local_path(self):
        assert get_basename_from_url("/mnt/a/b/file.txt") == "file.txt"

    def test_http_url(self):
        assert get_basename_from_url("https://example.com/path/file.pdf") == "file.pdf"

    def test_url_with_query(self):
        result = get_basename_from_url("https://example.com/file?q=1")
        assert "file" in result

    def test_url_with_trailing_slash(self):
        result = get_basename_from_url("https://example.com/")
        assert result == "example.com"

    def test_windows_path(self):
        result = get_basename_from_url("C:\\a\\b\\file.txt")
        assert "file.txt" in result

    def test_url_encoded(self):
        result = get_basename_from_url("https://example.com/my%20file.txt")
        assert "my file.txt" == result

    def test_remove_uuid(self):
        name = "a1b2c3d4-e5f6-7890-abcd-ef1234567890_report.pdf"
        result = get_basename_from_url(name, need_rm_uuid=True)
        assert result == "report.pdf"


class TestIsHttpUrl:
    def test_https(self):
        assert is_http_url("https://example.com") is True

    def test_http(self):
        assert is_http_url("http://example.com") is True

    def test_local_path(self):
        assert is_http_url("/tmp/file.txt") is False

    def test_empty_string(self):
        assert is_http_url("") is False


class TestIsImage:
    def test_jpg(self):
        assert is_image("photo.jpg") is True

    def test_png(self):
        assert is_image("image.png") is True

    def test_non_image(self):
        assert is_image("doc.pdf") is False

    def test_image_url(self):
        assert is_image("https://example.com/photo.jpeg") is True


class TestSanitizeChromeFilePath:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = sanitize_chrome_file_path(str(f))
        assert result == str(f)

    def test_file_uri(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = sanitize_chrome_file_path(f"file://{f}")
        # Should resolve to existing file or return sanitized path
        assert isinstance(result, str)


class TestSanitizeWindowsFilePath:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = sanitize_windows_file_path(str(f))
        assert result == str(f)

    def test_nonexistent_returns_original(self):
        result = sanitize_windows_file_path("/nonexistent/path/file.txt")
        assert result == "/nonexistent/path/file.txt"


class TestSaveUrlToLocalWorkDir:
    def test_copies_local_file(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("content")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        result = save_url_to_local_work_dir(str(src), str(dest_dir))
        assert os.path.exists(result)
        with open(result) as f:
            assert f.read() == "content"

    def test_downloads_http_file(self, tmp_path):
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        with patch("examples.deepresearch.tools.file_tools.utils.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"downloaded content"
            mock_get.return_value = mock_response
            result = save_url_to_local_work_dir("https://example.com/file.txt", str(dest_dir))
            assert os.path.exists(result)

    def test_download_failure(self, tmp_path):
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        with patch("examples.deepresearch.tools.file_tools.utils.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response
            with pytest.raises(ValueError, match="Can not download"):
                save_url_to_local_work_dir("https://example.com/missing.txt", str(dest_dir))

    def test_custom_filename(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("content")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        result = save_url_to_local_work_dir(str(src), str(dest_dir), save_filename="custom.txt")
        assert result.endswith("custom.txt")


class TestGetFileType:
    def test_pdf(self):
        assert get_file_type("/path/to/file.pdf") == "pdf"

    def test_docx(self):
        assert get_file_type("/path/to/file.docx") == "docx"

    def test_csv(self):
        assert get_file_type("/path/to/file.csv") == "csv"

    def test_image_returns_image(self):
        assert get_file_type("/path/to/photo.jpg") == "image"

    def test_html_by_extension(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body>Hello</body></html>")
        assert get_file_type(str(f)) == "html"

    def test_txt_fallback(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("Just some text without html tags")
        assert get_file_type(str(f)) == "txt"

    def test_http_url_defaults_html(self):
        with patch("examples.deepresearch.tools.file_tools.utils.get_content_type_by_head_request", return_value="text/html"):
            assert get_file_type("https://example.com/page") == "html"

    def test_http_url_pdf_content_type(self):
        with patch("examples.deepresearch.tools.file_tools.utils.get_content_type_by_head_request", return_value="application/pdf"):
            assert get_file_type("https://example.com/doc") == "pdf"


class TestExtractUrls:
    def test_extracts_http_urls(self):
        text = "Visit https://example.com and http://test.com for info"
        urls = extract_urls(text)
        assert "https://example.com" in urls
        assert "http://test.com" in urls

    def test_no_urls(self):
        assert extract_urls("no urls here") == []


class TestExtractMarkdownUrls:
    def test_extracts_markdown_links(self):
        text = "[click here](https://example.com) and ![img](https://img.com/pic.png)"
        urls = extract_markdown_urls(text)
        assert "https://example.com" in urls
        assert "https://img.com/pic.png" in urls

    def test_no_markdown_urls(self):
        assert extract_markdown_urls("plain text") == []


class TestExtractCode:
    def test_triple_backtick_block(self):
        text = '```python\nprint("hello")\n```'
        result = extract_code(text)
        assert 'print("hello")' in result

    def test_json_code_field(self):
        text = '{"code": "x = 1"}'
        result = extract_code(text)
        assert "x = 1" in result

    def test_plain_text(self):
        text = "x = 1"
        result = extract_code(text)
        assert result == "x = 1"


class TestJsonLoads:
    def test_valid_json(self):
        result = json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json5_syntax(self):
        result = json_loads('{key: "value"}')
        assert result == {"key": "value"}

    def test_markdown_wrapped(self):
        result = json_loads('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_raises(self):
        with pytest.raises(json.decoder.JSONDecodeError):
            json_loads("{invalid}")


class TestJsonDumps:
    def test_pretty_format(self):
        result = json_dumps_pretty({"a": 1})
        assert "\n" in result
        assert '"a"' in result

    def test_compact_format(self):
        result = json_dumps_compact({"a": 1})
        assert "\n" not in result


class TestHasChineseChars:
    def test_chinese(self):
        assert has_chinese_chars("你好") is True

    def test_english(self):
        assert has_chinese_chars("hello") is False

    def test_mixed(self):
        assert has_chinese_chars("hello你好") is True

    def test_empty(self):
        assert has_chinese_chars("") is False


class TestHasChineseMessages:
    def test_chinese_user_message(self):
        from qwen_agent.llm.schema import Message
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="你好"),
        ]
        assert has_chinese_messages(messages) is True

    def test_no_chinese(self):
        from qwen_agent.llm.schema import Message
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]
        assert has_chinese_messages(messages) is False


class TestContainsHtmlTags:
    def test_html_content(self):
        assert contains_html_tags("<p>Hello</p>") is True

    def test_no_html(self):
        assert contains_html_tags("plain text") is False


class TestGetContentTypeByHeadRequest:
    def test_success(self):
        with patch("examples.deepresearch.tools.file_tools.utils.requests.head") as mock_head:
            mock_response = MagicMock()
            mock_response.headers = {"Content-Type": "application/pdf"}
            mock_head.return_value = mock_response
            result = get_content_type_by_head_request("https://example.com/file")
            assert result == "application/pdf"

    def test_failure(self):
        with patch("examples.deepresearch.tools.file_tools.utils.requests.head", side_effect=requests.RequestException("error")):
            result = get_content_type_by_head_request("https://example.com/file")
            assert result == "unk"


class TestGetLocalIp:
    def test_returns_string(self):
        ip = get_local_ip()
        assert isinstance(ip, str)
        assert len(ip) > 0


class TestSaveTextToFile:
    def test_saves_content(self, tmp_path):
        f = tmp_path / "output.txt"
        save_text_to_file(str(f), "hello world")
        assert f.read_text() == "hello world"


class TestMergeGenerateCfgs:
    def test_merges_configs(self):
        base = {"temperature": 0.5, "stop": ["\n"]}
        new = {"temperature": 0.8, "stop": ["<end>"]}
        result = merge_generate_cfgs(base, new)
        assert result["temperature"] == 0.8
        assert "\n" in result["stop"]
        assert "<end>" in result["stop"]

    def test_none_base(self):
        result = merge_generate_cfgs(None, {"temperature": 0.5})
        assert result["temperature"] == 0.5

    def test_none_new(self):
        result = merge_generate_cfgs({"temperature": 0.5}, None)
        assert result["temperature"] == 0.5