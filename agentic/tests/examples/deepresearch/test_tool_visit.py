"""Tests for Visit tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from examples.deepresearch.tools.tool_visit import Visit


@pytest.fixture
def visit_tool():
    return Visit()


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        from examples.deepresearch.tools.tool_visit import truncate_to_tokens
        with patch("examples.deepresearch.tools.tool_visit.tiktoken") as mock_tiktoken:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_encoding.decode.return_value = "Hello world"
            mock_tiktoken.get_encoding.return_value = mock_encoding

            result = truncate_to_tokens("Hello world", max_tokens=100)
            assert result == "Hello world"

    def test_empty_string(self):
        from examples.deepresearch.tools.tool_visit import truncate_to_tokens
        with patch("examples.deepresearch.tools.tool_visit.tiktoken") as mock_tiktoken:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = []
            mock_encoding.decode.return_value = ""
            mock_tiktoken.get_encoding.return_value = mock_encoding

            result = truncate_to_tokens("", max_tokens=100)
            assert result == ""

    def test_long_text_truncated(self):
        from examples.deepresearch.tools.tool_visit import truncate_to_tokens
        with patch("examples.deepresearch.tools.tool_visit.tiktoken") as mock_tiktoken:
            mock_encoding = MagicMock()
            # Simulate 200 tokens (exceeds max_tokens=100)
            mock_encoding.encode.return_value = list(range(200))
            mock_encoding.decode.return_value = "truncated text"
            mock_tiktoken.get_encoding.return_value = mock_encoding

            result = truncate_to_tokens("very long text", max_tokens=100)
            # Should have called decode with only first 100 tokens
            mock_encoding.decode.assert_called_once_with(list(range(100)))


class TestVisitCall:
    def test_single_url(self, visit_tool):
        with patch.object(visit_tool, "readpage_jina", return_value="page content"):
            result = visit_tool.call({"url": "https://example.com", "goal": "find info"})
        assert "page content" in result

    def test_multiple_urls(self, visit_tool):
        with patch.object(visit_tool, "readpage_jina", return_value="content"):
            result = visit_tool.call({"url": ["https://a.com", "https://b.com"], "goal": "find info"})
        assert "content" in result

    def test_invalid_format(self, visit_tool):
        result = visit_tool.call({})
        assert "Invalid request" in result


class TestJinaReadpage:
    @patch("examples.deepresearch.tools.tool_visit.requests.get")
    def test_success(self, mock_get, visit_tool):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Webpage content here"
        mock_get.return_value = mock_response

        result = visit_tool.jina_readpage("https://example.com")
        assert result == "Webpage content here"

    @patch("examples.deepresearch.tools.tool_visit.requests.get")
    def test_failure_returns_error(self, mock_get, visit_tool):
        mock_get.side_effect = Exception("connection error")

        result = visit_tool.jina_readpage("https://example.com")
        assert "Failed" in result


class TestHtmlReadpageJina:
    def test_returns_content_on_success(self, visit_tool):
        with patch.object(visit_tool, "jina_readpage", return_value="Good content"):
            result = visit_tool.html_readpage_jina("https://example.com")
        assert result == "Good content"

    def test_returns_error_on_all_failures(self, visit_tool):
        with patch.object(visit_tool, "jina_readpage", return_value="[visit] Failed to read page."):
            result = visit_tool.html_readpage_jina("https://example.com")
        assert "Failed" in result


class TestCallServer:
    @patch("examples.deepresearch.tools.tool_visit.OpenAI")
    def test_success(self, mock_openai_cls, visit_tool):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"evidence": "test", "summary": "test summary"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = visit_tool.call_server([{"role": "user", "content": "test"}])
        assert "evidence" in result

    @patch("examples.deepresearch.tools.tool_visit.OpenAI")
    def test_json_extraction_from_markdown(self, mock_openai_cls, visit_tool):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_client.chat.completions.create.return_value = mock_response

        result = visit_tool.call_server([{"role": "user", "content": "test"}])
        assert '"key"' in result

    @patch("examples.deepresearch.tools.tool_visit.OpenAI")
    def test_failure_returns_empty(self, mock_openai_cls, visit_tool):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = visit_tool.call_server([{"role": "user", "content": "test"}])
        assert result == ""
