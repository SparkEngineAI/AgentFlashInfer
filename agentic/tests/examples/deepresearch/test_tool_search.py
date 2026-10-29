"""Tests for Search tool."""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from examples.deepresearch.tools.tool_search import Search


@pytest.fixture
def search_tool():
    return Search()


@pytest.fixture
def mock_serp_response():
    return json.dumps({
        "organic": [
            {"title": "Result 1", "link": "https://a.com", "snippet": "Snippet 1", "date": "2024-01-01"},
            {"title": "Result 2", "link": "https://b.com", "snippet": "Snippet 2"},
        ]
    })


class TestSearchCall:
    def test_string_query(self, search_tool):
        with patch.object(search_tool, "google_search_with_serp", return_value="results"):
            result = search_tool.call({"query": "quantum computing"})
        assert result == "results"

    def test_list_query(self, search_tool):
        with patch.object(search_tool, "google_search_with_serp", return_value="results"):
            result = search_tool.call({"query": ["q1", "q2"]})
        assert "results" in result
        assert result.count("results") == 2

    def test_invalid_format(self, search_tool):
        result = search_tool.call({})
        assert "Invalid request" in result


class TestGoogleSearchWithSerp:
    @patch("http.client.HTTPSConnection")
    def test_chinese_query_uses_cn_params(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            search_tool.google_search_with_serp("量子计算")

        # Verify the payload includes Chinese locale
        call_args = mock_conn.request.call_args
        payload = json.loads(call_args[0][2])
        assert payload["gl"] == "cn"
        assert payload["hl"] == "zh-cn"

    @patch("http.client.HTTPSConnection")
    def test_english_query_uses_us_params(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            search_tool.google_search_with_serp("quantum computing")

        call_args = mock_conn.request.call_args
        payload = json.loads(call_args[0][2])
        assert payload["gl"] == "us"
        assert payload["hl"] == "en"

    @patch("http.client.HTTPSConnection")
    def test_proxy_mode_uses_set_tunnel(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", "10.0.0.1"), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", "8080"), \
             patch("examples.deepresearch.tools.tool_search.PROXY_USER", "user"), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PASS", "pass"):
            search_tool.google_search_with_serp("test")

        mock_conn.set_tunnel.assert_called_once()
        tunnel_args = mock_conn.set_tunnel.call_args
        assert tunnel_args[0][0] == "google.serper.dev"

    @patch("http.client.HTTPSConnection")
    def test_direct_mode_no_set_tunnel(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            search_tool.google_search_with_serp("test")

        mock_conn.set_tunnel.assert_not_called()

    @patch("http.client.HTTPSConnection")
    def test_retries_on_failure(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        # Fail 2 times then succeed
        mock_conn.request.side_effect = [
            Exception("conn error"),
            Exception("conn error"),
            None,
        ]
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            result = search_tool.google_search_with_serp("test")

        assert mock_conn.request.call_count == 3

    @patch("http.client.HTTPSConnection")
    def test_all_retries_fail(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_conn.request.side_effect = Exception("conn error")

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            result = search_tool.google_search_with_serp("test")

        assert "Timeout" in result or "timeout" in result.lower()

    @patch("http.client.HTTPSConnection")
    def test_parses_organic_results(self, mock_conn_cls, search_tool, mock_serp_response):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = mock_serp_response.encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            result = search_tool.google_search_with_serp("test")

        assert "Result 1" in result
        assert "https://a.com" in result
        assert "found 2 results" in result

    @patch("http.client.HTTPSConnection")
    def test_no_organic_results_fallback(self, mock_conn_cls, search_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"error": "no results"}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_search.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_search.PROXY_PORT", ""):
            result = search_tool.google_search_with_serp("obscure query")

        assert "No results found" in result
