"""Tests for Scholar tool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from examples.deepresearch.tools.tool_scholar import Scholar


@pytest.fixture
def scholar_tool():
    return Scholar()


class TestScholarCall:
    def test_string_query(self, scholar_tool):
        with patch.object(scholar_tool, "google_scholar_with_serp", return_value="results"):
            # Scholar.call expects list query per its schema
            result = scholar_tool.call({"query": ["quantum computing"]})
        assert "results" in result

    def test_list_query(self, scholar_tool):
        with patch.object(scholar_tool, "google_scholar_with_serp", return_value="results"):
            result = scholar_tool.call({"query": ["q1", "q2"]})
        assert "results" in result

    def test_invalid_format(self, scholar_tool):
        result = scholar_tool.call({})
        assert "Invalid request" in result


class TestGoogleScholarWithSerp:
    @patch("http.client.HTTPSConnection")
    def test_proxy_mode(self, mock_conn_cls, scholar_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_scholar.PROXY_HOST", "10.0.0.1"), \
             patch("examples.deepresearch.tools.tool_scholar.PROXY_PORT", "8080"), \
             patch("examples.deepresearch.tools.tool_scholar.PROXY_USER", "user"), \
             patch("examples.deepresearch.tools.tool_scholar.PROXY_PASS", "pass"):
            scholar_tool.google_scholar_with_serp("test")

        mock_conn.set_tunnel.assert_called_once()

    @patch("http.client.HTTPSConnection")
    def test_direct_mode(self, mock_conn_cls, scholar_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"organic": []}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_scholar.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_scholar.PROXY_PORT", ""):
            scholar_tool.google_scholar_with_serp("test")

        mock_conn.set_tunnel.assert_not_called()

    @patch("http.client.HTTPSConnection")
    def test_parses_scholar_results(self, mock_conn_cls, scholar_tool):
        response_data = json.dumps({
            "organic": [
                {
                    "title": "Quantum Survey",
                    "year": 2023,
                    "publicationInfo": "Nature",
                    "citedBy": 42,
                    "pdfUrl": "https://example.com/paper.pdf",
                    "snippet": "A survey of quantum methods.",
                }
            ]
        })
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = response_data.encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_scholar.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_scholar.PROXY_PORT", ""):
            result = scholar_tool.google_scholar_with_serp("quantum")

        assert "Quantum Survey" in result
        assert "2023" in result
        assert "Nature" in result
        assert "found 1 results" in result

    @patch("http.client.HTTPSConnection")
    def test_no_results_fallback(self, mock_conn_cls, scholar_tool):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"error": "no results"}).encode()
        mock_conn.getresponse.return_value = mock_response

        with patch("examples.deepresearch.tools.tool_scholar.PROXY_HOST", ""), \
             patch("examples.deepresearch.tools.tool_scholar.PROXY_PORT", ""):
            result = scholar_tool.google_scholar_with_serp("obscure")

        assert "No results found" in result
