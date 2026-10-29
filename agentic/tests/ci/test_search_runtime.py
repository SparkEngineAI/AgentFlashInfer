import asyncio
import inspect
import logging
import os
import subprocess
import unittest
import sys
from pathlib import Path

from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8103/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("EXAMPLE_MODEL_NAME", "pangu_auto")
os.environ.setdefault("SEARCH_URL", "https://search.example.test")
os.environ.setdefault("SEARCH_API_KEY", "search-key")
os.environ.setdefault("URL_CRAWLER_API_KEYS", "crawler-key")
os.environ.pop("PROXY_CACHE", None)

from agent_infer.mcp_factory.tools.searcher import batch_web_search as search_module
from agent_infer.mcp_factory.tools.searcher import jina_reader as jina_module
from agent_infer.mcp_factory.tools.searcher import url_crawler as crawler_module
from agent_infer.mcp_factory.tools.searcher import url_crawler_compress as compress_module
from agent_infer.mcp_factory.tools.tool_result_format import MCPToolResult


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "organic": [
                {
                    "title": "result",
                    "link": "https://example.test/result",
                    "snippet": "snippet",
                }
            ]
        }


class FakeSession:
    def __init__(self):
        self.post_kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def mount(self, *_args):
        return None

    def post(self, *_args, **kwargs):
        self.post_kwargs = kwargs
        return FakeResponse()


class GenericSearchRuntimeTest(unittest.TestCase):
    def test_search_tools_import_without_forwarder_sdk(self):
        code = """
import importlib.abc
import sys

class BlockForwarder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if 'forwarder_client' in fullname:
            raise ImportError('Forwarder SDK is unavailable')

sys.meta_path.insert(0, BlockForwarder())
from agent_infer.mcp_factory.tools.searcher import batch_web_search
from agent_infer.mcp_factory.tools.searcher import url_crawler
from agent_infer.mcp_factory.tools.searcher import url_crawler_compress
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generic_search_does_not_configure_requests_proxy_when_unset(self):
        session = FakeSession()
        with patch.object(search_module, "requests", SimpleNamespace(Session=lambda: session)):
            result = search_module._generic_search("query", 3)

        self.assertTrue(result.success)
        self.assertIsNone(session.post_kwargs.get("proxies"))

        self.assertFalse(getattr(session, "trust_env", True))

    def test_jina_reader_keeps_tls_verification_enabled(self):
        response = SimpleNamespace(
            status_code=200,
            text="content",
            raise_for_status=lambda: None,
        )
        with patch.object(jina_module.requests, "get", return_value=response) as get:
            result = jina_module._jina_reader("https://example.test", 100)

        self.assertTrue(result.success)
        self.assertIsNot(get.call_args.kwargs.get("verify"), False)


class BatchSearchRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_batch_search_logs_non_empty_success_evidence(self):
        search_result = MCPToolResult(
            success=True,
            data={"organic": [{"title": "result", "link": "https://example.test"}]},
        )
        with (
            patch.object(search_module, "_generic_search", return_value=search_result),
            self.assertLogs(search_module.logger, level=logging.INFO) as logs,
        ):
            results = await search_module.batch_web_search.__wrapped__(["query"])

        self.assertTrue(results[0]["success"])
        self.assertIn(
            "[search] batch_web_search completed: successful_queries=1, result_items=1",
            "\n".join(logs.output),
        )


class CrawlerRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_direct_crawler_saves_jina_content(self):
        document = {"url": "https://example.test/article", "file_path": "article.md"}
        with (
            patch.object(crawler_module.requests, "get", return_value=SimpleNamespace(text="article content")) as get,
            patch.object(crawler_module, "file_write", new_callable=AsyncMock, return_value={"success": True}) as write,
        ):
            result = await inspect.unwrap(crawler_module.url_crawler)([document], "test-session")

        self.assertEqual(get.call_args.args[0], "https://r.jina.ai/https://example.test/article")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"],
                         f"Bearer {crawler_module.config.URL_CRAWLER_API_KEYS}")
        write.assert_awaited_once_with(file_path="article.md", content="article content",
                                      session_id="test-session", create_dirs=True)
        self.assertTrue(result["data"][0]["success"])
        self.assertEqual(result["metadata"]["successful_writes"], 1)

    async def test_compressed_crawler_saves_compressed_jina_content(self):
        document = {"url": "https://example.test/article", "file_path": "article.md"}
        with (
            patch.object(crawler_module.requests, "get", return_value=SimpleNamespace(text="article content")),
            patch.object(compress_module, "compress_content", new_callable=AsyncMock, return_value="summary") as compress,
            patch.object(compress_module, "file_write", new_callable=AsyncMock, return_value={"success": True}) as write,
        ):
            result = await inspect.unwrap(compress_module.url_crawler_compress)([document], "test-session")

        compress.assert_awaited_once_with("article content")
        self.assertEqual(write.call_args.kwargs["content"], "summary")
        self.assertTrue(result["data"][0]["success"])
        self.assertEqual(result["data"][0]["content_length"], len("summary"))


if __name__ == "__main__":
    GenericSearchRuntimeTest(
        "test_generic_search_does_not_configure_requests_proxy_when_unset"
    ).test_generic_search_does_not_configure_requests_proxy_when_unset()
    asyncio.run(BatchSearchRuntimeTest(
        "test_batch_search_logs_non_empty_success_evidence"
    ).test_batch_search_logs_non_empty_success_evidence())
    print("PASS: search runtime contract")
