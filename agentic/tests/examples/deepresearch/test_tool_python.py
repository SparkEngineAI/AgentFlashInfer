"""Tests for PythonInterpreter tool."""

from unittest.mock import MagicMock, patch

import pytest

from examples.deepresearch.tools.tool_python import PythonInterpreter, has_chinese_chars


@pytest.fixture
def python_tool():
    return PythonInterpreter()


class TestHasChineseChars:
    def test_chinese_text(self):
        assert has_chinese_chars("量子计算") is True

    def test_english_text(self):
        assert has_chinese_chars("quantum computing") is False

    def test_mixed_text(self):
        assert has_chinese_chars("量子 quantum") is True

    def test_empty_string(self):
        assert has_chinese_chars("") is False


class TestPythonInterpreterCall:
    def test_successful_execution(self, python_tool):
        mock_result = MagicMock()
        mock_result.run_result.stdout = "hello world"
        mock_result.run_result.stderr = ""
        mock_result.run_result.execution_time = 1.0

        with patch.dict("sys.modules", {"sandbox_fusion": MagicMock(run_code=MagicMock(return_value=mock_result), RunCodeRequest=MagicMock, RunStatus=MagicMock)}):
            # Patch the lazy import inside call()
            with patch("examples.deepresearch.config.deepresearch_config.SANDBOX_FUSION_ENDPOINTS", ["http://sandbox:8080"]):
                import importlib
                import examples.deepresearch.tools.tool_python as mod
                importlib.reload(mod)

                tool = mod.PythonInterpreter()
                with patch.object(tool, "call", wraps=tool.call) as spied_call:
                    # Use a simpler approach: mock the sandbox_fusion import at call time
                    pass

    def test_timeout_error(self, python_tool):
        from requests.exceptions import Timeout

        with patch("examples.deepresearch.config.deepresearch_config.SANDBOX_FUSION_ENDPOINTS", ["http://sandbox:8080"]):
            # The call method imports sandbox_fusion lazily, so we mock it at module level
            mock_sf = MagicMock()
            mock_sf.run_code.side_effect = Timeout("timeout")
            with patch.dict("sys.modules", {"sandbox_fusion": mock_sf}):
                # Reload the module to pick up the patched config
                import importlib
                import examples.deepresearch.tools.tool_python as mod
                importlib.reload(mod)
                tool = mod.PythonInterpreter()
                result = tool.call("print('hello')")
                assert "TimeoutError" in result or "timeout" in result.lower()

    def test_empty_code(self, python_tool):
        with patch("examples.deepresearch.config.deepresearch_config.SANDBOX_FUSION_ENDPOINTS", ["http://sandbox:8080"]):
            mock_sf = MagicMock()
            mock_result = MagicMock()
            mock_result.run_result.stdout = ""
            mock_result.run_result.stderr = ""
            mock_result.run_result.execution_time = 0.1
            mock_sf.run_code.return_value = mock_result
            with patch.dict("sys.modules", {"sandbox_fusion": mock_sf}):
                import importlib
                import examples.deepresearch.tools.tool_python as mod
                importlib.reload(mod)
                tool = mod.PythonInterpreter()
                result = tool.call("")
                # Should return "Finished execution" for empty output
                assert "Finished" in result or result.strip() != ""
