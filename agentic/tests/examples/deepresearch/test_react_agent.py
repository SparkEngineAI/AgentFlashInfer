"""Tests for MultiTurnReactAgent core logic."""

import json
import json5
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from examples.deepresearch.react_agent import (
    MultiTurnReactAgent,
    _get_next_port,
    _normalize_model_output,
    TOOL_MAP,
    OBS_START,
    OBS_END,
)
from examples.deepresearch.config import deepresearch_config

# The actual thinking tokens checked by sanity_check_output
THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'

# The actual action delimiters used for tool call detection
ACTION_START = '<tool_call>'
ACTION_END = '</tool_call>'


@pytest.fixture
def agent():
    llm_config = {
        "model": "/tmp/test-model",
        "generate_cfg": {
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 1.1,
        }
    }
    with patch("examples.deepresearch.react_agent.AutoTokenizer"):
        a = MultiTurnReactAgent(llm=llm_config)
    a.model = "test-model"
    return a


@pytest.fixture(autouse=True)
def reset_port_index():
    import examples.deepresearch.react_agent as mod
    mod._port_index = 0
    yield
    mod._port_index = 0


class TestGetNextPort:
    def test_round_robin(self):
        with patch("examples.deepresearch.react_agent.VLLM_PORTS", [7001, 7002]):
            import examples.deepresearch.react_agent as mod
            mod._port_index = 0
            p1 = _get_next_port()
            p2 = _get_next_port()
            assert p1 == 7001
            assert p2 == 7002

    def test_wrap_around(self):
        with patch("examples.deepresearch.react_agent.VLLM_PORTS", [7001, 7002]):
            import examples.deepresearch.react_agent as mod
            mod._port_index = 0
            p1 = _get_next_port()
            p2 = _get_next_port()
            p3 = _get_next_port()
            assert p3 == p1


class TestSanityCheckOutput:
    def test_valid_output(self, agent):
        content = f"{THINK_OPEN}reasoning{THINK_CLOSE} answer"
        assert agent.sanity_check_output(content) is True

    def test_invalid_output(self, agent):
        assert agent.sanity_check_output("plain text") is False

    def test_only_open_tag(self, agent):
        assert agent.sanity_check_output(f"{THINK_OPEN} reasoning") is False

    def test_only_close_tag(self, agent):
        assert agent.sanity_check_output(f"reasoning {THINK_CLOSE}") is False


class TestCallServer:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"name="search", "arguments": {"query": ["test"]}}',
            '{"name="search", "arguments": {"query": ["test"]}}}',
        ],
    )
    def test_normalizes_qwen_balanced_and_extra_tool_braces(self, payload):
        content = f"<think>reasoning</think>\n<tool_call>{payload}</tool_call>"

        result = _normalize_model_output(content)

        normalized = result.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        assert json5.loads(normalized)["name"] == "search"

    @patch("examples.deepresearch.react_agent.OpenAI")
    def test_normalizes_qwen_missing_think_open_and_malformed_tool_json(
        self, mock_openai_cls, agent
    ):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            'reasoning</think>\n<tool_call>\n'
            '{"name="search", "arguments": {"query": ["test"]}\n'
            '</tool_call>'
        )
        mock_openai_cls.return_value.chat.completions.create.return_value = (
            mock_response
        )

        result = agent.call_server([{"role": "user", "content": "hi"}], 7001)

        assert result.startswith("<think>\nreasoning</think>")
        payload = result.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        assert json5.loads(payload)["name"] == "search"

    @patch("examples.deepresearch.react_agent.OpenAI")
    def test_success(self, mock_openai_cls, agent):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  valid response  "
        mock_client.chat.completions.create.return_value = mock_response

        result = agent.call_server([{"role": "user", "content": "hi"}], 7001)
        assert result == "valid response"

    @patch("examples.deepresearch.react_agent.OpenAI")
    def test_empty_response_retries(self, mock_openai_cls, agent):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        empty_resp = MagicMock()
        empty_resp.choices = [MagicMock()]
        empty_resp.choices[0].message.content = ""
        valid_resp = MagicMock()
        valid_resp.choices = [MagicMock()]
        valid_resp.choices[0].message.content = "valid"
        mock_client.chat.completions.create.side_effect = [empty_resp, valid_resp]

        result = agent.call_server([{"role": "user", "content": "hi"}], 7001)
        assert result == "valid"

    @patch("examples.deepresearch.react_agent.OpenAI")
    def test_all_retries_fail(self, mock_openai_cls, agent):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("connection error")

        result = agent.call_server([{"role": "user", "content": "hi"}], 7001, max_tries=2)
        assert "error" in result.lower()

    @patch("examples.deepresearch.react_agent.OpenAI")
    def test_api_error_retries(self, mock_openai_cls, agent):
        from openai import APIError
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIError(
            message="server error", request=MagicMock(), body=None
        )
        result = agent.call_server([{"role": "user", "content": "hi"}], 7001, max_tries=1)
        assert "error" in result.lower()


class TestTokenizerProperty:
    def test_lazy_initialization(self, agent):
        mock_tok = MagicMock()
        with patch("examples.deepresearch.react_agent.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = mock_tok
            result = agent.tokenizer
            assert result == mock_tok
            mock_at.from_pretrained.assert_called_once()

    def test_cached_after_first_call(self, agent):
        mock_tok = MagicMock()
        with patch("examples.deepresearch.react_agent.AutoTokenizer") as mock_at:
            mock_at.from_pretrained.return_value = mock_tok
            _ = agent.tokenizer
            _ = agent.tokenizer
            mock_at.from_pretrained.assert_called_once()


class TestCountTokens:
    def test_counts_tokens(self, agent):
        mock_tok = MagicMock()
        mock_tok.apply_chat_template.return_value = "rendered prompt"
        mock_ids = MagicMock()
        mock_ids.__len__ = MagicMock(return_value=42)
        mock_result = {"input_ids": [mock_ids]}
        mock_tok.return_value = mock_result

        with patch.object(type(agent), "tokenizer", new_callable=PropertyMock, return_value=mock_tok):
            count = agent.count_tokens([{"role": "user", "content": "hi"}])
            assert count == 42


class TestCustomCallTool:
    def test_search_tool(self, agent):
        with patch.object(TOOL_MAP["search"], "call", return_value="search results"):
            result = agent.custom_call_tool("search", {"query": ["test"]})
        assert result == "search results"

    def test_unknown_tool(self, agent):
        result = agent.custom_call_tool("nonexistent_tool", {})
        assert "not found" in result.lower()

    def test_python_tool(self, agent):
        with patch.object(TOOL_MAP["PythonInterpreter"], "call", return_value="stdout: hello"):
            result = agent.custom_call_tool("PythonInterpreter", {"code": "print('hello')"})
        assert "hello" in result

    def test_scholar_tool(self, agent):
        with patch.object(TOOL_MAP["google_scholar"], "call", return_value="scholar results"):
            result = agent.custom_call_tool("google_scholar", {"query": ["quantum"]})
        assert result == "scholar results"

    def test_visit_tool(self, agent):
        with patch.object(TOOL_MAP["visit"], "call", return_value="visit results"):
            result = agent.custom_call_tool("visit", {"url": "https://example.com", "goal": "find info"})
        assert result == "visit results"

    def test_parse_file_tool(self, agent):
        with patch("examples.deepresearch.react_agent.asyncio") as mock_asyncio, \
             patch.object(TOOL_MAP["parse_file"], "call", return_value="parsed content"):
            mock_asyncio.run.return_value = "parsed content"
            result = agent.custom_call_tool("parse_file", {"files": ["doc.pdf"]})
        assert "parsed content" in result

    def test_python_in_name_dispatches_to_pythoninterpreter(self, agent):
        # The "python" in tool_name check dispatches to TOOL_MAP['PythonInterpreter']
        # but tool_name must be in TOOL_MAP first, so use "PythonInterpreter"
        with patch.object(TOOL_MAP["PythonInterpreter"], "call", return_value="python output"):
            result = agent.custom_call_tool("PythonInterpreter", {"code": "x=1"})
        assert "python output" in result


class TestRun:
    def test_normal_flow_with_answer(self, agent):
        answer_content = "<answer>The answer is 42</answer>"
        with patch.object(agent, "call_server", return_value=answer_content), \
             patch.object(agent, "count_tokens", return_value=1000):
            data = {"item": {"question": "test question", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["prediction"] == "The answer is 42"
        assert result["termination"] == "answer"

    def test_max_llm_calls_exceeded(self, agent):
        with patch("examples.deepresearch.react_agent.MAX_LLM_CALL_PER_RUN", 2), \
             patch.object(agent, "call_server", return_value="thinking without answer"), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["termination"] == "exceed available llm calls"

    def test_search_tool_call_then_answer(self, agent):
        """Test _run with a search tool call followed by an answer."""
        tool_call_content = (
            f"{THINK_OPEN}I need to search{THINK_CLOSE}\n"
            f"{ACTION_START}" + '{"name": "search", "arguments": {"query": ["quantum"]}}' + f"{ACTION_END}"
        )
        answer_content = "<answer>Quantum computing is... </answer>"
        with patch.object(agent, "call_server", side_effect=[tool_call_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100), \
             patch.object(TOOL_MAP["search"], "call", return_value="search results"):
            data = {"item": {"question": "What is quantum computing?", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["prediction"] == "Quantum computing is... "
        assert result["termination"] == "answer"

    def test_python_tool_call_in_run(self, agent):
        """Test _run with a python tool call containing code tags."""
        python_content = (
            f"{THINK_OPEN}Let me compute{THINK_CLOSE}\n"
            f"{ACTION_START}python code{ACTION_END}\n"
            "<code>print(2+2)</code>"
        )
        answer_content = "<answer>4</answer>"
        with patch.object(agent, "call_server", side_effect=[python_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100), \
             patch.object(TOOL_MAP["PythonInterpreter"], "call", return_value="stdout: 4"):
            data = {"item": {"question": "What is 2+2?", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["prediction"] == "4"

    def test_python_tool_formatting_error(self, agent):
        """Test _run when python tool call has bad formatting (no code tags)."""
        python_content = (
            f"{THINK_OPEN}Compute{THINK_CLOSE}\n"
            f"{ACTION_START}python code{ACTION_END}"
        )
        answer_content = "<answer>Cannot compute</answer>"
        with patch.object(agent, "call_server", side_effect=[python_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["termination"] == "answer"

    def test_invalid_json_tool_call(self, agent):
        """Test _run when tool call JSON is invalid."""
        invalid_tool_content = (
            f"{THINK_OPEN}Search{THINK_CLOSE}\n"
            f"{ACTION_START}not valid json{ACTION_END}"
        )
        answer_content = "<answer>Could not parse</answer>"
        with patch.object(agent, "call_server", side_effect=[invalid_tool_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["termination"] == "answer"

    def test_timeout_termination(self, agent):
        """Test _run when timeout is reached."""
        with patch("examples.deepresearch.react_agent.MAX_TIMEOUT_MINUTES", 0), \
             patch.object(agent, "call_server", return_value="thinking"), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert "timeout" in result["termination"].lower() or "No answer found" in result["termination"]

    def test_token_limit_exceeded_with_answer(self, agent):
        """Test _run when token count exceeds limit and LLM returns an answer."""
        thinking_content = "thinking about the problem"
        answer_content = "<answer>Final answer</answer>"
        with patch.object(agent, "call_server", side_effect=[thinking_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=200000), \
             patch("examples.deepresearch.react_agent.MAX_TOKEN_LIMIT", 100000):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["prediction"] == "Final answer"
        assert "token limit" in result["termination"]

    def test_token_limit_exceeded_without_answer(self, agent):
        """Test _run when token count exceeds limit and LLM returns no answer."""
        thinking_content = "thinking"
        no_answer_content = "I do not know the answer"
        with patch.object(agent, "call_server", side_effect=[thinking_content, no_answer_content]), \
             patch.object(agent, "count_tokens", return_value=200000), \
             patch("examples.deepresearch.react_agent.MAX_TOKEN_LIMIT", 100000):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert "token limit" in result["termination"]
        assert "format error" in result["termination"]

    def test_obs_start_truncation(self, agent):
        """Test _run truncates content at OBS_START."""
        content_with_obs = f"thinking part{OBS_START}observation data{OBS_END} more text"
        answer_content = "<answer>Done</answer>"
        with patch.object(agent, "call_server", side_effect=[content_with_obs, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run(data, "test-model")
        assert result["termination"] == "answer"

    def test_messages_format_question(self, agent):
        """Test _run when item has messages format instead of question."""
        answer_content = "<answer>Answer from messages</answer>"
        with patch.object(agent, "call_server", return_value=answer_content), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {
                "item": {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "User: What is AI?"},
                    ],
                    "answer": ""
                },
                "planning_port": 7001
            }
            result = agent._run(data, "test-model")
        assert result["prediction"] == "Answer from messages"

    def test_messages_format_without_user_prefix(self, agent):
        """Test _run with messages format that has no User: prefix."""
        answer_content = "<answer>Direct answer</answer>"
        with patch.object(agent, "call_server", return_value=answer_content), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {
                "item": {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "What is AI?"},
                    ],
                    "answer": ""
                },
                "planning_port": 7001
            }
            result = agent._run(data, "test-model")
        assert result["prediction"] == "Direct answer"


class TestRunWithEvents:
    def test_plain_progress_without_think_tags_does_not_crash(self, agent):
        manager = MagicMock()
        manager.gen_agent_id.return_value = "agent-1"
        data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}

        with patch("examples.deepresearch.react_agent.es_manager", manager), \
             patch.object(
                 agent,
                 "call_server",
                 side_effect=["plain progress", "<answer>done</answer>"],
             ), \
             patch.object(agent, "count_tokens", return_value=100):
            result = agent._run_with_events(data, "test-model", MagicMock())

        assert result["prediction"] == "done"
        manager.check.assert_not_called()

    def test_initializes_and_removes_generated_early_stop_agent_id(self, agent):
        emitter = MagicMock()
        manager = MagicMock()
        manager.gen_agent_id.return_value = "agent-1"
        data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}

        with patch("examples.deepresearch.react_agent.es_manager", manager), \
             patch.object(agent, "call_server", return_value="<answer>done</answer>"), \
             patch.object(agent, "count_tokens", return_value=100):
            result = agent._run_with_events(data, "test-model", emitter)

        assert result["prediction"] == "done"
        manager.init_agent.assert_called_once_with("agent-1")
        manager.remove_agent.assert_called_once_with("agent-1")

    def test_initializes_enabled_compression_manager(self, agent):
        config = MagicMock()
        config.enabled = True
        config.token_threshold = 1
        config.trigger_tool_names = ["search"]
        manager = MagicMock()
        manager.on_turn_start.side_effect = lambda _round, messages: messages
        emitter = MagicMock()
        data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}

        with patch("examples.deepresearch.react_agent.es_manager") as early_stop, \
             patch("examples.deepresearch.react_agent.DeepResearchConversationAdapter") as adapter_cls, \
             patch("examples.deepresearch.react_agent.CompressionManager", return_value=manager) as manager_cls, \
             patch.object(agent, "call_server", return_value="<answer>done</answer>"), \
             patch.object(agent, "count_tokens", return_value=100):
            early_stop.gen_agent_id.return_value = "agent-1"
            agent._compression_config = config
            result = agent._run_with_events(data, "test-model", emitter)

        assert result["prediction"] == "done"
        adapter_cls.assert_called_once()
        manager_cls.assert_called_once()
        assert manager.on_turn_start.call_count == 2

    def test_logs_early_stop_injection_and_next_turn_observation(self, agent, capsys):
        manager = MagicMock()
        manager.gen_agent_id.return_value = "agent-1"
        injected_message = MagicMock()
        injected_message.content = "review progress"
        manager.check.return_value = MagicMock(
            should_inject=True,
            injected_message=injected_message,
            reason="min_iter",
        )
        data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}

        with patch("examples.deepresearch.react_agent.es_manager", manager), \
             patch.object(
                 agent,
                 "call_server",
                 side_effect=["<think>continue</think>", "<answer>done</answer>"],
             ), \
             patch.object(agent, "count_tokens", return_value=100):
            result = agent._run_with_events(data, "test-model", MagicMock())

        assert result["prediction"] == "done"
        output = capsys.readouterr().out
        assert "[early_stop] prompt injected: reason=min_iter, iter=1, target=new_step_items" in output
        assert "[early_stop] prompt observed in next turn: reason=min_iter, injected_iter=1, current_iter=2" in output

    def test_does_not_log_observed_prompt_removed_by_compression(self, agent, capsys):
        config = MagicMock()
        config.enabled = True
        config.token_threshold = 1
        config.trigger_tool_names = ["search"]
        compression_manager = MagicMock()
        compression_manager.on_turn_start.side_effect = (
            lambda current_turn, messages: [] if current_turn == 2 else messages
        )
        early_stop = MagicMock()
        early_stop.gen_agent_id.return_value = "agent-1"
        injected_message = MagicMock()
        injected_message.content = "review progress"
        early_stop.check.return_value = MagicMock(
            should_inject=True,
            injected_message=injected_message,
            reason="min_iter",
        )
        data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}

        with patch("examples.deepresearch.react_agent.es_manager", early_stop), \
             patch(
                 "examples.deepresearch.react_agent.CompressionManager",
                 return_value=compression_manager,
             ), \
             patch.object(
                 agent,
                 "call_server",
                 side_effect=["<think>continue</think>", "<answer>done</answer>"],
             ), \
             patch.object(agent, "count_tokens", return_value=100):
            agent._compression_config = config
            result = agent._run_with_events(data, "test-model", MagicMock())

        assert result["prediction"] == "done"
        output = capsys.readouterr().out
        assert "prompt injected: reason=min_iter" in output
        assert "prompt observed in next turn" not in output

    def test_emits_thinking_and_answer_events(self, agent):
        answer_content = "<answer>The answer is 42</answer>"
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch.object(agent, "call_server", return_value=answer_content), \
             patch.object(agent, "count_tokens", return_value=1000):
            data = {"item": {"question": "test question", "answer": ""}, "planning_port": 7001}
            result = agent._run_with_events(data, "test-model", capture_emitter)
        event_types = [e[1]["type"] for e in events]
        assert "thinking" in event_types
        assert "answer" in event_types
        assert result["prediction"] == "The answer is 42"

    def test_emits_tool_call_and_result_events(self, agent):
        """Test _run_with_events emits tool_call and tool_result events."""
        tool_call_content = (
            f"{THINK_OPEN}I need to search{THINK_CLOSE}\n"
            f"{ACTION_START}" + '{"name": "search", "arguments": {"query": ["quantum"]}}' + f"{ACTION_END}"
        )
        answer_content = "<answer>Found it</answer>"
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch.object(agent, "call_server", side_effect=[tool_call_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100), \
             patch.object(TOOL_MAP["search"], "call", return_value="search results"):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run_with_events(data, "test-model", capture_emitter)
        event_types = [e[1]["type"] for e in events]
        assert "thinking" in event_types
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "answer" in event_types

    def test_emits_python_tool_call_events(self, agent):
        """Test _run_with_events with python tool call."""
        python_content = (
            f"{THINK_OPEN}Run code{THINK_CLOSE}\n"
            f"{ACTION_START}python code{ACTION_END}\n"
            "<code>print(1+1)</code>"
        )
        answer_content = "<answer>2</answer>"
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch.object(agent, "call_server", side_effect=[python_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100), \
             patch.object(TOOL_MAP["PythonInterpreter"], "call", return_value="stdout: 2"):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run_with_events(data, "test-model", capture_emitter)
        tool_call_events = [e for e in events if e[1].get("type") == "tool_call"]
        assert len(tool_call_events) >= 1
        assert tool_call_events[0][1]["tool_name"] == "PythonInterpreter"

    def test_emits_timeout_event(self, agent):
        """Test _run_with_events emits termination event on timeout."""
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch("examples.deepresearch.react_agent.MAX_TIMEOUT_MINUTES", 0), \
             patch.object(agent, "call_server", return_value="thinking"), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run_with_events(data, "test-model", capture_emitter)
        termination_events = [e for e in events if e[1].get("type") == "termination"]
        assert len(termination_events) >= 1
        assert termination_events[0][1]["reason"] == "timeout"

    def test_emits_token_limit_answer_event(self, agent):
        """Test _run_with_events emits answer event when token limit reached."""
        thinking_content = "thinking"
        answer_content = "<answer>Limited answer</answer>"
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch.object(agent, "call_server", side_effect=[thinking_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=200000), \
             patch("examples.deepresearch.react_agent.MAX_TOKEN_LIMIT", 100000):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run_with_events(data, "test-model", capture_emitter)
        answer_events = [e for e in events if e[1].get("type") == "answer"]
        assert len(answer_events) >= 1

    def test_invalid_json_tool_call_emits_unknown(self, agent):
        """Test _run_with_events when tool call JSON is invalid."""
        invalid_content = (
            f"{THINK_OPEN}Bad call{THINK_CLOSE}\n"
            f"{ACTION_START}not json{ACTION_END}"
        )
        answer_content = "<answer>error recovered</answer>"
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch.object(agent, "call_server", side_effect=[invalid_content, answer_content]), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {"item": {"question": "test", "answer": ""}, "planning_port": 7001}
            result = agent._run_with_events(data, "test-model", capture_emitter)
        tool_result_events = [e for e in events if e[1].get("type") == "tool_result"]
        assert len(tool_result_events) >= 1
        assert tool_result_events[0][1]["tool_name"] == "unknown"

    def test_messages_format(self, agent):
        """Test _run_with_events with messages format input."""
        answer_content = "<answer>Messages answer</answer>"
        events = []
        def capture_emitter(event_type, event_data):
            events.append((event_type, event_data))
        with patch.object(agent, "call_server", return_value=answer_content), \
             patch.object(agent, "count_tokens", return_value=100):
            data = {
                "item": {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "User: What is AI?"},
                    ],
                    "answer": ""
                },
                "planning_port": 7001
            }
            result = agent._run_with_events(data, "test-model", capture_emitter)
        assert result["prediction"] == "Messages answer"


class TestAsyncRun:
    @pytest.mark.asyncio
    async def test_delegates_to_run(self, agent):
        with patch.object(agent, "_run", return_value={"prediction": "test", "termination": "answer"}) as mock_run:
            result = await agent.async_run(
                {"item": {"question": "q", "answer": ""}, "planning_port": 7001},
                "test-model"
            )
        mock_run.assert_called_once()
        assert result["prediction"] == "test"

    @pytest.mark.asyncio
    async def test_delegates_to_run_with_events(self, agent):
        with patch.object(agent, "_run_with_events", return_value={"prediction": "test", "termination": "answer"}) as mock_run:
            emitter = lambda etype, edata: None
            result = await agent.async_run(
                {"item": {"question": "q", "answer": ""}, "planning_port": 7001},
                "test-model",
                sse_emitter=emitter
            )
        mock_run.assert_called_once()
        assert result["prediction"] == "test"
AGENTIC_DIR = Path(__file__).resolve().parents[3]


def test_model_routing_uses_shared_switch_environment_variable() -> None:
    env = os.environ.copy()
    env["MODEL_ROUTING_SWITCH"] = "true"
    env.pop("MODEL_ROUTING_ENABLED", None)
    env["PYTHONPATH"] = f"{AGENTIC_DIR}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from examples.deepresearch.config import deepresearch_config as c; "
            "assert c.MODEL_ROUTING_SWITCH is True; "
            "assert not hasattr(c, 'MODEL_ROUTING_ENABLED')",
        ],
        cwd=AGENTIC_DIR,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr

