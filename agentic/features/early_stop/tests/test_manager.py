import threading
from unittest.mock import patch, MagicMock

from early_stop.config import EarlyStopConfig
from early_stop.state import EarlyStopState
from early_stop.checker import EarlyStopResult
from early_stop.manager import EarlyStopManager


class TestEarlyStopManagerInit:
    def test_initial_config_is_default(self):
        mgr = EarlyStopManager()
        assert mgr._config.enabled is True
        assert mgr._config.min_iter == 2

    def test_initial_states_empty(self):
        mgr = EarlyStopManager()
        assert len(mgr._states) == 0


class TestCheck:
    def test_check_creates_state_for_new_agent(self):
        mgr = EarlyStopManager()
        result = mgr.check("agent-1", current_turn=1, tool_name="search", current_thought="hi")
        assert "agent-1" in mgr._states

    def test_check_persists_state(self):
        mgr = EarlyStopManager()
        mgr.check("agent-1", current_turn=1, tool_name="think", current_thought="hello")
        # 第二次调用应使用上次的 state
        result = mgr.check("agent-1", current_turn=2, tool_name="think", current_thought="hello")
        assert result.reason == "repeat"

    def test_check_delegates_to_checker(self):
        mgr = EarlyStopManager()
        result = mgr.check(
            "agent-1",
            current_turn=5,
            tool_name="think",
            current_thought="fresh",
        )
        # min_iter=2, turn=5, 首次 last_early_stop_iter=-1, 差值 > 1
        assert result.reason == "min_iter"
        assert result.should_inject is True

    def test_multiple_agents_independent(self):
        mgr = EarlyStopManager()
        mgr.check("agent-A", current_turn=1, tool_name="think", current_thought="thought-A")
        mgr.check("agent-B", current_turn=1, tool_name="think", current_thought="thought-B")

        # agent-A 重复
        result_a = mgr.check("agent-A", current_turn=2, tool_name="think", current_thought="thought-A")
        assert result_a.reason == "repeat"

        # agent-B 不重复
        result_b = mgr.check("agent-B", current_turn=2, tool_name="think", current_thought="thought-B-new")
        assert result_b.reason != "repeat"


class TestInitAgent:
    def test_init_agent_creates_fresh_state(self):
        mgr = EarlyStopManager()
        mgr.init_agent("agent-1")
        assert "agent-1" in mgr._states
        assert mgr._states["agent-1"] == EarlyStopState()

    def test_init_agent_resets_existing_state(self):
        mgr = EarlyStopManager()
        mgr.check("agent-1", current_turn=5, tool_name="think", current_thought="hello")
        assert mgr._states["agent-1"].last_thought == "hello"

        mgr.init_agent("agent-1")
        assert mgr._states["agent-1"] == EarlyStopState()


class TestRemoveAgent:
    def test_remove_agent_deletes_state(self):
        mgr = EarlyStopManager()
        mgr.init_agent("agent-1")
        mgr.remove_agent("agent-1")
        assert "agent-1" not in mgr._states

    def test_remove_nonexistent_agent_no_error(self):
        mgr = EarlyStopManager()
        mgr.remove_agent("ghost")  # 不应抛异常


class TestRefreshConfig:
    def test_refresh_config_updates(self):
        mock_config = MagicMock(spec=[])
        mock_config.early_stop_enabled = False
        mock_config.early_stop_min_iter = 99
        with patch.dict("sys.modules", {"config.config": mock_config}):
            mgr = EarlyStopManager()
            mgr.refresh_config()
        assert mgr._config.enabled is False
        assert mgr._config.min_iter == 99

    def test_refresh_config_with_overrides(self):
        mock_config = MagicMock(spec=[])
        with patch.dict("sys.modules", {"config.config": mock_config}):
            mgr = EarlyStopManager()
            mgr.refresh_config(config_overrides={"early_stop_enabled": False, "early_stop_min_iter": 10})
        assert mgr._config.enabled is False
        assert mgr._config.min_iter == 10


class TestThreadSafety:
    def test_concurrent_checks(self):
        """多线程并发 check 不应抛异常"""
        mgr = EarlyStopManager()
        errors = []

        def worker(agent_id, turns):
            try:
                for t in range(1, turns + 1):
                    mgr.check(agent_id, current_turn=t, tool_name="think", current_thought=f"thought-{t}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"agent-{i}", 50))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(mgr._states) == 10

    def test_concurrent_init_and_remove(self):
        mgr = EarlyStopManager()
        errors = []

        def init_worker(i):
            try:
                for _ in range(100):
                    mgr.init_agent(f"agent-{i}")
            except Exception as e:
                errors.append(e)

        def remove_worker(i):
            try:
                for _ in range(100):
                    mgr.remove_agent(f"agent-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=init_worker, args=(i % 5,)) if i % 2 == 0
            else threading.Thread(target=remove_worker, args=(i % 5,))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0