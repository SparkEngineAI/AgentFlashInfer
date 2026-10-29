from early_stop.config import EarlyStopConfig
from early_stop.state import EarlyStopState
from early_stop.checker import EarlyStopChecker, EarlyStopResult
from early_stop.constants import REPEAT_OUTPUT_PROMPT, MIN_ITER_ANALYSIS_PROMPT


def _default_config(**kwargs):
    return EarlyStopConfig(**kwargs)


def _default_state(**kwargs):
    return EarlyStopState(**kwargs)


# ── Guard clause ──────────────────────────────────────────────

class TestGuardClause:
    def test_disabled_config_returns_none_reason(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=False),
            state=_default_state(),
            current_turn=5,
            tool_name="think",
            current_thought="hello",
        )
        assert result.should_inject is False
        assert result.reason == "none"
        assert result.injected_message is None

    def test_non_think_tool_returns_none_reason(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True),
            state=_default_state(),
            current_turn=5,
            tool_name="search",
            current_thought="hello",
        )
        assert result.should_inject is False
        assert result.reason == "none"

    def test_disabled_and_non_think_still_returns_state(self):
        original_state = _default_state(last_thought="old", last_early_stop_iter=3)
        result = EarlyStopChecker.check(
            config=_default_config(enabled=False),
            state=original_state,
            current_turn=5,
            tool_name="search",
            current_thought="new",
        )
        assert result.state is original_state


# ── Repeat output detection ───────────────────────────────────

class TestRepeatOutput:
    def test_repeat_output_triggers_inject(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True),
            state=_default_state(last_thought="same thought"),
            current_turn=3,
            tool_name="think",
            current_thought="same thought",
        )
        assert result.should_inject is True
        assert result.reason == "repeat"
        assert result.injected_message is not None
        assert result.injected_message.content == REPEAT_OUTPUT_PROMPT
        assert result.injected_message.role == "user"

    def test_repeat_output_updates_last_early_stop_iter(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True),
            state=_default_state(last_thought="same", last_early_stop_iter=-1),
            current_turn=4,
            tool_name="think",
            current_thought="same",
        )
        assert result.state.last_early_stop_iter == 4

    def test_repeat_output_preserves_last_thought(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True),
            state=_default_state(last_thought="dup"),
            current_turn=2,
            tool_name="think",
            current_thought="dup",
        )
        assert result.state.last_thought == "dup"

    def test_different_thought_no_repeat(self, capsys):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True),
            state=_default_state(last_thought="old"),
            current_turn=2,
            tool_name="think",
            current_thought="new",
        )
        assert result.reason != "repeat"

    def test_repeat_output_prints_marker(self, capsys):
        EarlyStopChecker.check(
            config=_default_config(enabled=True),
            state=_default_state(last_thought="x"),
            current_turn=1,
            tool_name="think",
            current_thought="x",
        )
        captured = capsys.readouterr()
        assert "repeat output detected" in captured.out


# ── Minimum iteration check ──────────────────────────────────

class TestMinIter:
    def test_min_iter_triggers_inject(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="old", last_early_stop_iter=-1),
            current_turn=3,
            tool_name="think",
            current_thought="new thought",
        )
        assert result.should_inject is True
        assert result.reason == "min_iter"
        assert result.injected_message is not None
        assert result.injected_message.content == MIN_ITER_ANALYSIS_PROMPT
        assert result.injected_message.role == "user"

    def test_min_iter_not_reached(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=5),
            state=_default_state(last_thought="old", last_early_stop_iter=-1),
            current_turn=3,
            tool_name="think",
            current_thought="new",
        )
        assert result.reason == "none"
        assert result.should_inject is False

    def test_min_iter_exact_boundary(self):
        """current_turn == min_iter 时应触发"""
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="old", last_early_stop_iter=-1),
            current_turn=2,
            tool_name="think",
            current_thought="new",
        )
        assert result.reason == "min_iter"

    def test_cooldown_prevents_consecutive_trigger(self):
        """last_early_stop_iter 与 current_turn 差值 <=1 时不触发"""
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="old", last_early_stop_iter=3),
            current_turn=4,
            tool_name="think",
            current_thought="new",
        )
        # 4 - 3 = 1, 不满足 > 1, 不触发
        assert result.reason == "none"

    def test_cooldown_allows_after_gap(self):
        """差值 > 1 时允许再次触发"""
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="old", last_early_stop_iter=2),
            current_turn=5,
            tool_name="think",
            current_thought="new",
        )
        # 5 - 2 = 3 > 1, 应触发
        assert result.reason == "min_iter"

    def test_min_iter_updates_state(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="old", last_early_stop_iter=-1),
            current_turn=3,
            tool_name="think",
            current_thought="new",
        )
        assert result.state.last_early_stop_iter == 3
        assert result.state.last_thought == "new"

    def test_min_iter_prints_marker(self, capsys):
        EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="old", last_early_stop_iter=-1),
            current_turn=3,
            tool_name="think",
            current_thought="new",
        )
        captured = capsys.readouterr()
        assert "min_iter reached" in captured.out


# ── Default (no early stop) ──────────────────────────────────

class TestDefaultNoTrigger:
    def test_no_trigger_updates_last_thought(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=100),
            state=_default_state(last_thought="old", last_early_stop_iter=-1),
            current_turn=1,
            tool_name="think",
            current_thought="fresh",
        )
        assert result.reason == "none"
        assert result.state.last_thought == "fresh"

    def test_no_trigger_preserves_last_early_stop_iter(self):
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=100),
            state=_default_state(last_thought="old", last_early_stop_iter=5),
            current_turn=1,
            tool_name="think",
            current_thought="fresh",
        )
        assert result.state.last_early_stop_iter == 5


# ── Priority: repeat > min_iter ──────────────────────────────

class TestPriority:
    def test_repeat_takes_priority_over_min_iter(self):
        """当同时满足重复检测和 min_iter 条件时，repeat 优先"""
        result = EarlyStopChecker.check(
            config=_default_config(enabled=True, min_iter=2),
            state=_default_state(last_thought="dup", last_early_stop_iter=-1),
            current_turn=3,
            tool_name="think",
            current_thought="dup",
        )
        assert result.reason == "repeat"


# ── EarlyStopResult dataclass ────────────────────────────────

class TestEarlyStopResult:
    def test_defaults(self):
        r = EarlyStopResult()
        assert r.should_inject is False
        assert r.injected_message is None
        assert r.state is None
        assert r.reason == "none"