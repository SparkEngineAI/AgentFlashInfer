from early_stop.state import EarlyStopState


class TestEarlyStopStateDefaults:
    def test_default_last_thought(self):
        state = EarlyStopState()
        assert state.last_thought == ""

    def test_default_last_early_stop_iter(self):
        state = EarlyStopState()
        assert state.last_early_stop_iter == -1


class TestEarlyStopStateCustom:
    def test_custom_last_thought(self):
        state = EarlyStopState(last_thought="thinking about X")
        assert state.last_thought == "thinking about X"

    def test_custom_last_early_stop_iter(self):
        state = EarlyStopState(last_early_stop_iter=3)
        assert state.last_early_stop_iter == 3

    def test_full_custom(self):
        state = EarlyStopState(last_thought="abc", last_early_stop_iter=5)
        assert state.last_thought == "abc"
        assert state.last_early_stop_iter == 5


class TestEarlyStopStateEquality:
    def test_equal_states(self):
        s1 = EarlyStopState(last_thought="x", last_early_stop_iter=1)
        s2 = EarlyStopState(last_thought="x", last_early_stop_iter=1)
        assert s1 == s2

    def test_unequal_thought(self):
        s1 = EarlyStopState(last_thought="a", last_early_stop_iter=1)
        s2 = EarlyStopState(last_thought="b", last_early_stop_iter=1)
        assert s1 != s2

    def test_unequal_iter(self):
        s1 = EarlyStopState(last_thought="a", last_early_stop_iter=1)
        s2 = EarlyStopState(last_thought="a", last_early_stop_iter=2)
        assert s1 != s2