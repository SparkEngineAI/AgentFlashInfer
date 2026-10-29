from early_stop.constants import (
    LOG_MARKER_REPEAT,
    LOG_MARKER_MIN_ITER,
    REPEAT_OUTPUT_PROMPT,
    MIN_ITER_ANALYSIS_PROMPT,
)


def test_log_marker_repeat_value():
    assert LOG_MARKER_REPEAT == "repeat output detected"


def test_log_marker_min_iter_value():
    assert LOG_MARKER_MIN_ITER == "min_iter reached"


def test_repeat_output_prompt_not_empty():
    assert isinstance(REPEAT_OUTPUT_PROMPT, str)
    assert len(REPEAT_OUTPUT_PROMPT) > 0


def test_min_iter_analysis_prompt_not_empty():
    assert isinstance(MIN_ITER_ANALYSIS_PROMPT, str)
    assert len(MIN_ITER_ANALYSIS_PROMPT) > 0


def test_repeat_output_prompt_is_chinese():
    assert "请停止重复" in REPEAT_OUTPUT_PROMPT


def test_min_iter_analysis_prompt_contains_keywords():
    assert "分析" in MIN_ITER_ANALYSIS_PROMPT
    assert "总结" in MIN_ITER_ANALYSIS_PROMPT