"""Focused behavior tests for the DeepResearch model router."""

from unittest.mock import MagicMock, patch

from examples.deepresearch.features.model_router import (
    DRModelRouter,
    DRProgressCheckStrategy,
)
from features.model_routing.config import ModelRoutingConfig


def _routing_config() -> ModelRoutingConfig:
    return ModelRoutingConfig(
        enabled=True,
        small_model_url="http://small/v1",
        large_model_url="http://large/v1",
        small_model_name="small-model",
        large_model_name="large-model",
    )


def test_progress_check_uses_tongyi_compatible_messages() -> None:
    strategy = DRProgressCheckStrategy(
        config=_routing_config(),
        trigger_tool_name="<think>",
        progress_check_min_iter=1,
    )
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": "FALSE"}}]
    }

    with patch(
        "examples.deepresearch.features.model_router.requests.post",
        return_value=response,
    ) as post:
        should_escalate = strategy.should_escalate_sync(
            "DeepResearch",
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>search</think>"},
                {"role": "user", "content": "search result"},
            ],
        )

    assert should_escalate is True
    payload = post.call_args.kwargs["json"]
    assert {message["role"] for message in payload["messages"]} <= {"system", "user"}
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 8
    assert payload["structured_outputs"] == {"choice": ["TRUE", "FALSE"]}


def test_large_round_count_includes_escalation_call() -> None:
    config = _routing_config()
    strategy = DRProgressCheckStrategy(config=config, large_rounds=1)
    router = DRModelRouter(config=config, strategy=strategy)

    with patch.object(strategy, "should_escalate_sync", return_value=True):
        first = router.resolve("DeepResearch", [])
        second = router.resolve("DeepResearch", [])

    assert first.model_type == "large_model"
    assert second.model_type == "small_model"
