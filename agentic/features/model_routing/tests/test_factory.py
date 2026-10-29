"""Tests for ModelFactory and OpenAIModelFactory."""

from unittest.mock import MagicMock, patch

import pytest

from features.model_routing.config import ModelRoutingConfig
from features.model_routing.factory import ModelFactory, OpenAIModelFactory


# ── ModelFactory ABC tests ──


class DummyFactory(ModelFactory):
    def get_or_create_small_model(self):
        return MagicMock(_model_type="small", model=self._config.small_model_name)

    def get_or_create_large_model(self):
        return MagicMock(_model_type="large", model=self._config.large_model_name)


def test_factory_create_dispatches_by_type():
    """create() 应根据 model_type 分发到对应方法"""
    config = ModelRoutingConfig()
    factory = DummyFactory(config=config)
    small = factory.create("small_model")
    assert small._model_type == "small"
    large = factory.create("large_model")
    assert large._model_type == "large"


def test_factory_create_raises_on_unknown_type():
    """create() 传入未知 model_type 应抛出 ValueError"""
    factory = DummyFactory(config=ModelRoutingConfig())
    with pytest.raises(ValueError, match="Unknown model_type"):
        factory.create("unknown_model")


# ── OpenAIModelFactory tests ──
# OpenAIChatCompletionsModel is lazy-imported inside methods (from agents import ...),
# so we patch at the source module `agents`.


class TestOpenAIModelFactory:

    @pytest.fixture
    def small_config(self):
        return ModelRoutingConfig(
            small_model_url="http://7B:7000/v1",
            small_model_name="pangu_small",
            small_model_api_key="sk-small",
        )

    @pytest.fixture
    def large_config(self):
        return ModelRoutingConfig(
            large_model_url="http://38B:7000/v1",
            large_model_name="pangu_large",
            large_model_api_key="sk-large",
        )

    def test_creates_small_model(self, small_config):
        """OpenAIModelFactory 应创建基于 small URL 的 Model"""
        with patch("features.model_routing.factory.AsyncOpenAI") as mock_openai, \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_model_cls.return_value = MagicMock(_model_type="small")
            factory = OpenAIModelFactory(config=small_config)
            model = factory.get_or_create_small_model()
            assert model._model_type == "small"
            mock_openai.assert_called_once_with(
                base_url="http://7B:7000/v1", api_key="sk-small"
            )
            mock_model_cls.assert_called_once_with(
                model="pangu_small", openai_client=mock_client
            )

    def test_creates_large_model(self, large_config):
        """OpenAIModelFactory 应创建基于 large URL 的 Model"""
        with patch("features.model_routing.factory.AsyncOpenAI") as mock_openai, \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_model_cls.return_value = MagicMock(_model_type="large")
            factory = OpenAIModelFactory(config=large_config)
            model = factory.get_or_create_large_model()
            assert model._model_type == "large"
            mock_openai.assert_called_once_with(
                base_url="http://38B:7000/v1", api_key="sk-large"
            )

    def test_caches_small_model_instance(self, small_config):
        """同一 factory 应复用 small Model 实例"""
        with patch("features.model_routing.factory.AsyncOpenAI"), \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_model = MagicMock(_model_type="small")
            mock_model_cls.return_value = mock_model
            factory = OpenAIModelFactory(config=small_config)
            model1 = factory.get_or_create_small_model()
            model2 = factory.get_or_create_small_model()
            assert model1 is model2
            assert mock_model_cls.call_count == 1

    def test_caches_large_model_instance(self, large_config):
        """同一 factory 应复用 large Model 实例"""
        with patch("features.model_routing.factory.AsyncOpenAI"), \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_model = MagicMock(_model_type="large")
            mock_model_cls.return_value = mock_model
            factory = OpenAIModelFactory(config=large_config)
            model1 = factory.get_or_create_large_model()
            model2 = factory.get_or_create_large_model()
            assert model1 is model2
            assert mock_model_cls.call_count == 1

    def test_small_and_large_models_are_separate(self):
        """small 和 large Model 应分别缓存，互不干扰"""
        config = ModelRoutingConfig(
            small_model_url="http://7B:7000/v1",
            large_model_url="http://38B:7000/v1",
            small_model_name="pangu_small",
            large_model_name="pangu_large",
        )
        with patch("features.model_routing.factory.AsyncOpenAI") as mock_openai, \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_openai.return_value = MagicMock()
            small_model = MagicMock(_model_type="small")
            large_model = MagicMock(_model_type="large")
            mock_model_cls.side_effect = [small_model, large_model]
            factory = OpenAIModelFactory(config=config)
            s = factory.get_or_create_small_model()
            l = factory.get_or_create_large_model()
            assert s is small_model
            assert l is large_model
            assert mock_openai.call_count == 2

    def test_create_dispatches_to_small(self, small_config):
        """create("small_model") 应等价于 get_or_create_small_model()"""
        with patch("features.model_routing.factory.AsyncOpenAI"), \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_model_cls.return_value = MagicMock(_model_type="small")
            factory = OpenAIModelFactory(config=small_config)
            model = factory.create("small_model")
            assert model._model_type == "small"

    def test_create_dispatches_to_large(self, large_config):
        """create("large_model") 应等价于 get_or_create_large_model()"""
        with patch("features.model_routing.factory.AsyncOpenAI"), \
             patch("agents.OpenAIChatCompletionsModel", create=True) as mock_model_cls:
            mock_model_cls.return_value = MagicMock(_model_type="large")
            factory = OpenAIModelFactory(config=large_config)
            model = factory.create("large_model")
            assert model._model_type == "large"