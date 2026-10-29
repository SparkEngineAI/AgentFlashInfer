"""Model factory — creates model instances for routing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from agents import Model, OpenAIChatCompletionsModel

    from .config import ModelRoutingConfig


class ModelFactory(ABC):
    """模型创建的抽象基类"""

    def __init__(self, config: ModelRoutingConfig) -> None:
        self._config = config

    def create(self, model_type: str) -> Model:
        """根据模型类型创建模型实例，model_type 为 "small_model" 或 "large_model" """
        if model_type == "small_model":
            return self.get_or_create_small_model()
        elif model_type == "large_model":
            return self.get_or_create_large_model()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @abstractmethod
    def get_or_create_small_model(self) -> Model:
        ...

    @abstractmethod
    def get_or_create_large_model(self) -> Model:
        ...


class OpenAIModelFactory(ModelFactory):
    """基于 AsyncOpenAI 的模型工厂（当前实现）

    缓存 AsyncOpenAI 客户端和 OpenAIChatCompletionsModel 实例，
    避免每轮重建连接和模型对象。
    """

    def __init__(self, config: ModelRoutingConfig) -> None:
        super().__init__(config)
        self._small_client: AsyncOpenAI | None = None
        self._large_client: AsyncOpenAI | None = None
        self._small_model: Model | None = None
        self._large_model: Model | None = None

    def get_or_create_small_model(self) -> Model:
        from agents import OpenAIChatCompletionsModel

        if self._small_model is None:
            client = self._get_or_create_client("small")
            self._small_model = OpenAIChatCompletionsModel(
                model=self._config.small_model_name, openai_client=client
            )
        return self._small_model

    def get_or_create_large_model(self) -> Model:
        from agents import OpenAIChatCompletionsModel

        if self._large_model is None:
            client = self._get_or_create_client("large")
            self._large_model = OpenAIChatCompletionsModel(
                model=self._config.large_model_name, openai_client=client
            )
        return self._large_model

    def _get_or_create_client(self, model_type: str) -> AsyncOpenAI:
        """缓存客户端实例，避免每轮重建连接"""
        if model_type == "small":
            if self._small_client is None:
                self._small_client = AsyncOpenAI(
                    base_url=self._config.small_model_url,
                    api_key=self._config.small_model_api_key or "",
                )
            return self._small_client
        elif model_type == "large":
            if self._large_client is None:
                self._large_client = AsyncOpenAI(
                    base_url=self._config.large_model_url,
                    api_key=self._config.large_model_api_key or "",
                )
            return self._large_client
        else:
            raise ValueError(f"Unknown model_type: {model_type}")