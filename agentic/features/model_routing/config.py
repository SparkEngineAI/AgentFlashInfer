"""Model routing configuration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class ModelRoutingConfig:
    """Model routing 全部配置，不依赖全局 config 模块"""

    # --- 开关 ---
    enabled: bool = False

    # --- 模型端点 ---
    small_model_url: str = ""
    large_model_url: str = ""
    small_model_name: str = ""
    large_model_name: str = ""
    small_model_api_key: str = ""
    large_model_api_key: str = ""
    small_model_template: str = ""
    large_model_template: str = ""

    # env key → (field name, default)
    _ENV_MAPPING: ClassVar[dict[str, tuple[str, Any]]] = {
        "MODEL_ROUTING_SWITCH": ("enabled", False),
        "SMALL_MODEL_URL": ("small_model_url", ""),
        "LARGE_MODEL_URL": ("large_model_url", ""),
        "SMALL_MODEL_NAME": ("small_model_name", ""),
        "LARGE_MODEL_NAME": ("large_model_name", ""),
        "SMALL_MODEL_API_KEY": ("small_model_api_key", ""),
        "LARGE_MODEL_API_KEY": ("large_model_api_key", ""),
        "SMALL_MODEL_TEMPLATE": ("small_model_template", ""),
    }

    @classmethod
    def from_dict(cls, d: dict) -> ModelRoutingConfig:
        """从 context["config"] 或任意 dict 构造，忽略未知 key"""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_config_module(cls, overrides: dict, config_module: Any) -> ModelRoutingConfig:
        """从 context["config"] + 全局 config 模块构造，env key 缺失时从 config_module 读取默认值

        overrides 中的 key 可以是 env key（如 MODEL_ROUTING_SWITCH）或 field name（如 enabled），
        env key 会被映射为对应的 field name。
        """
        resolved: dict = {}
        for env_key, (field_name, default) in cls._ENV_MAPPING.items():
            if env_key in overrides:
                resolved[field_name] = overrides[env_key]
            elif field_name in overrides:
                resolved[field_name] = overrides[field_name]
            elif env_key.lower() in overrides:
                resolved[field_name] = overrides[env_key.lower()]
            else:
                resolved[field_name] = getattr(config_module, env_key, default)
        return cls.from_dict(resolved)
