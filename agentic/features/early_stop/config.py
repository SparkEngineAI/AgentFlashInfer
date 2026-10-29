from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopConfig:
    """早停配置，优先级：config_overrides > 全局 config > 默认值"""
    enabled: bool = True
    min_iter: int = 2

    @classmethod
    def from_context(cls, config_overrides: Optional[dict] = None) -> "EarlyStopConfig":
        """从运行时上下文构建配置，与 model_routing.py 使用相同模式"""
        import config.config as config
        overrides = config_overrides or {}
        return cls(
            enabled=overrides.get("early_stop_enabled",
                                  getattr(config, "early_stop_enabled", cls.enabled)),
            min_iter=overrides.get("early_stop_min_iter",
                                   getattr(config, "early_stop_min_iter", cls.min_iter)),
        )
    
    @classmethod
    def from_env(cls, early_stop_enabled: bool, early_stop_min_iter: int) -> "EarlyStopConfig":
        return cls(
            enabled=early_stop_enabled,
            min_iter=early_stop_min_iter,
        )
        