"""Tests for ModelRoutingConfig."""

from types import SimpleNamespace

from features.model_routing.config import ModelRoutingConfig


def test_default_values():
    """Config 默认值应符合设计"""
    config = ModelRoutingConfig()
    assert config.enabled is False
    assert config.small_model_url == ""
    assert config.large_model_url == ""
    assert config.small_model_name == ""
    assert config.large_model_name == ""
    assert config.small_model_api_key == ""
    assert config.large_model_api_key == ""
    assert config.small_model_template == ""
    assert config.large_model_template == ""


def test_from_dict_with_known_keys():
    """from_dict 应只接受已知字段，忽略未知 key"""
    config = ModelRoutingConfig.from_dict({
        "enabled": True,
        "small_model_url": "http://7B:7000/v1",
        "unknown_key": "should_be_ignored",
    })
    assert config.enabled is True
    assert config.small_model_url == "http://7B:7000/v1"
    assert not hasattr(config, "unknown_key")


def test_from_dict_with_empty_dict():
    """from_dict 空字典应返回默认配置"""
    config = ModelRoutingConfig.from_dict({})
    assert config == ModelRoutingConfig()


def test_from_dict_with_missing_keys():
    """from_dict 缺少部分 key 应使用默认值"""
    config = ModelRoutingConfig.from_dict({"enabled": True})
    assert config.enabled is True
    assert config.small_model_url == ""


def test_from_dict_preserves_all_fields():
    """from_dict 应能正确传递所有字段"""
    d = {
        "enabled": True,
        "small_model_url": "http://7B:7000/v1",
        "large_model_url": "http://38B:7000/v1",
        "small_model_name": "pangu_small",
        "large_model_name": "pangu_large",
        "small_model_api_key": "sk-small",
        "large_model_api_key": "sk-large",
        "small_model_template": "small_tpl",
        "large_model_template": "large_tpl",
    }
    config = ModelRoutingConfig.from_dict(d)
    assert config.enabled is True
    assert config.small_model_url == "http://7B:7000/v1"
    assert config.large_model_url == "http://38B:7000/v1"
    assert config.small_model_name == "pangu_small"
    assert config.large_model_name == "pangu_large"
    assert config.small_model_api_key == "sk-small"
    assert config.large_model_api_key == "sk-large"
    assert config.small_model_template == "small_tpl"
    assert config.large_model_template == "large_tpl"


# ── from_config_module ──


def test_from_config_module_reads_env_attrs():
    """from_config_module 应从 config_module 读取缺失的 env key 对应的属性"""
    mod = SimpleNamespace(
        MODEL_ROUTING_SWITCH=True,
        SMALL_MODEL_URL="http://7B:7000/v1",
        LARGE_MODEL_URL="http://38B:7000/v1",
        SMALL_MODEL_NAME="pangu_small",
        LARGE_MODEL_NAME="pangu_large",
        SMALL_MODEL_API_KEY="sk-small",
        LARGE_MODEL_API_KEY="sk-large",
        SMALL_MODEL_TEMPLATE="small_tpl",
    )
    config = ModelRoutingConfig.from_config_module({}, mod)
    assert config.enabled is True
    assert config.small_model_url == "http://7B:7000/v1"
    assert config.large_model_url == "http://38B:7000/v1"
    assert config.small_model_name == "pangu_small"
    assert config.large_model_name == "pangu_large"
    assert config.small_model_api_key == "sk-small"
    assert config.large_model_api_key == "sk-large"
    assert config.small_model_template == "small_tpl"


def test_from_config_module_overrides_take_precedence():
    """overrides 中的值应优先于 config_module"""
    mod = SimpleNamespace(
        MODEL_ROUTING_SWITCH=False,
        SMALL_MODEL_URL="http://fallback:7000/v1",
    )

    overrides = {
        "MODEL_ROUTING_SWITCH": True,
        "SMALL_MODEL_URL": "http://override:7000/v1",
    }
    config = ModelRoutingConfig.from_config_module(overrides, mod)
    assert config.enabled is True
    assert config.small_model_url == "http://override:7000/v1"


def test_from_config_module_accepts_lowercase_request_switch():
    mod = SimpleNamespace(MODEL_ROUTING_SWITCH=True)
    config = ModelRoutingConfig.from_config_module(
        {"model_routing_switch": False}, mod
    )
    assert config.enabled is False


def test_from_config_module_missing_attrs_use_defaults():
    """config_module 缺少属性时应使用默认值"""
    mod = SimpleNamespace()  # 无任何属性
    config = ModelRoutingConfig.from_config_module({}, mod)
    assert config == ModelRoutingConfig()


def test_from_config_module_partial_overrides():
    """部分 overrides + 部分 config_module 应正确合并"""
    mod = SimpleNamespace(
        MODEL_ROUTING_SWITCH=True,
        SMALL_MODEL_URL="http://7B:7000/v1",
        LARGE_MODEL_URL="http://38B:7000/v1",
    )
    overrides = {"SMALL_MODEL_URL": "http://custom:7000/v1"}
    config = ModelRoutingConfig.from_config_module(overrides, mod)
    assert config.small_model_url == "http://custom:7000/v1"
    assert config.large_model_url == "http://38B:7000/v1"
    assert config.enabled is True
