from unittest.mock import patch, MagicMock

from early_stop.config import EarlyStopConfig


class TestEarlyStopConfigDefaults:
    def test_default_enabled(self):
        cfg = EarlyStopConfig()
        assert cfg.enabled is True

    def test_default_min_iter(self):
        cfg = EarlyStopConfig()
        assert cfg.min_iter == 2


class TestEarlyStopConfigCustom:
    def test_custom_enabled(self):
        cfg = EarlyStopConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_min_iter(self):
        cfg = EarlyStopConfig(min_iter=5)
        assert cfg.min_iter == 5


class TestFromContext:
    @patch("early_stop.config.EarlyStopConfig.enabled", True)
    @patch("early_stop.config.EarlyStopConfig.min_iter", 2)
    def test_no_overrides_no_global_config(self):
        """当无 overrides 且全局 config 模块无对应属性时，使用 dataclass 默认值"""
        mock_config = MagicMock(spec=[])
        with patch.dict("sys.modules", {"config.config": mock_config}):
            cfg = EarlyStopConfig.from_context()
        assert cfg.enabled is True
        assert cfg.min_iter == 2

    def test_overrides_take_priority(self):
        """config_overrides 优先级最高"""
        mock_config = MagicMock(spec=[])
        mock_config.early_stop_enabled = False
        mock_config.early_stop_min_iter = 10
        with patch.dict("sys.modules", {"config.config": mock_config}):
            cfg = EarlyStopConfig.from_context(
                config_overrides={"early_stop_enabled": True, "early_stop_min_iter": 3}
            )
        assert cfg.enabled is True
        assert cfg.min_iter == 3

    def test_global_config_as_fallback(self):
        """无 overrides 时回退到全局 config 模块属性"""
        mock_config = MagicMock(spec=[])
        mock_config.early_stop_enabled = False
        mock_config.early_stop_min_iter = 7
        with patch.dict("sys.modules", {"config.config": mock_config}):
            cfg = EarlyStopConfig.from_context()
        assert cfg.enabled is False
        assert cfg.min_iter == 7

    def test_partial_overrides(self):
        """部分 overrides 只覆盖指定字段，其余回退全局 config"""
        mock_config = MagicMock(spec=[])
        mock_config.early_stop_enabled = False
        mock_config.early_stop_min_iter = 8
        with patch.dict("sys.modules", {"config.config": mock_config}):
            cfg = EarlyStopConfig.from_context(
                config_overrides={"early_stop_min_iter": 5}
            )
        assert cfg.enabled is False  # 来自全局 config
        assert cfg.min_iter == 5    # 来自 overrides

    def test_empty_overrides_dict(self):
        """空 dict 等同于无 overrides"""
        mock_config = MagicMock(spec=[])
        mock_config.early_stop_enabled = True
        mock_config.early_stop_min_iter = 4
        with patch.dict("sys.modules", {"config.config": mock_config}):
            cfg = EarlyStopConfig.from_context(config_overrides={})
        assert cfg.enabled is True
        assert cfg.min_iter == 4