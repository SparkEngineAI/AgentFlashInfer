"""Tests for deepresearch_config.py environment variable parsing."""

import os


def _reload_config():
    """Force re-import of config to pick up env var changes."""
    import importlib
    import examples.deepresearch.config.deepresearch_config as cfg
    importlib.reload(cfg)
    return cfg


class TestDefaults:
    def test_vllm_ports_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_PORTS", raising=False)
        cfg = _reload_config()
        assert cfg.VLLM_PORTS == [6001]

    def test_max_llm_call_default(self, monkeypatch):
        monkeypatch.delenv("MAX_LLM_CALL_PER_RUN", raising=False)
        cfg = _reload_config()
        assert cfg.MAX_LLM_CALL_PER_RUN == 100

    def test_max_timeout_minutes_default(self, monkeypatch):
        monkeypatch.delenv("MAX_TIMEOUT_MINUTES", raising=False)
        cfg = _reload_config()
        assert cfg.MAX_TIMEOUT_MINUTES == 150

    def test_generation_temperature_default(self, monkeypatch):
        monkeypatch.delenv("GENERATION_TEMPERATURE", raising=False)
        cfg = _reload_config()
        assert cfg.GENERATION_TEMPERATURE == 0.6

    def test_dr_server_port_default(self, monkeypatch):
        monkeypatch.delenv("DR_SERVER_PORT", raising=False)
        cfg = _reload_config()
        assert cfg.DR_SERVER_PORT == 8290


class TestEnvOverride:
    def test_vllm_ports_multi(self, monkeypatch):
        monkeypatch.setenv("VLLM_PORTS", "7001,7002,7003")
        cfg = _reload_config()
        assert cfg.VLLM_PORTS == [7001, 7002, 7003]

    def test_max_llm_call_override(self, monkeypatch):
        monkeypatch.setenv("MAX_LLM_CALL_PER_RUN", "50")
        cfg = _reload_config()
        assert cfg.MAX_LLM_CALL_PER_RUN == 50

    def test_generation_temperature_override(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TEMPERATURE", "0.8")
        cfg = _reload_config()
        assert cfg.GENERATION_TEMPERATURE == 0.8

    def test_sandbox_fusion_endpoints_multi(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_FUSION_ENDPOINT", "http://h1:8080,http://h2:8080")
        cfg = _reload_config()
        assert cfg.SANDBOX_FUSION_ENDPOINTS == ["http://h1:8080", "http://h2:8080"]


class TestTypeCoercion:
    def test_int_coercion(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKEN_LIMIT", "50000")
        cfg = _reload_config()
        assert cfg.MAX_TOKEN_LIMIT == 50000
        assert isinstance(cfg.MAX_TOKEN_LIMIT, int)

    def test_float_coercion(self, monkeypatch):
        monkeypatch.setenv("GENERATION_TOP_P", "0.9")
        cfg = _reload_config()
        assert cfg.GENERATION_TOP_P == 0.9
        assert isinstance(cfg.GENERATION_TOP_P, float)

    def test_list_from_csv(self, monkeypatch):
        monkeypatch.setenv("VLLM_PORTS", "8001,8002")
        cfg = _reload_config()
        assert isinstance(cfg.VLLM_PORTS, list)
        assert all(isinstance(p, int) for p in cfg.VLLM_PORTS)


class TestEdgeCases:
    def test_vllm_ports_empty_entries_filtered(self, monkeypatch):
        monkeypatch.setenv("VLLM_PORTS", "7001,,7003,")
        cfg = _reload_config()
        assert cfg.VLLM_PORTS == [7001, 7003]

    def test_sandbox_fusion_empty_string(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_FUSION_ENDPOINT", "")
        cfg = _reload_config()
        assert cfg.SANDBOX_FUSION_ENDPOINTS == []

    def test_use_idp_false(self, monkeypatch):
        monkeypatch.setenv("USE_IDP", "false")
        cfg = _reload_config()
        assert cfg.USE_IDP is False

    def test_use_idp_true(self, monkeypatch):
        monkeypatch.setenv("USE_IDP", "True")
        cfg = _reload_config()
        assert cfg.USE_IDP is True
