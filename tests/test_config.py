"""
Tests for configuration management.
TDD approach: tests define expected behavior.
"""

import pytest
import os
import tempfile
from pathlib import Path
from dataclasses import asdict

from src.direct_indexing.config import (
    AppConfig, AlpacaConfig, TLHConfig, 
    RebalanceConfig, PortfolioConfig, DashboardConfig,
    ConfigManager
)


class TestAlpacaConfig:
    """Alpaca configuration tests."""

    def test_default_values(self):
        cfg = AlpacaConfig()
        assert cfg.api_key == ""
        assert cfg.api_secret == ""
        assert cfg.paper_trading is True
        assert "paper-api" in cfg.base_url

    def test_custom_values(self):
        cfg = AlpacaConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper_trading=False
        )
        assert cfg.api_key == "test_key"
        assert cfg.api_secret == "test_secret"
        assert cfg.paper_trading is False

    def test_base_url_defaults_to_paper(self):
        cfg = AlpacaConfig()
        assert "paper" in cfg.base_url


class TestTLHConfig:
    """TLH configuration tests."""

    def test_default_loss_threshold(self):
        cfg = TLHConfig()
        assert cfg.loss_threshold_percent == 1.5  # Optimization-based threshold
        assert cfg.min_loss_amount == 100.0
        assert cfg.enabled is True

    def test_wash_sale_window_days_default(self):
        cfg = TLHConfig()
        assert cfg.wash_sale_window_days == 31  # 31 days for direct indexing
        assert cfg.wash_sale_enabled is True

    def test_no_swap_etfs(self):
        cfg = TLHConfig()
        assert not hasattr(cfg, 'swap_etfs')  # No ETF wrapper pattern

    def test_optimizer_settings(self):
        cfg = TLHConfig()
        assert cfg.min_weight_multiplier == 0.5
        assert cfg.max_weight_multiplier == 2.0
        assert cfg.min_notional == 100.0
        assert cfg.solve_time_limit == 60

    def test_wash_sale_enabled_by_default(self):
        cfg = TLHConfig()
        assert cfg.wash_sale_enabled is True
        assert cfg.carryforward_enabled is True

    def test_frequency_validation(self):
        cfg = TLHConfig(frequency="weekly")
        assert cfg.frequency in ["daily", "weekly", "monthly"]


class TestAppConfig:
    """Main app config tests."""

    def test_defaults_create_valid_config(self):
        cfg = AppConfig()
        assert cfg.alpaca is not None
        assert cfg.tlh is not None
        assert cfg.rebalance is not None
        assert cfg.portfolio is not None

    def test_yaml_roundtrip(self, tmp_path):
        # Create a config file
        config_file = tmp_path / "config.yaml"
        
        cfg = AppConfig(
            alpaca=AlpacaConfig(api_key="my_key", api_secret="my_secret"),
            tlh=TLHConfig(loss_threshold_percent=7.0)
        )
        
        cfg.to_yaml(config_file)
        assert config_file.exists()
        
        # Load it back
        loaded = AppConfig.from_yaml(config_file)
        assert loaded.alpaca.api_key == "my_key"
        assert loaded.tlh.loss_threshold_percent == 7.0

    def test_env_override(self):
        # Set environment variables
        os.environ["ALPACA_API_KEY"] = "env_key"
        os.environ["ALPACA_API_SECRET"] = "env_secret"
        os.environ["TLH_THRESHOLD"] = "8.5"
        
        cfg = AppConfig.from_env()
        
        assert cfg.alpaca.api_key == "env_key"
        assert cfg.alpaca.api_secret == "env_secret"
        assert cfg.tlh.loss_threshold_percent == 8.5
        
        # Cleanup
        del os.environ["ALPACA_API_KEY"]
        del os.environ["ALPACA_API_SECRET"]
        del os.environ["TLH_THRESHOLD"]

    def test_yaml_file_not_found_with_no_env_vars(self):
        # When no file exists and no env vars set, use defaults
        cfg = ConfigManager(Path("/nonexistent/config.yaml")).load()
        assert cfg.alpaca.api_key == ""
        assert cfg.alpaca.api_secret == ""
        assert cfg.tlh.enabled is True  # Default

    def test_yaml_file_not_found_with_env_vars(self):
        # When no file exists but env vars are set, use them
        os.environ["ALPACA_API_KEY"] = "env_key"
        os.environ["ALPACA_API_SECRET"] = "env_secret"
        try:
            cfg = ConfigManager(Path("/nonexistent/config.yaml")).load()
            assert cfg.alpaca.api_key == "env_key"
            assert cfg.alpaca.api_secret == "env_secret"
        finally:
            del os.environ["ALPACA_API_KEY"]
            del os.environ["ALPACA_API_SECRET"]


class TestConfigManager:
    """ConfigManager tests."""

    def test_load_existing_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
alpaca:
  api_key: "file_key"
  api_secret: "file_secret"
tlh:
  enabled: false
""")
        
        manager = ConfigManager(config_file)
        cfg = manager.load()
        
        assert cfg.alpaca.api_key == "file_key"
        assert cfg.tlh.enabled is False

    def test_missing_api_key_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
alpaca:
  api_key: ""
  api_secret: "secret"
""")
        
        manager = ConfigManager(config_file)
        
        with pytest.raises(ValueError, match="API_KEY"):
            manager.load()

    def test_missing_api_secret_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
alpaca:
  api_key: "key"
  api_secret: ""
""")
        
        manager = ConfigManager(config_file)
        
        with pytest.raises(ValueError, match="API_SECRET"):
            manager.load()


class TestConfigDataclasses:
    """Test all config dataclasses serialize correctly."""

    def test_all_configs_serialize_to_dict(self):
        cfg = AppConfig()
        asdict(cfg)  # Should not raise

    def test_all_configs_from_dict(self):
        data = {
            "alpaca": {"api_key": "key", "api_secret": "secret"},
            "tlh": {"enabled": True, "loss_threshold_percent": 5.0},
            "rebalance": {"enabled": True, "threshold_percent": 2.0},
            "portfolio": {"target_etf": "SPY"},
            "dashboard": {"port": 8000}
        }
        
        cfg = AppConfig._from_dict(data)
        assert cfg.alpaca.api_key == "key"
        assert cfg.tlh.enabled is True
        assert cfg.portfolio.target_etf == "SPY"