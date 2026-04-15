"""
Configuration management for Direct Indexing.
Uses modern Python patterns: dataclasses, type hints, YAML config.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
import yaml
import os


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    paper_trading: bool = True


@dataclass
class TLHConfig:
    """Tax-Loss Harvesting configuration."""
    enabled: bool = True
    loss_threshold_percent: float = 5.0
    min_loss_amount: float = 100.0
    max_harvests_per_year: int = 10
    frequency: str = "daily"  # daily, weekly, monthly
    swap_etfs: List[str] = field(default_factory=lambda: ["VOO", "SPY", "IVV"])
    wash_sale_enabled: bool = True
    carryforward_enabled: bool = True


@dataclass
class RebalanceConfig:
    """Portfolio rebalancing configuration."""
    enabled: bool = True
    threshold_percent: float = 2.0  # 2% drift triggers rebalance
    frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class PortfolioConfig:
    """Portfolio settings."""
    target_etf: str = "SPY"
    etf_constituents_url: str = ""
    min_position_value: float = 1.0  # Minimum $1 per position


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    enable: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    tlh: TLHConfig = field(default_factory=TLHConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "AppConfig":
        """Recursively build config from dict."""
        if data is None:
            return cls()
        
        alpaca_data = data.get("alpaca", {})
        tlh_data = data.get("tlh", {})
        rebalance_data = data.get("rebalance", {})
        portfolio_data = data.get("portfolio", {})
        dashboard_data = data.get("dashboard", {})
        
        return cls(
            alpaca=AlpacaConfig(**alpaca_data) if alpaca_data else AlpacaConfig(),
            tlh=TLHConfig(**tlh_data) if tlh_data else TLHConfig(),
            rebalance=RebalanceConfig(**rebalance_data) if rebalance_data else RebalanceConfig(),
            portfolio=PortfolioConfig(**portfolio_data) if portfolio_data else PortfolioConfig(),
            dashboard=DashboardConfig(**dashboard_data) if dashboard_data else DashboardConfig(),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            alpaca=AlpacaConfig(
                api_key=os.getenv("ALPACA_API_KEY", ""),
                api_secret=os.getenv("ALPACA_API_SECRET", ""),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                paper_trading=os.getenv("ALPACA_PAPER", "true").lower() == "true",
            ),
            tlh=TLHConfig(
                enabled=os.getenv("TLH_ENABLED", "true").lower() == "true",
                loss_threshold_percent=float(os.getenv("TLH_THRESHOLD", "5.0")),
                min_loss_amount=float(os.getenv("TLH_MIN_LOSS", "100.0")),
                frequency=os.getenv("TLH_FREQUENCY", "daily"),
            ),
        )


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self._config: Optional[AppConfig] = None

    def load(self) -> AppConfig:
        """Load configuration from file or environment."""
        if self.config_path.exists():
            self._config = AppConfig.from_yaml(self.config_path)
            self._validate()  # File provided = explicit config, validate it
        else:
            # No file: try env vars, else use safe defaults (no validation)
            self._config = AppConfig.from_env()
            if self._config.alpaca.api_key or self._config.alpaca.api_secret:
                self._validate()  # Only validate if env vars set
        return self._config

    def _validate(self) -> None:
        """Validate configuration."""
        if not self._config.alpaca.api_key:
            raise ValueError("ALPACA_API_KEY is required")
        if not self._config.alpaca.api_secret:
            raise ValueError("ALPACA_API_SECRET is required")

    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self.load()
        return self._config


# Global config instance
_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the current configuration."""
    return _config_manager.load()


def reload_config(config_path: Optional[Path] = None) -> AppConfig:
    """Reload configuration from file."""
    if config_path:
        _config_manager.config_path = config_path
    return _config_manager.load()