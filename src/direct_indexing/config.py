"""
Configuration management for Direct Indexing.
Uses modern Python patterns: dataclasses, type hints, YAML config.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]
import pandas as pd


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
    """Tax-Loss Harvesting configuration for direct indexing.

    Note: No ETF wrapper pattern. Harvested losses rotate to cash,
    then repurchase original ticker after 31-day wash sale window.
    """
    enabled: bool = True
    # Loss thresholds for optimization
    loss_threshold_percent: float = 1.5  # 1.5% loss minimum to consider harvesting
    min_loss_amount: float = 100.0  # $100 minimum loss to trigger TLH
    # Gain harvesting: sell positions with large unrealized gains
    # to realize at favorable LTCG rates and reset cost basis
    max_gain_to_sell: float = 0.0  # 0 = disabled; percent gain threshold
    min_gain_amount: float = 1000.0  # Minimum $ gain to trigger gain harvest
    max_harvests_per_year: int = 10
    frequency: str = "daily"  # daily, weekly, monthly
    # Wash sale window (31 days = safe beyond IRS 30-day rule)
    wash_sale_window_days: int = 31
    wash_sale_enabled: bool = True
    carryforward_enabled: bool = True
    ltcg_rate: float = 0.20  # Long-term capital gains rate
    stcg_rate: float = 0.37  # Short-term capital gains rate
    swap_etfs: list[str] = None  # ETFs to swap into after harvest
    # Optimizer-specific settings
    min_weight_multiplier: float = 0.5  # Min position weight as % of target
    max_weight_multiplier: float = 2.0  # Max position weight as % of target
    min_notional: float = 100.0  # Minimum trade size in dollars
    solve_time_limit: int = 60  # Max seconds for MILP solver


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
class TaxRatesConfig:
    """Tax rates configuration for direct indexing.
    
    Default 2024 rates for individuals.
    """
    short_term_rate: float = 0.37  # Ordinary income (held < 1 year)
    long_term_rate: float = 0.20   # Capital gains (held >= 1 year)
    interest_rate: float = 0.37    # Interest income

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for optimizer."""
        return pd.DataFrame({
            "gain_type": ["short_term", "long_term", "interest"],
            "total_rate": [self.short_term_rate, self.long_term_rate, self.interest_rate],
        })


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
    tax_rates: TaxRatesConfig = field(default_factory=TaxRatesConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
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
        tax_rates_data = data.get("tax_rates", {})

        return cls(
            alpaca=(
                AlpacaConfig(**alpaca_data) if alpaca_data else AlpacaConfig()
            ),
            tlh=TLHConfig(**tlh_data) if tlh_data else TLHConfig(),
            rebalance=(
                RebalanceConfig(**rebalance_data)
                if rebalance_data
                else RebalanceConfig()
            ),
            portfolio=(
                PortfolioConfig(**portfolio_data)
                if portfolio_data
                else PortfolioConfig()
            ),
            dashboard=(
                DashboardConfig(**dashboard_data)
                if dashboard_data
                else DashboardConfig()
            ),
            tax_rates=(
                TaxRatesConfig(**tax_rates_data)
                if tax_rates_data
                else TaxRatesConfig()
            ),
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

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or Path("config.yaml")
        self._config: AppConfig | None = None

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
        assert self._config is not None
        return self._config

    def _validate(self) -> None:
        """Validate configuration."""
        assert self._config is not None
        if not self._config.alpaca.api_key:
            raise ValueError("ALPACA_API_KEY is required")
        if not self._config.alpaca.api_secret:
            raise ValueError("ALPACA_API_SECRET is required")

    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            self.load()
        assert self._config is not None
        return self._config


# Global config instance
_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the current configuration."""
    return _config_manager.load()


def reload_config(config_path: Path | None = None) -> AppConfig:
    """Reload configuration from file."""
    if config_path:
        _config_manager.config_path = config_path
    return _config_manager.load()
