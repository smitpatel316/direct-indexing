"""
OracleStrategy — High-level strategy interface for direct indexing optimization.

Wraps Oracle with:
- Strategy-specific parameters
- TLH opportunity identification
- Trade execution workflow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from .oracle import Oracle, OptimizationResult, Trade


class OptimizationType(Enum):
    """Type of optimization to perform."""
    TAX_AWARE = "tax_aware"  # Tax-aware rebalancing
    DIRECT_INDEX = "direct_index"  # Direct indexing with TLH
    PAIRS_TLH = "pairs_tlh"  # Pairs-based TLH with replacements


@dataclass
class TLHConfig:
    """TLH-specific configuration."""
    enabled: bool = True
    min_loss_threshold: float = 0.015  # 1.5% loss minimum
    min_harvest_value: float = 100.0  # $100 minimum harvest
    wash_window_days: int = 31  # 31-day wash sale window
    max_harvests_per_year: int = 10  # Limit annual harvests


@dataclass
class StrategyConfig:
    """Strategy-wide configuration."""
    min_weight_multiplier: float = 0.5  # Min weight as % of target
    max_weight_multiplier: float = 2.0  # Max weight as % of target
    min_notional: float = 100.0  # Min trade size
    trade_rounding: float = 1.0  # Round quantities to this
    drift_threshold: float = 0.02  # 2% drift triggers rebalance


@dataclass
class TLHOpportunity:
    """A specific TLH opportunity identified by the optimizer."""
    tax_lot_id: str
    identifier: str
    quantity: float
    cost_basis: float
    current_value: float
    loss_amount: float
    loss_percentage: float
    potential_tax_savings: float
    priority: float  # Higher = harvest first


class OracleStrategy:
    """
    High-level strategy for direct indexing optimization.

    Integrates with Oracle optimizer to:
    1. Identify TLH opportunities (lots at a loss)
    2. Run optimization to find best trades
    3. Track wash sale restrictions
    4. Generate executable trade list

    Key difference from rule-based TLH:
    - Optimization considers ALL constraints simultaneously
    - Trades are sized optimally, not just threshold-based
    - Wash sale tracking is built-in at the optimizer level
    """

    # Normalization constants (from Oracle)
    TAX_NORMALIZATION = 800.0
    DRIFT_NORMALIZATION = 100.0
    TRANSACTION_NORMALIZATION = 1200.0

    def __init__(
        self,
        strategy_id: int | None = None,
        tax_lots: pd.DataFrame | None = None,
        target_weights: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        cash: float = 0.0,
        tax_rates: pd.DataFrame | None = None,
        tlh_config: TLHConfig | None = None,
        strategy_config: StrategyConfig | None = None,
        current_date: date | None = None,
    ):
        """
        Initialize OracleStrategy.

        Args:
            strategy_id: Unique identifier for this strategy
            tax_lots: Current tax lots [tax_lot_id, identifier, quantity, cost_basis, date_acquired]
            target_weights: Target weights [identifier, target_weight]
            prices: Current prices [identifier, price]
            cash: Available cash
            tax_rates: Tax rates [gain_type, total_rate]
            tlh_config: TLH configuration
            strategy_config: General strategy configuration
            current_date: Date for optimization
        """
        self.strategy_id = strategy_id or id(self)
        self.tax_lots = tax_lots or pd.DataFrame()
        self.target_weights = target_weights or pd.DataFrame()
        self.prices = prices or pd.DataFrame()
        self.cash = cash

        self.tlh_config = tlh_config or TLHConfig()
        self.strategy_config = strategy_config or StrategyConfig()
        self.current_date = current_date or date.today()

        # Initialize Oracle optimizer
        self.oracle = Oracle(
            current_date=self.current_date,
            tax_rates=tax_rates,
            wash_window_days=self.tlh_config.wash_window_days,
            min_weight_multiplier=self.strategy_config.min_weight_multiplier,
            max_weight_multiplier=self.strategy_config.max_weight_multiplier,
        )

        self._wash_sale_history: list[dict] = []

    def set_oracle(self, oracle: Oracle) -> None:
        """Set the Oracle instance (for cross-strategy wash tracking)."""
        self.oracle = oracle

    def identify_tlh_opportunities(
        self,
        min_loss_threshold: float | None = None,
        min_harvest_value: float | None = None,
    ) -> list[TLHOpportunity]:
        """
        Identify tax-loss harvesting opportunities.

        Args:
            min_loss_threshold: Minimum loss % to consider (default from config)
            min_harvest_value: Minimum $ loss to consider (default from config)

        Returns:
            List of TLHOpportunity sorted by priority (highest first)
        """
        min_loss = min_loss_threshold or self.tlh_config.min_loss_threshold
        min_value = min_harvest_value or self.tlh_config.min_harvest_value

        if self.tax_lots.empty or self.prices.empty:
            return []

        # Compute gain/loss
        gain_loss = self.oracle.compute_gain_loss(self.tax_lots, self.prices)

        opportunities = []

        for _, lot in gain_loss.iterrows():
            loss_pct = lot.get("gain_loss_percentage", 0)
            loss_amt = lot.get("gain_loss", 0)

            # Skip if not a loss or below threshold
            if loss_pct >= 0 or abs(loss_pct) < min_loss:
                continue

            # Skip if loss amount too small
            if abs(loss_amt) < min_value:
                continue

            # Check if ticker is restricted from selling
            if self.oracle.wash_sale.is_restricted_from_selling(lot["identifier"]):
                continue

            # Calculate potential tax savings
            tax_savings = abs(loss_amt) * lot.get("tax_rate", 0.37)

            opportunities.append(TLHOpportunity(
                tax_lot_id=lot["tax_lot_id"],
                identifier=lot["identifier"],
                quantity=lot["quantity"],
                cost_basis=lot["cost_basis"],
                current_value=lot["current_value"],
                loss_amount=abs(loss_amt),
                loss_percentage=abs(loss_pct),
                potential_tax_savings=tax_savings,
                priority=tax_savings,  # Higher savings = higher priority
            ))

        # Sort by priority (highest first)
        opportunities.sort(key=lambda x: x.priority, reverse=True)
        return opportunities

    def run_optimization(
        self,
        min_trade_value: float | None = None,
    ) -> OptimizationResult:
        """
        Run the full optimization to find best trades.

        Args:
            min_trade_value: Minimum trade $ to execute

        Returns:
            OptimizationResult with optimal trades
        """
        min_value = min_trade_value or self.strategy_config.min_notional

        return self.oracle.optimize(
            tax_lots=self.tax_lots,
            target_weights=self.target_weights,
            prices=self.prices,
            cash=self.cash,
            min_trade_value=min_value,
        )

    def record_harvest(
        self,
        ticker: str,
        sell_date: date | None = None,
        loss_amount: float = 0.0,
    ) -> None:
        """
        Record a TLH harvest to enforce wash sale restrictions.

        Args:
            ticker: The ticker harvested at a loss
            sell_date: Date of harvest (default: current_date)
            loss_amount: Realized loss amount
        """
        self.oracle.record_tlh_harvest(
            ticker=ticker,
            sell_date=sell_date,
            loss_amount=loss_amount,
        )
        self._wash_sale_history.append({
            "date": sell_date or self.current_date,
            "ticker": ticker,
            "loss_amount": loss_amount,
            "restriction_ends": (
                (sell_date or self.current_date).isoformat()
            ),
        })

    def get_wash_sale_status(self) -> pd.DataFrame:
        """Get current wash sale restrictions as DataFrame."""
        return self.oracle.wash_sale.to_dataframe()

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        if self.prices.empty or self.tax_lots.empty:
            return self.cash
        total = self.tax_lots.merge(self.prices, on="identifier", how="left")
        return self.cash + (total["quantity"] * total["price"]).sum()

    def get_current_weights(self) -> pd.DataFrame:
        """Get current portfolio weights."""
        if self.tax_lots.empty or self.prices.empty:
            return pd.DataFrame(columns=["identifier", "current_weight", "target_weight"])

        total_value = self.get_portfolio_value()
        if total_value == 0:
            return pd.DataFrame(columns=["identifier", "current_weight", "target_weight"])

        merged = self.tax_lots.merge(self.prices, on="identifier", how="left")
        merged["value"] = merged["quantity"] * merged["price"]
        weights = merged.groupby("identifier")["value"].sum().reset_index()
        weights["current_weight"] = weights["value"] / total_value

        target_map = dict(zip(
            self.target_weights["identifier"],
            self.target_weights["target_weight"]
        ))
        weights["target_weight"] = weights["identifier"].map(target_map).fillna(0)

        return weights[["identifier", "current_weight", "target_weight"]]

    def generate_rebalance_trades(
        self,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
        cash: float,
    ) -> pd.DataFrame:
        """
        Generate rebalancing trades to reach target weights.

        Args:
            target_weights: Target weights [identifier, target_weight]
            prices: Current prices [identifier, price]
            cash: Available cash

        Returns:
            DataFrame of trades to execute
        """
        self.target_weights = target_weights
        self.prices = prices
        self.cash = cash

        result = self.run_optimization()
        return result.trades
