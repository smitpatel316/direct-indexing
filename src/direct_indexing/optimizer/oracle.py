"""
Oracle — Main portfolio optimization engine.

This is the core optimizer that coordinates:
- Tax-aware objective function
- Weight bound constraints
- Wash sale restrictions
- MILP solving via PuLP + CBC
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

from .solver import solve_optimization_problem
from .decision_vars import create_decision_variables
from .objectives.tax import calculate_tax_impact
from .objectives.drift import calculate_drift_impact
from .objectives.transaction import calculate_transaction_cost
from .constraints.wash_sale import WashSaleConstraints
from .constraints.weight import WeightConstraints


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""
    status: int
    objective_value: float | None
    trades: pd.DataFrame
    tax_impact: float
    drift_impact: float
    transaction_cost: float
    solve_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False


@dataclass
class Trade:
    """A single trade recommendation."""
    ticker: str
    action: str  # "BUY" or "SELL"
    quantity: float
    price: float
    value: float
    tax_lot_id: Optional[str] = None
    reason: str = ""


class Oracle:
    """
    MILP-based portfolio optimizer for direct indexing.

    Uses Oracle-style optimization to find the best set of trades that:
    1. Minimizes tax cost (sells losers, avoids selling winners)
    2. Minimizes drift from target weights
    3. Respects wash sale restrictions (31-day window)
    4. Keeps positions within weight bounds

    Unlike rule-based systems (threshold → harvest), this finds the
    globally optimal set of trades given the full problem structure.
    """

    def __init__(
        self,
        current_date: date | None = None,
        tax_rates: pd.DataFrame | None = None,
        wash_window_days: int = 31,
        min_weight_multiplier: float = 0.5,
        max_weight_multiplier: float = 2.0,
        time_limit: int = 60,
        allow_warm_start: bool = True,
    ):
        """
        Initialize Oracle optimizer.

        Args:
            current_date: Date for optimization (default: today)
            tax_rates: DataFrame with [gain_type, total_rate]
                gain_type values: "short_term", "long_term", "interest"
                Example: pd.DataFrame({"gain_type": ["short_term", "long_term"],
                                       "total_rate": [0.37, 0.20]})
            wash_window_days: Days to restrict after TLH sale (default 31)
            min_weight_multiplier: Min weight as fraction of target (default 0.5)
            max_weight_multiplier: Max weight as fraction of target (default 2.0)
            time_limit: Max solve time in seconds (default 60)
            allow_warm_start: Use previous solution as starting point (default True)
        """
        self.current_date = current_date or date.today()
        self.tax_rates = tax_rates if tax_rates is not None else self._default_tax_rates()
        self.wash_window_days = wash_window_days
        self.time_limit = time_limit

        self.wash_sale = WashSaleConstraints(
            current_date=self.current_date,
            wash_window_days=wash_window_days,
        )
        self.weight_constraints = WeightConstraints(
            min_weight_multiplier=min_weight_multiplier,
            max_weight_multiplier=max_weight_multiplier,
        )

    @staticmethod
    def _default_tax_rates() -> pd.DataFrame:
        """Default 2024 tax rates for individuals."""
        return pd.DataFrame({
            "gain_type": ["short_term", "long_term", "interest"],
            "total_rate": [0.37, 0.20, 0.37],
        })

    def compute_gain_loss(
        self,
        tax_lots: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute unrealized gain/loss for each tax lot.

        Args:
            tax_lots: DataFrame with [tax_lot_id, identifier, quantity, cost_basis, date_acquired]
            prices: DataFrame with [identifier, price]

        Returns:
            DataFrame with gain/loss columns added:
            [tax_lot_id, identifier, quantity, cost_basis, current_value,
             gain_loss, gain_loss_percentage, per_share_tax_liability, gain_type]
        """
        gl = tax_lots.merge(prices, on="identifier", how="left")
        gl["current_value"] = gl["quantity"] * gl["price"]
        gl["gain_loss"] = gl["current_value"] - gl["cost_basis"]
        gl["gain_loss_percentage"] = gl["gain_loss"] / gl["cost_basis"]

        # Determine gain type based on holding period
        today = pd.Timestamp(self.current_date)
        gl["date_acquired"] = pd.to_datetime(gl["date_acquired"])
        gl["holding_days"] = (today - gl["date_acquired"]).dt.days

        # Short term: < 365 days, Long term: >= 365 days
        gl["gain_type"] = gl["holding_days"].apply(
            lambda d: "long_term" if d >= 365 else "short_term"
        )

        # Lookup tax rate per share
        rate_map = dict(zip(self.tax_rates["gain_type"], self.tax_rates["total_rate"]))
        gl["tax_rate"] = gl["gain_type"].map(rate_map).fillna(0.37)

        # Per-share tax liability: negative for losses, positive for gains
        # For losses: liability is negative (tax savings)
        gl["per_share_tax_liability"] = gl["gain_loss"] / gl["quantity"] * gl["tax_rate"]

        return gl

    def optimize(
        self,
        tax_lots: pd.DataFrame,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
        cash: float,
        min_trade_value: float = 100.0,
        drift_threshold: float | None = None,
    ) -> OptimizationResult:
        """
        Run portfolio optimization.

        Args:
            tax_lots: DataFrame with [tax_lot_id, identifier, quantity, cost_basis, date_acquired]
            target_weights: DataFrame with [identifier, target_weight]
            prices: DataFrame with [identifier, price]
            cash: Available cash for buying
            min_trade_value: Minimum trade value to execute ($100 default)
            drift_threshold: Optional max drift before rebalancing (not implemented)

        Returns:
            OptimizationResult with optimal trades
        """
        import time
        start = time.time()

        # Compute gain/loss
        gain_loss = self.compute_gain_loss(tax_lots, prices)

        # Total portfolio value
        total_value = cash + gain_loss["current_value"].sum()

        # Valid tickers (from target weights)
        valid_tickers = target_weights["identifier"].tolist()

        # Create problem
        prob = pulp.LpProblem("Direct_Indexing", pulp.LpMinimize)

        # Decision variables
        sells, buys, cash_used = create_decision_variables(
            prob, tax_lots, valid_tickers, prices, cash
        )

        # Objective: minimize tax + drift + transaction cost
        tax_impact, baseline_tax = calculate_tax_impact(
            prob, sells, gain_loss, total_value,
            enforce_wash_sale_prevention=True,
        )
        drift_impact, baseline_drift = calculate_drift_impact(
            prob, sells, buys, gain_loss, target_weights, prices, total_value,
        )
        txn_cost, _ = calculate_transaction_cost(
            prob, sells, buys, prices, total_value,
        )

        prob += tax_impact + drift_impact + txn_cost

        # Add wash sale constraints
        self.wash_sale.add_constraints_to_problem(prob, buys, sells, tax_lots)

        # Add weight constraints
        self.weight_constraints.add_constraints(
            prob, sells, buys, tax_lots, target_weights, prices, total_value,
        )

        # Cash constraint: can't spend more than available
        total_buy_value = pulp.lpSum(
            buys[t] * prices[prices["identifier"] == t]["price"].iloc[0]
            for t in valid_tickers if t in buys
        )
        prob += total_buy_value <= cash, "cash_constraint"

        # Solve
        status, objective_value = solve_optimization_problem(
            prob,
            time_limit=self.time_limit,
        )

        solve_time_ms = (time.time() - start) * 1000

        # Extract trades
        trades = self._extract_trades(sells, buys, prices, tax_lots, min_trade_value)

        return OptimizationResult(
            status=status,
            objective_value=objective_value,
            trades=trades,
            tax_impact=float(tax_impact.value()) if tax_impact.value() else 0.0,
            drift_impact=float(drift_impact.value()) if drift_impact.value() else 0.0,
            transaction_cost=float(txn_cost.value()) if txn_cost.value() else 0.0,
            solve_time_ms=solve_time_ms,
            success=status == pulp.LpStatusOptimal,
        )

    def _extract_trades(
        self,
        sells: dict,
        buys: dict,
        prices: pd.DataFrame,
        tax_lots: pd.DataFrame,
        min_trade_value: float,
    ) -> pd.DataFrame:
        """Extract trade list from solved problem."""
        trade_list = []
        price_map = dict(zip(prices["identifier"], prices["price"]))

        # Build lot_id -> ticker mapping
        lot_to_ticker = dict(zip(tax_lots["tax_lot_id"], tax_lots["identifier"]))

        # Extract sells
        for lot_id, var in sells.items():
            qty = var.value()
            if qty is not None and qty > 0.001:
                ticker = lot_to_ticker.get(lot_id, "UNKNOWN")
                price = price_map.get(ticker, 0)
                value = qty * price
                if value >= min_trade_value:
                    trade_list.append(Trade(
                        ticker=ticker,
                        action="SELL",
                        quantity=qty,
                        price=price,
                        value=value,
                        tax_lot_id=lot_id,
                        reason="tlh_harvest",
                    ))

        # Extract buys
        for ticker, var in buys.items():
            qty = var.value()
            if qty is not None and qty > 0.001:
                price = price_map.get(ticker, 0)
                value = qty * price
                if value >= min_trade_value:
                    trade_list.append(Trade(
                        ticker=ticker,
                        action="BUY",
                        quantity=qty,
                        price=price,
                        value=value,
                        reason="rebalance",
                    ))

        return pd.DataFrame([{
            "ticker": t.ticker,
            "action": t.action,
            "quantity": t.quantity,
            "price": t.price,
            "value": t.value,
            "tax_lot_id": t.tax_lot_id,
            "reason": t.reason,
        } for t in trade_list])

    def record_tlh_harvest(
        self,
        ticker: str,
        sell_date: date | None = None,
        loss_amount: float = 0.0,
    ) -> None:
        """
        Record a TLH harvest so wash sale restrictions are enforced.

        After harvesting ticker X at a loss:
        - Cannot buy X for 31 days
        - Cash from sale sits idle (no replacement buy in same ticker)

        Args:
            ticker: The ticker harvested at a loss
            sell_date: Date of the sale (default: current_date)
            loss_amount: The realized loss amount (for tracking)
        """
        sell_date = sell_date or self.current_date
        self.wash_sale.add_sell_restriction(
            ticker=ticker,
            sell_date=sell_date,
            restriction_end_date=sell_date + timedelta(days=self.wash_window_days),
        )
