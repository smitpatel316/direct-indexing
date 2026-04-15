"""
Portfolio Manager — bridges Alpaca positions with ETF target weights.

Coordinates:
- ETF replica (target constituents and weights)
- Live positions from Alpaca
- Drift calculation against ETF targets
- Rebalancing recommendations
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca_trading_client import AlpacaClient


@dataclass
class PortfolioPosition:
    """A position with both target (ETF) and actual (Alpaca) data."""
    ticker: str
    # Target weights (from ETF)
    target_weight: float
    target_value: float
    target_shares: float
    # Actual holdings (from Alpaca)
    current_shares: float
    current_price: float
    current_value: float
    current_weight: float
    # Drift
    drift_value: float
    drift_percent: float
    shares_to_trade: float


@dataclass
class PortfolioDriftReport:
    """Full drift report comparing Alpaca positions to ETF targets."""
    timestamp: datetime
    total_value: float
    target_etf: str
    positions: list[PortfolioPosition]
    max_drift_percent: float
    total_drift_value: float
    needs_rebalance: bool


class PortfolioManager:
    """
    Manages a direct indexing portfolio against an ETF target.

    Workflow:
    1. Load ETF target constituents (from file or API)
    2. Sync current positions from Alpaca
    3. Calculate drift from ETF targets
    4. Generate rebalancing recommendations
    """

    def __init__(
        self,
        target_etf: str,
        drift_threshold: float = 2.0,
        data_dir: Path = Path("data"),
    ):
        self.target_etf = target_etf
        self.drift_threshold = drift_threshold
        self.data_dir = data_dir
        self._target_constituents: dict[str, dict] = {}  # ticker → {weight, ...}

    # -------------------------------------------------------------------------
    # Target constituents (from ETF)
    # -------------------------------------------------------------------------

    def set_targets_from_csv(self, csv_path: Path) -> int:
        """
        Load ETF target weights from a CSV file.

        CSV format: ticker,weight
        Example:
            AAPL,7.25
            MSFT,5.50

        Returns the number of constituents loaded.
        """
        import csv

        count = 0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row["ticker"].strip().upper()
                weight = float(row["weight"])
                self._target_constituents[ticker] = {
                    "weight": weight,
                }
                count += 1
        return count

    def set_targets_from_dict(self, constituents: dict[str, float]) -> None:
        """
        Set target constituents from a dict of {ticker: weight_percent}.

        Example:
            manager.set_targets({"AAPL": 7.25, "MSFT": 5.50})
        """
        self._target_constituents = {
            ticker.upper(): {"weight": weight}
            for ticker, weight in constituents.items()
        }

    def get_target_constituents(self) -> dict[str, float]:
        """Return a copy of target constituents as {ticker: weight_percent}."""
        return {
            ticker: data["weight"]
            for ticker, data in self._target_constituents.items()
        }

    # -------------------------------------------------------------------------
    # Sync with Alpaca
    # -------------------------------------------------------------------------

    def sync_from_alpaca(self, client: "AlpacaClient") -> list[str]:
        """
        Fetch current positions from Alpaca and update internal state.

        Returns list of tickers that are currently held.
        """
        positions = client.get_positions()
        held = []
        for pos in positions:
            ticker = pos.symbol.upper()
            held.append(ticker)
        return held

    # -------------------------------------------------------------------------
    # Drift calculation
    # -------------------------------------------------------------------------

    def calculate_drift(
        self,
        positions: list,  # list of Position dataclasses from AlpacaClient
    ) -> PortfolioDriftReport:
        """
        Calculate drift between actual Alpaca positions and ETF target weights.

        Args:
            positions: List of Position dataclasses from AlpacaClient

        Returns:
            PortfolioDriftReport with per-ticker drift analysis
        """
        # Build lookup of current positions
        current_by_ticker: dict[str, dict] = {}
        total_value = 0.0
        for pos in positions:
            ticker = pos.symbol.upper()
            current_by_ticker[ticker] = {
                "shares": pos.qty,
                "price": pos.current_price,
                "value": pos.market_value,
            }
            total_value += pos.market_value

        # Calculate drift for each target constituent
        portfolio_positions: list[PortfolioPosition] = []
        max_drift = 0.0
        total_drift_value = 0.0

        for ticker, target in self._target_constituents.items():
            target_weight = target["weight"]
            target_value = (target_weight / 100) * total_value
            current = current_by_ticker.get(
                ticker,
                {"shares": 0.0, "price": 0.0, "value": 0.0},
            )

            current_shares = current["shares"]
            current_price = current["price"]
            current_value = current["value"]
            current_weight = (
                (current_value / total_value * 100)
                if total_value > 0
                else 0.0
            )

            drift_value = current_value - target_value
            drift_percent = current_weight - target_weight
            max_drift = max(max_drift, abs(drift_percent))
            total_drift_value += abs(drift_value)

            # Calculate shares to trade
            if current_price > 0:
                shares_to_trade = (target_value - current_value) / current_price
            else:
                shares_to_trade = 0.0

            portfolio_positions.append(PortfolioPosition(
                ticker=ticker,
                target_weight=target_weight,
                target_value=target_value,
                target_shares=max(shares_to_trade, 0),  # only buy side
                current_shares=current_shares,
                current_price=current_price,
                current_value=current_value,
                current_weight=current_weight,
                drift_value=drift_value,
                drift_percent=drift_percent,
                shares_to_trade=abs(shares_to_trade),
            ))

        # Also flag positions held that aren't in ETF target
        for ticker, current in current_by_ticker.items():
            if ticker not in self._target_constituents:
                current_weight = (
                    (current["value"] / total_value * 100)
                    if total_value > 0
                    else 0.0
                )
                drift_percent = current_weight  # Not in ETF = full weight is "drift"
                max_drift = max(max_drift, abs(drift_percent))
                total_drift_value += abs(current["value"])

                portfolio_positions.append(PortfolioPosition(
                    ticker=ticker,
                    target_weight=0.0,
                    target_value=0.0,
                    target_shares=0.0,
                    current_shares=current["shares"],
                    current_price=current["price"],
                    current_value=current["value"],
                    current_weight=current_weight,
                    drift_value=current["value"],
                    drift_percent=current_weight,
                    shares_to_trade=current["shares"],  # sell all
                ))

        return PortfolioDriftReport(
            timestamp=datetime.now(),
            total_value=total_value,
            target_etf=self.target_etf,
            positions=portfolio_positions,
            max_drift_percent=max_drift,
            total_drift_value=total_drift_value,
            needs_rebalance=max_drift > self.drift_threshold,
        )

    def rebalance_recommendations(
        self,
        report: PortfolioDriftReport,
    ) -> list[dict]:
        """
        Generate rebalancing recommendations from a drift report.

        Returns list of dicts with trade instructions.
        """
        recommendations = []
        for pos in report.positions:
            if pos.shares_to_trade < 1:  # Skip if < 1 share to trade
                continue

            action = "buy" if pos.drift_value < 0 else "sell"
            if abs(pos.drift_percent) < self.drift_threshold:
                continue  # Within threshold, no action needed

            recommendations.append({
                "ticker": pos.ticker,
                "action": action,
                "shares": round(pos.shares_to_trade, 0),
                "estimated_value": round(pos.shares_to_trade * pos.current_price, 2),
                "drift_percent": round(pos.drift_percent, 2),
                "reason": (
                    f"{action.upper()} to reduce drift from"
                    f" {pos.target_weight}% → {pos.current_weight}% target"
                ),
            })

        return recommendations
