"""
Portfolio Replicator - ETF constituent tracking and portfolio construction.

Manages a portfolio that replicates an ETF's holdings, tracks drift,
and calculates rebalancing trades.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class PositionSide(Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Constituent:
    """A single ETF constituent holding."""
    ticker: str
    shares: float
    weight: float  # Target weight as percentage (e.g., 5.5 for 5.5%)
    price: float

    @property
    def market_value(self) -> float:
        """Current market value of this holding."""
        return self.shares * self.price

    def target_value(self, portfolio_value: float) -> float:
        """Target market value given portfolio value and target weight."""
        return self.weight / 100 * portfolio_value

    def target_shares(self, portfolio_value: float) -> float:
        """Number of shares needed to hit target weight."""
        if self.price <= 0:
            return 0.0
        return self.target_value(portfolio_value) / self.price


@dataclass
class TradeRecommendation:
    """A single trade recommendation."""
    ticker: str
    side: PositionSide
    current_shares: float
    target_shares: float
    current_value: float
    target_value: float
    drift_percent: float

    @property
    def shares_to_trade(self) -> float:
        """Number of shares to buy or sell."""
        return abs(self.target_shares - self.current_shares)


@dataclass
class DriftPosition:
    """Drift information for a single position."""
    ticker: str
    current_weight: float
    target_weight: float
    drift: float
    current_value: float
    target_value: float
    current_shares: float
    current_price: float


@dataclass
class DriftReport:
    """Report of current portfolio drift from target weights."""
    timestamp: datetime
    total_value: float
    positions: list[DriftPosition]
    needs_rebalance: bool
    max_drift: float


class ETFReplica:
    """Manages a portfolio that replicates an ETF's holdings."""

    def __init__(
        self,
        target_etf: str,
        drift_threshold: float = 2.0,
        data_dir: Path = Path("data"),
    ):
        self.target_etf = target_etf
        self.drift_threshold = drift_threshold
        self.data_dir = data_dir
        self.constituents: dict[str, Constituent] = {}
        self.last_rebalance: datetime | None = None

    @property
    def total_market_value(self) -> float:
        """Total portfolio market value."""
        return sum(c.market_value for c in self.constituents.values())

    def add(
        self,
        ticker: str,
        shares: float,
        weight: float,
        price: float,
    ) -> None:
        """Add or update a constituent."""
        self.constituents[ticker] = Constituent(
            ticker=ticker,
            shares=shares,
            weight=weight,
            price=price,
        )

    def remove(self, ticker: str) -> bool:
        """Remove a constituent. Returns True if it existed."""
        if ticker in self.constituents:
            del self.constituents[ticker]
            return True
        return False

    def update_price(self, ticker: str, price: float) -> bool:
        """Update the current price of a constituent."""
        if ticker not in self.constituents:
            return False
        self.constituents[ticker].price = price
        return True

    def update_shares(self, ticker: str, shares: float) -> bool:
        """Update the share count of a constituent."""
        if ticker not in self.constituents:
            return False
        self.constituents[ticker].shares = shares
        return True

    def needs_rebalance(self) -> bool:
        """Check if portfolio drift exceeds threshold."""
        if not self.constituents:
            return False

        total = self.total_market_value
        if total <= 0:
            return False

        for c in self.constituents.values():
            current_weight = (c.market_value / total) * 100
            drift = abs(current_weight - c.weight)
            if drift > self.drift_threshold:
                return True

        return False

    def drift_report(self) -> DriftReport:
        """Generate a full drift report."""
        total = self.total_market_value
        positions = []
        max_drift = 0.0

        for c in self.constituents.values():
            current_weight = (c.market_value / total * 100) if total > 0 else 0.0
            drift = current_weight - c.weight
            abs_drift = abs(drift)
            max_drift = max(max_drift, abs_drift)

            target_value = c.target_value(total)
            current_value = c.market_value

            positions.append(
                DriftPosition(
                    ticker=c.ticker,
                    current_weight=current_weight,
                    target_weight=c.weight,
                    drift=drift,
                    current_value=current_value,
                    target_value=target_value,
                    current_shares=c.shares,
                    current_price=c.price,
                )
            )

        return DriftReport(
            timestamp=datetime.now(),
            total_value=total,
            positions=positions,
            needs_rebalance=max_drift > self.drift_threshold,
            max_drift=max_drift,
        )

    def rebalance_trades(self) -> dict[str, TradeRecommendation]:
        """Calculate trades needed to rebalance to target weights."""
        if not self.constituents:
            return {}

        total = self.total_market_value
        if total <= 0:
            return {}

        trades = {}

        for c in self.constituents.values():
            current_weight = (c.market_value / total) * 100
            drift = current_weight - c.weight
            target_shares = c.target_shares(total)
            target_value = c.target_value(total)

            if drift > self.drift_threshold / 2:
                side = PositionSide.SELL
            elif drift < -self.drift_threshold / 2:
                side = PositionSide.BUY
            else:
                side = PositionSide.HOLD

            trades[c.ticker] = TradeRecommendation(
                ticker=c.ticker,
                side=side,
                current_shares=c.shares,
                target_shares=target_shares,
                current_value=c.market_value,
                target_value=target_value,
                drift_percent=drift,
            )

        return trades

    def apply_trades(self, trades: dict[str, TradeRecommendation]) -> None:
        """Apply executed trades to update share counts."""
        for ticker, trade in trades.items():
            if trade.side == PositionSide.HOLD:
                continue
            if ticker in self.constituents:
                self.constituents[ticker].shares = trade.target_shares
        self.last_rebalance = datetime.now()
