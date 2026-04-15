"""
LotTracker — local lot-level basis tracking for Tax-Loss Harvesting.

Key concepts:
- A Lot is a single buy transaction: symbol, qty, cost_per_share, acquired_date
- Lots are matched FIFO (oldest first) when selling
- Wash sale: can't harvest if we bought the same/complementary ETF within 31 days

Why local tracking?
Alpaca's API doesn't expose per-lot data (unlike eTrade). We record our own
buy lots and match them against positions when scanning for TLH opportunities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums & Dataclasses
# =============================================================================


class LotStatus(Enum):
    """Lifecycle status of a tax lot."""
    OPEN = "open"       # Available to sell
    PARTIAL = "partial" # Partially sold (remaining qty > 0)
    CLOSED = "closed"   # Fully sold


@dataclass
class Lot:
    """
    A tax lot — a single buy of a security.

    Lots are immutable records of purchase events. The qty field records
    the ORIGINAL purchase quantity. The remaining qty after sells is tracked
    separately in the tracker (not in this dataclass) to keep lots clean.

    For gain/loss calculations at any point in time, pass the current_price
    to the lot_gain() / lot_gain_percent() methods.
    """
    lot_id: str
    symbol: str
    qty: float                     # Original purchase quantity
    cost_per_share: float          # Price per share at purchase
    acquired_date: datetime
    order_id: str
    status: LotStatus = LotStatus.OPEN
    notes: str = ""

    @property
    def cost_basis(self) -> float:
        """Total cost basis of this lot (original qty × cost_per_share)."""
        return self.qty * self.cost_per_share

    @property
    def current_price(self) -> float:
        """Placeholder — current price is passed at call time, not stored."""
        return 0.0


@dataclass
class LotMatch:
    """Result of matching a sell order to one or more lots."""
    lot_id: str
    qty_matched: float
    gain: float  # Positive = gain, negative = loss


@dataclass
class RecentTrade:
    """A recently executed buy or sell — used for wash sale detection."""
    symbol: str
    side: str  # "buy" or "sell"
    date: datetime


# =============================================================================
# LotTracker
# =============================================================================


class LotTracker:
    """
    Tracks tax lots locally and determines which lots can be harvested.

    Usage:
        tracker = LotTracker(data_dir=Path("data"))

        # When we buy shares (e.g., building a position):
        tracker.record_buy("AAPL", qty=100, cost_per_share=150.0, order_id="...")

        # When scanning for TLH opportunities:
        harvestable = tracker.scan_harvestable_lots("AAPL", current_price=130.0)

        # When executing a TLH harvest (sell specific lot at loss):
        matches = tracker.record_sell("AAPL", qty=lot.qty, current_price=130.0)

        # Record that we BOUGHT the replacement ETF (for wash sale tracking):
        tracker.record_recent_trade("VOO", side="buy", date=today)
    """

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._lots_file = self.data_dir / "lots.json"
        self._recent_trades_file = self.data_dir / "recent_trades.json"

        self._lots: list[Lot] = []
        self._recent_trades: list[RecentTrade] = []

        self._load()

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------

    def _load(self) -> None:
        """Load state from disk."""
        # Load lots
        if self._lots_file.exists():
            with open(self._lots_file) as f:
                raw: list[dict[str, Any]] = json.load(f)
                self._lots = [
                    Lot(
                        lot_id=l["lot_id"],
                        symbol=l["symbol"],
                        qty=l["qty"],
                        cost_per_share=l["cost_per_share"],
                        acquired_date=datetime.fromisoformat(l["acquired_date"]),
                        order_id=l["order_id"],
                        status=LotStatus(l.get("status", "open")),
                        notes=l.get("notes", ""),
                    )
                    for l in raw
                ]

        # Load recent trades
        if self._recent_trades_file.exists():
            with open(self._recent_trades_file) as f:
                raw = json.load(f)
                self._recent_trades = [
                    RecentTrade(
                        symbol=t["symbol"],
                        side=t["side"],
                        date=datetime.fromisoformat(t["date"]),
                    )
                    for t in raw
                ]

    def _save(self) -> None:
        """Save state to disk."""
        lots_data = [
            {
                "lot_id": l.lot_id,
                "symbol": l.symbol,
                "qty": l.qty,
                "cost_per_share": l.cost_per_share,
                "acquired_date": l.acquired_date.isoformat(),
                "order_id": l.order_id,
                "status": l.status.value,
                "notes": l.notes,
            }
            for l in self._lots
        ]
        with open(self._lots_file, "w") as f:
            json.dump(lots_data, f, indent=2)

        trades_data = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "date": t.date.isoformat(),
            }
            for t in self._recent_trades
        ]
        with open(self._recent_trades_file, "w") as f:
            json.dump(trades_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Lot queries
    # --------------------------------------------------------------------------

    def get_lots(self, symbol: str) -> list[Lot]:
        """Get all lots for a symbol that still have remaining qty > 0.

        Only OPEN and PARTIAL lots with qty remaining are returned.
        Lots are sorted FIFO (oldest acquired_date first).
        """
        return sorted(
            [l for l in self._lots if l.symbol == symbol and l.qty > 0],
            key=lambda l: l.acquired_date,
        )

    def get_all_lots(self) -> list[Lot]:
        """Get ALL lots (including closed) — for debugging/admin."""
        return list(self._lots)

    def get_remaining_qty(self, symbol: str) -> float:
        """Total shares remaining across all open lots for symbol."""
        return sum(l.qty for l in self._lots if l.symbol == symbol and l.qty > 0)

    # --------------------------------------------------------------------------
    # Recording buys
    # --------------------------------------------------------------------------

    def record_buy(
        self,
        symbol: str,
        qty: float,
        cost_per_share: float,
        order_id: str,
        acquired_date: datetime | None = None,
    ) -> str:
        """Record a new buy lot. Returns the lot_id."""
        if acquired_date is None:
            acquired_date = datetime.now()

        lot = Lot(
            lot_id=str(uuid4())[:8].upper(),
            symbol=symbol.upper(),
            qty=qty,
            cost_per_share=cost_per_share,
            acquired_date=acquired_date,
            order_id=order_id,
            status=LotStatus.OPEN,
        )

        self._lots.append(lot)
        self._save()

        return lot.lot_id

    # --------------------------------------------------------------------------
    # Recording sells (FIFO lot matching)
    # --------------------------------------------------------------------------

    def record_sell(
        self,
        symbol: str,
        qty: float,
        current_price: float,
        sell_date: datetime | None = None,
    ) -> list[LotMatch]:
        """Match a sell order to lots using FIFO (oldest first).

        Returns a list of LotMatch records describing which lots were used
        and the gain/loss for each.

        Raises ValueError if total available qty < sell qty.
        """
        if sell_date is None:
            sell_date = datetime.now()

        symbol = symbol.upper()
        available_lots = self.get_lots(symbol)
        total_available = sum(l.qty for l in available_lots)

        if total_available < qty:
            raise ValueError(
                f"Insufficient lot quantity for {symbol}: "
                f"requested {qty}, available {total_available}"
            )

        matches: list[LotMatch] = []
        remaining_to_sell = qty

        for lot in available_lots:
            if remaining_to_sell <= 0:
                break

            qty_from_lot = min(remaining_to_sell, lot.qty)
            gain = self._calc_lot_gain(lot, qty_from_lot, current_price)

            matches.append(LotMatch(
                lot_id=lot.lot_id,
                qty_matched=qty_from_lot,
                gain=gain,
            ))

            # Reduce lot qty
            lot.qty = round(lot.qty - qty_from_lot, 6)
            remaining_to_sell = round(remaining_to_sell - qty_from_lot, 6)

            # Update lot status
            if lot.qty <= 0:
                lot.status = LotStatus.CLOSED
            else:
                lot.status = LotStatus.PARTIAL

        self._save()
        return matches

    # --------------------------------------------------------------------------
    # Gain/loss helpers
    # --------------------------------------------------------------------------

    def lot_gain(self, lot: Lot, current_price: float) -> float:
        """Calculate dollar gain/loss for a full lot at current price."""
        return self._calc_lot_gain(lot, lot.qty, current_price)

    def lot_gain_percent(
        self, lot: Lot, current_price: float
    ) -> float:
        """Calculate percent gain/loss from entry to current price."""
        if lot.cost_per_share <= 0 or current_price <= 0:
            return 0.0
        return ((current_price - lot.cost_per_share) / lot.cost_per_share) * 100

    def _calc_lot_gain(
        self, lot: Lot, qty: float, current_price: float
    ) -> float:
        """Dollar gain/loss = (current_price - cost_per_share) × qty."""
        return (current_price - lot.cost_per_share) * qty

    # --------------------------------------------------------------------------
    # TLH scanning — harvestable lots
    # --------------------------------------------------------------------------

    def scan_harvestable_lots(
        self,
        symbol: str,
        current_price: float,
        min_loss_amount: float = 0.0,
        as_of: datetime | None = None,
    ) -> list[Lot]:
        """Find lots at a loss that can be harvested (not in wash sale).

        Args:
            symbol: Ticker to scan
            current_price: Current market price per share
            min_loss_amount: Minimum dollar loss to qualify
            as_of: Date for wash sale check (default: now)

        Returns:
            List of Lots that are at a loss and not blocked by wash sale
        """
        if as_of is None:
            as_of = datetime.now()

        lots = self.get_lots(symbol)
        harvestable: list[Lot] = []

        for lot in lots:
            gain = self.lot_gain(lot, current_price)

            # Skip profitable lots
            if gain >= 0:
                continue

            loss_amount = abs(gain)

            # Skip if below minimum loss threshold
            if loss_amount < min_loss_amount:
                continue

            # Check wash sale: can we actually harvest this lot?
            if not self.can_harvest_lot(
                symbol=symbol,
                lot_id=lot.lot_id,
                as_of=as_of,
            ):
                continue

            harvestable.append(lot)

        # Sort by loss magnitude descending (harvest biggest losses first)
        harvestable.sort(
            key=lambda l: abs(self.lot_gain(l, current_price)),
            reverse=True,
        )
        return harvestable

    def can_harvest_lot(
        self,
        symbol: str,
        lot_id: str,
        replacement_etf: str | None = None,
        as_of: datetime | None = None,
    ) -> bool:
        """Check if a specific lot can be harvested (not blocked by wash sale).

        A lot is blocked if:
        1. The same symbol was bought within 31 days (30 before + 30 after)
        2. The replacement ETF was bought within 31 days

        Args:
            symbol: The symbol whose lot we're harvesting
            lot_id: The specific lot (currently just for validation)
            replacement_etf: The ETF we're planning to buy as replacement
            as_of: Date of the harvest (default: now)
        """
        if as_of is None:
            as_of = datetime.now()

        # Validate lot exists
        lot = next((l for l in self._lots if l.lot_id == lot_id), None)
        if lot is None:
            return False

        # Check: was this symbol bought in wash sale window?
        if self.was_bought_recently(symbol, as_of=as_of):
            return False

        # Check: was the replacement ETF bought in wash sale window?
        if replacement_etf:
            if self.was_bought_recently(replacement_etf.upper(), as_of=as_of):
                return False

        return True

    # --------------------------------------------------------------------------
    # Recent trades (wash sale detection)
    # --------------------------------------------------------------------------

    def record_recent_trade(
        self,
        symbol: str,
        side: str,
        date: datetime | None = None,
    ) -> None:
        """Record a buy or sell that happened recently — for wash sale tracking.

        This should be called whenever an order fills, so the tracker knows
        about recent activity. We track both buys and sells, but only buys
        matter for wash sale detection.

        Args:
            symbol: Ticker
            side: "buy" or "sell"
            date: Date of the trade (default: now)
        """
        if date is None:
            date = datetime.now()

        self._recent_trades.append(RecentTrade(
            symbol=symbol.upper(),
            side=side.lower(),
            date=date,
        ))
        self._save()

    def was_bought_recently(
        self,
        symbol: str,
        as_of: datetime | None = None,
    ) -> bool:
        """Check if a symbol was bought within the 30-day wash sale window.

        IRS wash sale rule: loss is disallowed if you buy substantially
        identical securities within 30 days BEFORE OR AFTER the sale date.

        Args:
            symbol: Ticker to check
            as_of: The sale date to check against (default: today)

        Returns:
            True if symbol was bought in the window [as_of - 30 days, as_of + 30 days]
        """
        if as_of is None:
            as_of = datetime.now()

        symbol = symbol.upper()
        # Window: 30 days before through 30 days after the sale
        window_start = as_of - timedelta(days=30)
        window_end = as_of + timedelta(days=30)

        for trade in self._recent_trades:
            if trade.symbol == symbol and trade.side == "buy":
                if window_start <= trade.date <= window_end:
                    return True

        return False

    def get_recent_trades(
        self,
        days: int = 31,
        side: str | None = None,
        as_of: datetime | None = None,
    ) -> list[RecentTrade]:
        """Get recent trades within the last N days, optionally filtered by side."""
        if as_of is None:
            as_of = datetime.now()

        cutoff = as_of - timedelta(days=days)
        trades = [t for t in self._recent_trades if t.date >= cutoff]

        if side:
            trades = [t for t in trades if t.side == side.lower()]

        return sorted(trades, key=lambda t: t.date, reverse=True)
