"""
Tax-Loss Harvesting Engine
Modern implementation with carryforward tracking and wash sale management.

Key improvements over v1:
- Lot-level basis tracking (which lots are at a loss, which are at a gain)
- Proper wash sale detection: checks if replacement ETF was bought in last 30 days
- FIFO lot matching when selling specific lots
- Recent trades tracking from Alpaca
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .alpaca_client import AlpacaClient, Position
from .config import TLHConfig

if TYPE_CHECKING:
    from .lot_tracker import LotTracker


@dataclass
class HarvestResult:
    """Result of a harvest operation."""
    symbol: str
    loss_amount: float
    loss_percent: float
    swap_target: str
    lots_harvested: int = 0  # Number of lots sold
    success: bool = False
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WashSaleEntry:
    """Wash sale tracking entry."""
    symbol: str
    sold_date: datetime
    sold_loss: float
    wash_sale_end_date: datetime
    status: str = "ACTIVE"  # ACTIVE, EXPIRED, UTILIZED
    notes: str = ""


@dataclass
class CarryforwardEntry:
    """Carryforward loss entry."""
    date: datetime
    amount: float
    source: str  # e.g., "AAPL harvest"
    utilized: float = 0.0  # Amount used against gains
    remaining: float = 0.0


class TLHEngine:
    """Tax-Loss Harvesting Engine with lot-level tracking."""

    def __init__(
        self,
        client: AlpacaClient,
        config: TLHConfig,
        data_dir: Path = Path("data"),
        lot_tracker: LotTracker | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lot tracker for lot-level basis tracking (optional for backwards compat)
        if lot_tracker is None:
            from .lot_tracker import LotTracker
            lot_tracker = LotTracker(data_dir=data_dir)
        self._lot_tracker = lot_tracker

        self.wash_sale_file = data_dir / "wash_sales.json"
        self.carryforward_file = data_dir / "carryforward.json"
        self.history_file = data_dir / "tlh_history.json"

        self._wash_sales: list[WashSaleEntry] = []
        self._carryforward: list[CarryforwardEntry] = []

        self._load_state()

        # Sync Alpaca positions into lot tracker on startup
        self._sync_positions_to_lots()

    def _load_state(self) -> None:
        """Load state from disk."""
        # Load wash sales
        if self.wash_sale_file.exists():
            with open(self.wash_sale_file) as f:
                data = json.load(f)
                self._wash_sales = [
                    WashSaleEntry(
                        symbol=e["symbol"],
                        sold_date=datetime.fromisoformat(e["sold_date"]),
                        sold_loss=e["sold_loss"],
                        wash_sale_end_date=datetime.fromisoformat(e["wash_sale_end_date"]),
                        status=e.get("status", "ACTIVE"),
                        notes=e.get("notes", ""),
                    )
                    for e in data
                ]

        # Load carryforward
        if self.carryforward_file.exists():
            with open(self.carryforward_file) as f:
                data = json.load(f)
                self._carryforward = [
                    CarryforwardEntry(
                        date=datetime.fromisoformat(e["date"]),
                        amount=e["amount"],
                        source=e["source"],
                        utilized=e.get("utilized", 0.0),
                        remaining=e.get("remaining", e["amount"]),
                    )
                    for e in data
                ]

    def _save_state(self) -> None:
        """Save state to disk."""
        # Save wash sales
        wash_sale_data = [
            {
                "symbol": e.symbol,
                "sold_date": e.sold_date.isoformat(),
                "sold_loss": e.sold_loss,
                "wash_sale_end_date": e.wash_sale_end_date.isoformat(),
                "status": e.status,
                "notes": e.notes,
            }
            for e in self._wash_sales
        ]
        with open(self.wash_sale_file, "w") as f:
            json.dump(wash_sale_data, f, indent=2)

        # Save carryforward
        carryforward_data = [
            {
                "date": e.date.isoformat(),
                "amount": e.amount,
                "source": e.source,
                "utilized": e.utilized,
                "remaining": e.remaining,
            }
            for e in self._carryforward
        ]
        with open(self.carryforward_file, "w") as f:
            json.dump(carryforward_data, f, indent=2)

    # --------------------------------------------------------------------------
    # Lot tracking integration
    # --------------------------------------------------------------------------

    def _sync_positions_to_lots(self) -> None:
        """Sync open positions from Alpaca into lot tracker.

        Called on startup. If a symbol is in Alpaca but has no lots in our
        tracker, we create a synthetic lot from the position data (avg_entry_price).

        This handles the case where positions were opened before lot tracking began.
        """
        positions = self.client.get_positions()

        for pos in positions:
            existing_lots = self._lot_tracker.get_lots(pos.symbol)
            if not existing_lots:
                # Bootstrap: create a single lot from position data
                # We use avg_entry_price as cost basis, qty as lot quantity
                if pos.qty > 0 and pos.avg_entry_price > 0:
                    self._lot_tracker.record_buy(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        cost_per_share=pos.avg_entry_price,
                        order_id=f"bootstrap-{pos.symbol}",
                        acquired_date=datetime.now(),  # We don't know actual date
                    )

    def _sync_recent_trades_from_alpaca(self, days: int = 31) -> None:
        """Pull recent trades from Alpaca and record in lot tracker.

        This keeps our wash sale tracking in sync with what actually happened.
        """
        try:
            recent = self.client.get_recent_trades(days=days)
            for symbol in recent.get("bought", []):
                self._lot_tracker.record_recent_trade(symbol, side="buy")
            for symbol in recent.get("sold", []):
                self._lot_tracker.record_recent_trade(symbol, side="sell")
        except Exception:
            # Non-fatal: Alpaca might not have recent trades available
            pass

    def record_buy(
        self,
        symbol: str,
        qty: float,
        cost_per_share: float,
        order_id: str,
        acquired_date: datetime | None = None,
    ) -> str:
        """Record a buy lot in the tracker. Call after order fills."""
        return self._lot_tracker.record_buy(
            symbol=symbol,
            qty=qty,
            cost_per_share=cost_per_share,
            order_id=order_id,
            acquired_date=acquired_date,
        )

    def record_sell(
        self,
        symbol: str,
        qty: float,
        current_price: float,
        sell_date: datetime | None = None,
    ) -> list:
        """Record a sell against lots (FIFO). Call after order fills."""
        return self._lot_tracker.record_sell(
            symbol=symbol,
            qty=qty,
            current_price=current_price,
            sell_date=sell_date,
        )

    def record_recent_trade(
        self, symbol: str, side: str, date: datetime | None = None
    ) -> None:
        """Record a buy or sell for wash sale tracking."""
        self._lot_tracker.record_recent_trade(symbol, side, date)

    # --------------------------------------------------------------------------
    # Wash sales
    # --------------------------------------------------------------------------

    def update_expired_wash_sales(self) -> int:
        """Update wash sale statuses, return count of newly expired."""
        now = datetime.now()
        expired_count = 0

        for entry in self._wash_sales:
            if entry.status == "ACTIVE" and entry.wash_sale_end_date < now:
                entry.status = "EXPIRED"
                expired_count += 1

        if expired_count > 0:
            self._save_state()

        return expired_count

    def is_in_wash_sale_period(self, symbol: str) -> bool:
        """Check if symbol is in wash sale period (from our tracked wash sales)."""
        now = datetime.now()
        for entry in self._wash_sales:
            if entry.symbol == symbol and entry.status == "ACTIVE":
                if entry.wash_sale_end_date > now:
                    return True
        return False

    def record_wash_sale(
        self,
        symbol: str,
        loss_amount: float,
        sold_date: datetime | None = None,
    ) -> WashSaleEntry:
        """Record a wash sale event."""
        if sold_date is None:
            sold_date = datetime.now()

        # 31 days: 30-day rule + 1 day buffer for boundary safety
        wash_end = sold_date + timedelta(days=31)

        entry = WashSaleEntry(
            symbol=symbol,
            sold_date=sold_date,
            sold_loss=loss_amount,
            wash_sale_end_date=wash_end,
            status="ACTIVE",
        )

        self._wash_sales.append(entry)
        self._save_state()

        return entry

    # --------------------------------------------------------------------------
    # Carryforward
    # --------------------------------------------------------------------------

    def add_to_carryforward(self, amount: float, source: str) -> CarryforwardEntry:
        """Add losses to carryforward ledger."""
        entry = CarryforwardEntry(
            date=datetime.now(),
            amount=amount,
            source=source,
            utilized=0.0,
            remaining=amount,
        )

        self._carryforward.append(entry)
        self._save_state()

        return entry

    def use_carryforward(self, amount: float, source: str) -> float:
        """Use carryforward losses against gains. Returns amount actually used."""
        total_available = sum(e.remaining for e in self._carryforward)

        if total_available == 0:
            return 0.0

        amount_to_use = min(amount, total_available)
        remaining = amount_to_use

        for entry in sorted(self._carryforward, key=lambda e: e.date):
            if remaining <= 0:
                break

            use_from_entry = min(remaining, entry.remaining)
            entry.utilized += use_from_entry
            entry.remaining -= use_from_entry
            remaining -= use_from_entry

        self._save_state()

        return amount_to_use

    def get_carryforward_balance(self) -> float:
        """Get total carryforward balance."""
        return sum(e.remaining for e in self._carryforward)

    def get_wash_sales(self, status: str | None = None) -> list[WashSaleEntry]:
        """Get wash sale entries, optionally filtered by status."""
        if status is None:
            return self._wash_sales.copy()
        return [e for e in self._wash_sales if e.status == status]

    # --------------------------------------------------------------------------
    # Portfolio scanning (lot-level)
    # --------------------------------------------------------------------------

    def scan_portfolio(self) -> list[Position]:
        """Scan portfolio for harvestable positions.

        Uses lot-level tracking to identify losing lots. A position is
        harvestable if it has at least one losing lot that can be sold
        (not blocked by wash sale).

        Returns positions with their harvestable lots attached.
        """
        positions = self.client.get_positions()
        harvestable: list[Position] = []

        for pos in positions:
            if pos.qty <= 0:
                continue

            # Get current price
            current_price = pos.current_price
            if current_price <= 0:
                try:
                    latest = self.client.get_latest_price(pos.symbol)
                    if latest:
                        current_price = latest
                except Exception:
                    continue

            if current_price <= 0:
                continue

            # Scan lots for this symbol using lot tracker
            # Pass the replacement ETF so wash sale check also blocks if
            # we've already bought the replacement (holding VOO from prior harvest)
            swap_target = (
                self.config.swap_etfs[0] if self.config.swap_etfs else "VOO"
            )
            harvestable_lots = self._lot_tracker.scan_harvestable_lots(
                symbol=pos.symbol,
                current_price=current_price,
                min_loss_amount=self.config.min_loss_amount,
                replacement_etf=swap_target,
            )

            if not harvestable_lots:
                continue

            # Calculate total loss across harvestable lots
            total_loss = sum(
                abs(self._lot_tracker.lot_gain(lot, current_price))
                for lot in harvestable_lots
            )

            # Check loss threshold on the aggregate position
            threshold = self.config.loss_threshold_percent
            if abs(pos.loss_percent) >= threshold and total_loss >= self.config.min_loss_amount:
                harvestable.append(pos)

        # Sort by loss amount (largest first)
        harvestable.sort(key=lambda p: abs(p.loss_amount), reverse=True)

        return harvestable

    def get_harvestable_lots_for_position(
        self,
        symbol: str,
        current_price: float | None = None,
    ) -> list:
        """Get losing lots for a specific symbol.

        This is the lot-level view — which specific lots are at a loss
        and can be harvested.
        """
        if current_price is None or current_price <= 0:
            latest = self.client.get_latest_price(symbol)
            if latest is None:
                return []
            current_price = latest

        swap_target = (
            self.config.swap_etfs[0] if self.config.swap_etfs else "VOO"
        )
        return self._lot_tracker.scan_harvestable_lots(
            symbol=symbol,
            current_price=current_price,
            min_loss_amount=self.config.min_loss_amount,
            replacement_etf=swap_target,
        )

    # --------------------------------------------------------------------------
    # Harvest execution
    # --------------------------------------------------------------------------

    def execute_harvest(
        self,
        position: Position,
        replacement_etf: str | None = None,
    ) -> HarvestResult:
        """Execute a single harvest: sell losing lots from a position.

        Args:
            position: The Alpaca position to harvest from
            replacement_etf: The ETF to buy as replacement (for wash sale tracking)

        Returns:
            HarvestResult with details of what was harvested
        """
        swap_target = replacement_etf or (
            self.config.swap_etfs[0] if self.config.swap_etfs else "VOO"
        )

        try:
            # Get current price
            current_price = position.current_price
            if current_price <= 0:
                current_price = self.client.get_latest_price(position.symbol) or 0

            # Get losing lots
            losing_lots = self.get_harvestable_lots_for_position(
                position.symbol, current_price
            )

            if not losing_lots:
                return HarvestResult(
                    symbol=position.symbol,
                    loss_amount=0.0,
                    loss_percent=0.0,
                    swap_target=swap_target,
                    success=False,
                    error="No losing lots found",
                )

            # Calculate total loss to harvest
            total_loss = sum(
                abs(self._lot_tracker.lot_gain(lot, current_price))
                for lot in losing_lots
            )

            # Sell the position (Alpaca sells entire position; lot tracking is
            # for gain/loss computation and record-keeping)
            total_qty = sum(lot.qty for lot in losing_lots)

            # Record sell in lot tracker BEFORE submitting (for FIFO tracking)
            # We record with current price as estimate; actual fill price may differ
            lot_matches = self._lot_tracker.record_sell(
                symbol=position.symbol,
                qty=total_qty,
                current_price=current_price,
            )

            # Submit sell order to Alpaca (entire position)
            from .alpaca_client import OrderSide, OrderType
            order = self.client.submit_order(
                symbol=position.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=position.qty,
            )

            # Record wash sale (we sold at a loss)
            if self.config.wash_sale_enabled:
                self.record_wash_sale(
                    symbol=position.symbol,
                    loss_amount=total_loss,
                )

            # Add to carryforward
            if self.config.carryforward_enabled:
                self.add_to_carryforward(
                    amount=total_loss,
                    source=f"{position.symbol} harvest ({len(losing_lots)} lots)",
                )

            # Record the replacement ETF buy in recent trades
            # This is critical: buying VOO after selling SPY triggers wash sale!
            if replacement_etf:
                self._lot_tracker.record_recent_trade(
                    replacement_etf.upper(),
                    side="buy",
                )

            return HarvestResult(
                symbol=position.symbol,
                loss_amount=total_loss,
                loss_percent=abs(position.loss_percent),
                swap_target=swap_target,
                lots_harvested=len(losing_lots),
                success=True,
            )

        except Exception as e:
            return HarvestResult(
                symbol=position.symbol,
                loss_amount=abs(position.loss_amount),
                loss_percent=abs(position.loss_percent),
                swap_target=swap_target,
                success=False,
                error=str(e),
            )

    def run_daily_scan(self) -> list[HarvestResult]:
        """Run the daily TLH scan and execute harvests."""
        results: list[HarvestResult] = []

        # Sync recent trades from Alpaca to keep wash sale tracking current
        self._sync_recent_trades_from_alpaca()

        # Update expired wash sales
        self.update_expired_wash_sales()

        # Scan for harvestable positions
        harvestable = self.scan_portfolio()

        if not harvestable:
            return results

        # Execute harvests
        for position in harvestable:
            result = self.execute_harvest(position)
            results.append(result)

            if result.success:
                self._schedule_swap(
                    position.symbol,
                    result.swap_target,
                    result.loss_amount,
                )

        return results

    def _schedule_swap(
        self, original_symbol: str, target_etf: str, amount: float
    ) -> None:
        """Schedule a swap buy for the next trading day."""
        swap_log = self.data_dir / "pending_swaps.json"

        swaps = []
        if swap_log.exists():
            with open(swap_log) as f:
                swaps = json.load(f)

        try:
            price = self.client.get_latest_price(target_etf)
            if price and price > 0:
                qty = int(amount / price)
                swaps.append({
                    "original_symbol": original_symbol,
                    "target_etf": target_etf,
                    "amount": amount,
                    "qty": qty,
                    "scheduled_date": (
                        datetime.now() + timedelta(days=1)
                    ).strftime("%Y-%m-%d"),
                    "status": "PENDING",
                })
        except Exception:
            pass

        with open(swap_log, "w") as f:
            json.dump(swaps, f, indent=2)

    # --------------------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Get TLH summary report."""
        harvestable = self.scan_portfolio()

        return {
            "carryforward_balance": self.get_carryforward_balance(),
            "active_wash_sales": len(self.get_wash_sales("ACTIVE")),
            "expired_wash_sales": len(self.get_wash_sales("EXPIRED")),
            "harvestable_positions": len(harvestable),
            "top_losses": [
                {
                    "symbol": p.symbol,
                    "loss_amount": abs(p.loss_amount),
                    "loss_percent": abs(p.loss_percent),
                }
                for p in harvestable[:5]
            ],
        }

    def get_ytd_harvested(self) -> float:
        """Get total harvested losses YTD."""
        return sum(e.amount for e in self._carryforward)
