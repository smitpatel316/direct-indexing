"""
Tax-Loss Harvesting Engine
Modern implementation with carryforward tracking and wash sale management.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .alpaca_client import AlpacaClient, Position
from .config import TLHConfig


@dataclass
class HarvestResult:
    """Result of a harvest operation."""
    symbol: str
    loss_amount: float
    loss_percent: float
    swap_target: str
    success: bool
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
    """Tax-Loss Harvesting Engine."""

    def __init__(
        self,
        client: AlpacaClient,
        config: TLHConfig,
        data_dir: Path = Path("data"),
    ) -> None:
        self.client = client
        self.config = config
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.wash_sale_file = data_dir / "wash_sales.json"
        self.carryforward_file = data_dir / "carryforward.json"
        self.history_file = data_dir / "tlh_history.json"

        self._wash_sales: list[WashSaleEntry] = []
        self._carryforward: list[CarryforwardEntry] = []

        self._load_state()

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
        """Check if symbol is in wash sale period."""
        now = datetime.now()
        for entry in self._wash_sales:
            if entry.symbol == symbol and entry.status == "ACTIVE":
                if entry.wash_sale_end_date > now:
                    return True
        return False

    def record_wash_sale(self, symbol: str, loss_amount: float,
                        sold_date: datetime | None = None) -> WashSaleEntry:
        """Record a wash sale event."""
        if sold_date is None:
            sold_date = datetime.now()

        wash_end = sold_date + timedelta(days=31)  # 30-day rule + 1 day buffer

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

        # Use oldest entries first (FIFO)
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

    def scan_portfolio(self) -> list[Position]:
        """Scan portfolio for harvestable positions."""
        positions = self.client.get_positions()
        harvestable = []

        for pos in positions:
            # Skip if quantity is zero
            if pos.qty <= 0:
                continue

            # Check loss threshold
            threshold = self.config.loss_threshold_percent
            exceeds_threshold = abs(pos.loss_percent) >= threshold
            is_loss = pos.loss_percent < 0
            if exceeds_threshold and is_loss:
                # Check minimum loss amount
                if abs(pos.loss_amount) >= self.config.min_loss_amount:
                    # Check wash sale
                    if not self.is_in_wash_sale_period(pos.symbol):
                        harvestable.append(pos)

        # Sort by loss amount (largest first)
        harvestable.sort(key=lambda p: abs(p.loss_amount), reverse=True)

        return harvestable

    def execute_harvest(self, position: Position) -> HarvestResult:
        """Execute a single harvest: sell the losing position."""
        try:
            # Determine swap target
            swap_target = self.config.swap_etfs[0] if self.config.swap_etfs else "VOO"

            # Submit sell order
            from .alpaca_client import OrderSide, OrderType
            self.client.submit_order(
                symbol=position.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=position.qty,
            )

            # Record wash sale
            if self.config.wash_sale_enabled:
                self.record_wash_sale(
                    symbol=position.symbol,
                    loss_amount=abs(position.loss_amount),
                )

            # Add to carryforward
            if self.config.carryforward_enabled:
                self.add_to_carryforward(
                    amount=abs(position.loss_amount),
                    source=f"{position.symbol} harvest",
                )

            return HarvestResult(
                symbol=position.symbol,
                loss_amount=abs(position.loss_amount),
                loss_percent=abs(position.loss_percent),
                swap_target=swap_target,
                success=True,
            )

        except Exception as e:
            return HarvestResult(
                symbol=position.symbol,
                loss_amount=abs(position.loss_amount),
                loss_percent=abs(position.loss_percent),
                swap_target="",
                success=False,
                error=str(e),
            )

    def run_daily_scan(self) -> list[HarvestResult]:
        """Run the daily TLH scan and execute harvests."""
        results: list[HarvestResult] = []

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
                # Schedule swap buy for tomorrow
                self._schedule_swap(
                    position.symbol,
                    result.swap_target,
                    abs(position.loss_amount),
                )

        return results

    def _schedule_swap(
        self, original_symbol: str, target_etf: str, amount: float
    ) -> None:
        """Schedule a swap buy for the next trading day."""
        # This would be called from a scheduled function
        # For now, just log it
        swap_log = self.data_dir / "pending_swaps.json"

        swaps = []
        if swap_log.exists():
            with open(swap_log) as f:
                swaps = json.load(f)

        # Calculate quantity based on target ETF price
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
        # This would track historical harvests
        return sum(e.amount for e in self._carryforward)
