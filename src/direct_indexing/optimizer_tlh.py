"""
Optimizer-based Tax-Loss Harvesting for Direct Indexing.

This module provides TLH using the Oracle optimizer instead of
rule-based threshold checking. Key differences:

- Optimization considers ALL constraints simultaneously
- Lot-level harvesting with proper sizing
- True direct indexing: harvested losses → cash → wait 31 days → repurchase original
- No ETF wrapper pattern (no VOO/SPY/IVV replacements)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .alpaca_client import AlpacaClient, Position
from .config import TLHConfig
from .lot_tracker import LotTracker
from .optimizer import Oracle, OracleStrategy, OptimizationResult


@dataclass
class HarvestResult:
    """Result of a harvest operation."""
    symbol: str
    loss_amount: float
    loss_percent: float
    lots_harvested: int = 0
    success: bool = False
    error: str | None = None
    harvest_date: datetime = field(default_factory=datetime.now)


@dataclass
class WashSaleEntry:
    """Wash sale tracking entry."""
    symbol: str
    sold_date: datetime
    sold_loss: float
    wash_sale_end_date: datetime
    status: str = "ACTIVE"
    notes: str = ""


class OptimizerTLHEngine:
    """
    Optimization-based Tax-Loss Harvesting Engine.

    Uses MILP optimization (Oracle) to find the best set of TLH trades
    rather than simple threshold-based rules.

    Key features:
    - Lot-level basis tracking
    - 31-day wash sale window (no replacement buy during this period)
    - Optimization considers tax cost + drift + transaction cost jointly
    - Cash held after harvest (no ETF wrapper)
    """

    def __init__(
        self,
        client: AlpacaClient,
        config: TLHConfig,
        data_dir: Path = Path("data"),
        lot_tracker: Optional[LotTracker] = None,
    ) -> None:
        self.client = client
        self.config = config
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if lot_tracker is None:
            lot_tracker = LotTracker(data_dir=data_dir)
        self._lot_tracker = lot_tracker

        self.wash_sale_file = data_dir / "optimizer_wash_sales.json"
        self.history_file = data_dir / "optimizer_tlh_history.json"
        self._wash_sales: list[WashSaleEntry] = []

        self._load_state()
        self._sync_positions_to_lots()

        # Initialize Oracle optimizer
        tax_rates = pd.DataFrame({
            "gain_type": ["short_term", "long_term", "interest"],
            "total_rate": [0.37, 0.20, 0.37],
        })
        self._oracle = Oracle(
            current_date=datetime.now().date(),
            tax_rates=tax_rates,
            wash_window_days=self.config.wash_sale_window_days,
            min_weight_multiplier=self.config.min_weight_multiplier,
            max_weight_multiplier=self.config.max_weight_multiplier,
        )

    def _load_state(self) -> None:
        """Load state from disk."""
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

    def _save_state(self) -> None:
        """Save state to disk."""
        data = [
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
            json.dump(data, f, indent=2)

    def _sync_positions_to_lots(self) -> None:
        """Sync Alpaca positions into lot tracker on startup."""
        positions = self.client.get_positions()
        for pos in positions:
            existing_lots = self._lot_tracker.get_lots(pos.symbol)
            total_lot_qty = sum(lot.qty for lot in existing_lots)

            if not existing_lots:
                if pos.qty > 0 and pos.avg_entry_price > 0:
                    self._lot_tracker.record_buy(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        cost_per_share=pos.avg_entry_price,
                        order_id=f"bootstrap-{pos.symbol}",
                        acquired_date=datetime.now(),
                    )
            elif total_lot_qty < pos.qty:
                delta_qty = pos.qty - total_lot_qty
                self._lot_tracker.record_buy(
                    symbol=pos.symbol,
                    qty=delta_qty,
                    cost_per_share=pos.avg_entry_price,
                    order_id=f"bootstrap-supplemental-{pos.symbol}",
                    acquired_date=datetime.now(),
                )

    def is_in_wash_sale_period(self, symbol: str) -> bool:
        """Check if symbol is in wash sale period."""
        now = datetime.now()
        for entry in self._wash_sales:
            if entry.symbol == symbol and entry.status == "ACTIVE":
                if entry.wash_sale_end_date > now:
                    return True
        return False

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

    def record_wash_sale(
        self,
        symbol: str,
        loss_amount: float,
        sold_date: datetime | None = None,
    ) -> WashSaleEntry:
        """Record a wash sale event (31-day window)."""
        if sold_date is None:
            sold_date = datetime.now()
        wash_end = sold_date + timedelta(days=self.config.wash_sale_window_days)
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

    def get_tax_lots_dataframe(self) -> pd.DataFrame:
        """Get current tax lots as DataFrame for optimizer."""
        positions = self.client.get_positions()
        rows = []

        for pos in positions:
            lots = self._lot_tracker.get_lots(pos.symbol)
            if not lots:
                continue
            for lot in lots:
                rows.append({
                    "tax_lot_id": lot.lot_id,
                    "identifier": pos.symbol,
                    "quantity": lot.qty,
                    "cost_basis": lot.cost_basis,
                    "date_acquired": lot.acquired_date,
                })

        if not rows:
            return pd.DataFrame(columns=["tax_lot_id", "identifier", "quantity", "cost_basis", "date_acquired"])
        return pd.DataFrame(rows)

    def get_prices_dataframe(self, tickers: list[str]) -> pd.DataFrame:
        """Get current prices as DataFrame for optimizer."""
        rows = []
        for ticker in tickers:
            price = self.client.get_latest_price(ticker)
            if price:
                rows.append({"identifier": ticker, "price": price})
        if not rows:
            return pd.DataFrame(columns=["identifier", "price"])
        return pd.DataFrame(rows)

    def get_target_weights(
        self,
        target_etf: str,
        csv_path: Path | None = None,
    ) -> pd.DataFrame:
        """Get target weights from CSV or ETF constituents."""
        if csv_path and csv_path.exists():
            import csv
            rows = []
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        "identifier": row["ticker"].strip().upper(),
                        "target_weight": float(row["weight"]),
                    })
            return pd.DataFrame(rows)

        return pd.DataFrame(columns=["identifier", "target_weight"])

    def identify_tlh_opportunities(
        self,
        tax_lots: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> list[dict]:
        """
        Identify TLH opportunities using Oracle strategy.

        Returns list of opportunities with lot-level details.
        """
        strategy = OracleStrategy(
            tax_lots=tax_lots,
            target_weights=self.get_target_weights("SPY"),
            prices=prices,
            cash=self.client.get_cash() or 0.0,
        )

        min_loss = self.config.loss_threshold_percent / 100.0
        min_value = self.config.min_loss_amount

        opportunities = strategy.identify_tlh_opportunities(
            min_loss_threshold=min_loss,
            min_harvest_value=min_value,
        )

        return [
            {
                "tax_lot_id": o.tax_lot_id,
                "identifier": o.identifier,
                "quantity": o.quantity,
                "cost_basis": o.cost_basis,
                "current_value": o.current_value,
                "loss_amount": o.loss_amount,
                "loss_percentage": o.loss_percentage,
                "potential_tax_savings": o.potential_tax_savings,
                "priority": o.priority,
            }
            for o in opportunities
        ]

    def run_optimization(
        self,
        tax_lots: pd.DataFrame,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
        cash: float,
    ) -> OptimizationResult:
        """
        Run Oracle optimization to find best trades.

        Args:
            tax_lots: Current tax lots
            target_weights: Target portfolio weights
            prices: Current prices
            cash: Available cash

        Returns:
            OptimizationResult with optimal trades
        """
        return self._oracle.optimize(
            tax_lots=tax_lots,
            target_weights=target_weights,
            prices=prices,
            cash=cash,
            min_trade_value=self.config.min_notional,
        )

    def execute_harvest(
        self,
        symbol: str,
        tax_lot_id: str,
        quantity: float,
        current_price: float,
    ) -> HarvestResult:
        """
        Execute a single lot harvest: sell at loss, hold cash.

        Key difference from rule-based TLH:
        - NO replacement buy
        - Cash sits until wash sale window expires
        - After 31 days, can repurchase original ticker

        Args:
            symbol: Ticker to harvest
            tax_lot_id: Specific lot to harvest
            quantity: Number of shares to sell
            current_price: Current market price

        Returns:
            HarvestResult with details
        """
        try:
            cost_basis = 0.0
            lots = self._lot_tracker.get_lots(symbol)
            for lot in lots:
                if lot.lot_id == tax_lot_id:
                    cost_basis = lot.cost_basis
                    break

            loss_amount = (current_price * quantity) - cost_basis

            if loss_amount >= 0:
                return HarvestResult(
                    symbol=symbol,
                    loss_amount=0.0,
                    loss_percent=0.0,
                    success=False,
                    error="Not a losing lot",
                )

            self._lot_tracker.record_sell(
                symbol=symbol,
                qty=quantity,
                current_price=current_price,
            )

            from .alpaca_client import OrderSide, OrderType
            self.client.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=quantity,
            )

            if self.config.wash_sale_enabled:
                self.record_wash_sale(
                    symbol=symbol,
                    loss_amount=abs(loss_amount),
                )

            if self.config.carryforward_enabled:
                self._add_to_carryforward(
                    amount=abs(loss_amount),
                    source=f"{symbol} harvest (lot {tax_lot_id})",
                )

            return HarvestResult(
                symbol=symbol,
                loss_amount=abs(loss_amount),
                loss_percent=abs(loss_amount / cost_basis * 100) if cost_basis > 0 else 0,
                lots_harvested=1,
                success=True,
                harvest_date=datetime.now(),
            )

        except Exception as e:
            return HarvestResult(
                symbol=symbol,
                loss_amount=0.0,
                loss_percent=0.0,
                success=False,
                error=str(e),
            )

    def _add_to_carryforward(self, amount: float, source: str) -> None:
        """Add losses to carryforward ledger."""
        cf_file = self.data_dir / "carryforward.json"
        entries = []
        if cf_file.exists():
            with open(cf_file) as f:
                entries = json.load(f)

        entries.append({
            "date": datetime.now().isoformat(),
            "amount": amount,
            "source": source,
            "utilized": 0.0,
            "remaining": amount,
        })

        with open(cf_file, "w") as f:
            json.dump(entries, f, indent=2)

    def get_wash_sales(self, status: str | None = None) -> list[WashSaleEntry]:
        """Get wash sale entries."""
        if status is None:
            return self._wash_sales.copy()
        return [e for e in self._wash_sales if e.status == status]

    def record_buy(
        self,
        symbol: str,
        qty: float,
        cost_per_share: float,
        order_id: str,
        acquired_date: datetime | None = None,
    ) -> str:
        """Record a buy lot."""
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
        """Record a sell against lots (FIFO)."""
        return self._lot_tracker.record_sell(
            symbol=symbol,
            qty=qty,
            current_price=current_price,
            sell_date=sell_date,
        )
