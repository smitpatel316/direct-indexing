"""
Pure Direct Indexer — The main rebalancing + TLH engine.

Coordinates:
1. S&P 500 weight tracking
2. 31-day rebalancing cycle
3. Tax-loss harvesting with sector substitutes
4. Wash sale tracking
5. Alpaca order execution

The 31-day rebalancing cycle sidesteps wash sale by:
- Selling at a rebalance triggers no wash sale IF the replacement is a sector substitute
- After 31 days, the restriction on the original ticker lapses
- We can swap back to the original if desired
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from .alpaca_client import AlpacaClient, OrderSide, Position, Order
from .sp500 import get_sp500
from .substitute_finder import get_substitute_finder


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RebalanceReason(str, Enum):
    SCHEDULED = "scheduled"
    DRIFT = "drift"
    RECONSTITUTION = "reconstitution"
    TLH = "tlh"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeOrder:
    """An order to be submitted to Alpaca."""
    symbol: str
    side: OrderSide
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    reason: str = ""
    substitute: Optional[str] = None


@dataclass
class RebalancePlan:
    """A planned rebalancing action."""
    sell_orders: list[TradeOrder] = field(default_factory=list)
    buy_orders: list[TradeOrder] = field(default_factory=list)
    tlh_sells: list[TradeOrder] = field(default_factory=list)
    tlh_buys: list[TradeOrder] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class WashSaleEntry:
    """A wash sale restriction record."""
    ticker: str
    harvest_date: date
    reopen_date: date
    substitute_used: str
    original_lot_id: Optional[str] = None

    def is_restricted(self, as_of: date) -> bool:
        return as_of < self.reopen_date


# ---------------------------------------------------------------------------
# Tax Lot Tracker
# ---------------------------------------------------------------------------

class TaxLotTracker:
    """Tracks buy lots per ticker for TLH decision-making."""

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or Path("data/tax_lots.json")
        self.lots: dict[str, list] = {}
        self.wash_sales: list[WashSaleEntry] = []
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                data = json.load(f)
            self.lots = data.get("lots", {})
            self.wash_sales = [
                WashSaleEntry(
                    ticker=e["ticker"],
                    harvest_date=date.fromisoformat(e["harvest_date"]),
                    reopen_date=date.fromisoformat(e["reopen_date"]),
                    substitute_used=e["substitute_used"],
                    original_lot_id=e.get("original_lot_id"),
                )
                for e in data.get("wash_sales", [])
            ]

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "lots": self.lots,
            "wash_sales": [
                {
                    "ticker": e.ticker,
                    "harvest_date": e.harvest_date.isoformat(),
                    "reopen_date": e.reopen_date.isoformat(),
                    "substitute_used": e.substitute_used,
                    "original_lot_id": e.original_lot_id,
                }
                for e in self.wash_sales
            ],
        }
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def is_restricted(self, ticker: str, as_of: date) -> bool:
        for ws in self.wash_sales:
            if ws.ticker == ticker and ws.is_restricted(as_of):
                return True
        return False

    def add_wash_sale(self, entry: WashSaleEntry) -> None:
        self.wash_sales.append(entry)
        self._save()

    def get_open_wash_sales(self, as_of: date) -> list[WashSaleEntry]:
        return [ws for ws in self.wash_sales if ws.is_restricted(as_of)]

    def prune_expired(self, as_of: date) -> int:
        before = len(self.wash_sales)
        self.wash_sales = [ws for ws in self.wash_sales if ws.is_restricted(as_of)]
        removed = before - len(self.wash_sales)
        if removed:
            self._save()
        return removed

    def update_lots_from_positions(self, positions: list[Position], prices: dict[str, float]) -> None:
        for pos in positions:
            if pos.symbol not in self.lots:
                self.lots[pos.symbol] = []
            if not self.lots[pos.symbol]:
                self.lots[pos.symbol] = [{
                    "lot_id": f"{pos.symbol}-synth-{date.today().isoformat()}",
                    "qty": pos.qty,
                    "cost_per_share": pos.avg_entry_price,
                    "buy_date": date.today().isoformat(),
                }]
        self._save()


# ---------------------------------------------------------------------------
# Main Direct Indexer
# ---------------------------------------------------------------------------

class PureDirectIndexer:
    """
    Pure Direct Indexing engine.

    Algorithm:
    1. On init: load S&P 500 weights, build substitute map
    2. Each 31-day rebalance:
       a. Compute current weights vs target weights
       b. Identify positions with drift > threshold → generate trade orders
       c. Scan for TLH opportunities: unrealized loss > $10 AND > 1% of position
       d. Execute sell orders (including TLH sells) first
       e. Execute buy orders (including substitute buys for TLH) second
       f. Log everything
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        cache_dir: Optional[Path] = None,
        drift_threshold: float = 0.0005,
        tlh_loss_min: float = 10.0,
        tlh_loss_pct: float = 0.01,
        rebalance_days: int = 31,
    ):
        self.alpaca = alpaca_client
        self.cache_dir = cache_dir or Path("data/direct_indexer")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold
        self.tlh_loss_min = tlh_loss_min
        self.tlh_loss_pct = tlh_loss_pct
        self.rebalance_days = rebalance_days

        self.sp500 = get_sp500()
        self.sub_finder = get_substitute_finder()
        self.lot_tracker = TaxLotTracker(cache_dir=self.cache_dir / "tax_lots.json")
        self.last_rebalance: Optional[date] = None

        self._state_file = self.cache_dir / "state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self._state_file.exists():
            with open(self._state_file) as f:
                data = json.load(f)
            self.last_rebalance = (
                date.fromisoformat(data["last_rebalance"])
                if data.get("last_rebalance")
                else None
            )

    def _save_state(self) -> None:
        with open(self._state_file, "w") as f:
            json.dump({
                "last_rebalance": (
                    self.last_rebalance.isoformat()
                    if self.last_rebalance
                    else None
                ),
            }, f, default=str)

    # -------------------------------------------------------------------------
    # Core rebalancing logic
    # -------------------------------------------------------------------------

    async def rebalance(self, reason: RebalanceReason = RebalanceReason.SCHEDULED) -> RebalancePlan:
        """Execute a full rebalance + TLH cycle."""
        today = date.today()
        print(f"\n=== Rebalance {today} (reason: {reason.value}) ===")

        account = self.alpaca.get_account()
        positions = self.alpaca.get_positions()
        portfolio_value = account.portfolio_value
        print(f"Portfolio value: ${portfolio_value:,.2f}")

        prices = self._get_current_prices([p.symbol for p in positions])
        self.lot_tracker.update_lots_from_positions(positions, prices)
        self.lot_tracker.prune_expired(today)

        target_weights = self.sp500.get_weights()
        current_weights = self._compute_current_weights(positions, prices, portfolio_value)

        plan = RebalancePlan()

        # --- DRIFT-BASED TRADES ---
        for ticker, target_w in target_weights.items():
            current_w = current_weights.get(ticker, 0.0)
            drift = abs(current_w - target_w)

            if drift > self.drift_threshold:
                target_value = portfolio_value * target_w
                current_value = portfolio_value * current_w
                delta_value = target_value - current_value
                price = prices.get(ticker) or 1.0

                if abs(delta_value) > 50.0:  # minimum $50 trade
                    qty = abs(delta_value) / price
                    if qty > 0.001:
                        if delta_value > 0:
                            plan.buy_orders.append(TradeOrder(
                                symbol=ticker, side=OrderSide.BUY, qty=qty,
                                reason=f"drift_{drift:.4f}",
                            ))
                        else:
                            plan.sell_orders.append(TradeOrder(
                                symbol=ticker, side=OrderSide.SELL, qty=qty,
                                reason=f"drift_{drift:.4f}",
                            ))

        # --- TLH SCAN ---
        print(f"\nScanning for TLH opportunities...")
        tlh_opportunities = self._scan_tlh(positions, prices, portfolio_value)
        for opp in tlh_opportunities:
            ticker = opp["ticker"]
            substitute = opp["substitute"]

            if self.lot_tracker.is_restricted(ticker, today):
                ws = next(w for w in self.lot_tracker.get_open_wash_sales(today) if w.ticker == ticker)
                print(f"  {ticker}: SKIPPED (wash sale until {ws.reopen_date})")
                continue

            plan.tlh_sells.append(TradeOrder(
                symbol=ticker,
                side=OrderSide.SELL,
                qty=opp["qty"],
                reason=f"tlh_${opp['loss']:.0f}",
                substitute=substitute,
            ))

            if substitute:
                sub_price = prices.get(substitute, 0.0)
                if sub_price > 0:
                    plan.tlh_buys.append(TradeOrder(
                        symbol=substitute,
                        side=OrderSide.BUY,
                        qty=opp["qty"],
                        reason=f"substitute_for_{ticker}",
                    ))

            self.lot_tracker.add_wash_sale(WashSaleEntry(
                ticker=ticker,
                harvest_date=today,
                reopen_date=today + timedelta(days=31),
                substitute_used=substitute or "",
            ))
            print(f"  {ticker}: Harvest ${opp['loss']:.2f} → {substitute}")

        # --- EXECUTE ---
        print(f"\nExecuting orders: {len(plan.sell_orders)} sells, "
              f"{len(plan.tlh_sells)} TLH, {len(plan.buy_orders)} buys, "
              f"{len(plan.tlh_buys)} TLH buys...")

        for order in plan.sell_orders + plan.tlh_sells:
            try:
                self.alpaca.submit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=order.side,
                    order_type="market",
                )
                print(f"  SELL {order.symbol}: {order.qty:.4f} shares")
            except Exception as e:
                print(f"  ERROR selling {order.symbol}: {e}")

        for order in plan.buy_orders + plan.tlh_buys:
            try:
                self.alpaca.submit_order(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=order.side,
                    order_type="market",
                )
                print(f"  BUY {order.symbol}: {order.qty:.4f} shares")
            except Exception as e:
                print(f"  ERROR buying {order.symbol}: {e}")

        self.last_rebalance = today
        self._save_state()
        return plan

    def _scan_tlh(
        self,
        positions: list[Position],
        prices: dict[str, float],
        portfolio_value: float,
    ) -> list[dict]:
        """Scan positions for TLH opportunities."""
        opportunities = []
        for pos in positions:
            price = prices.get(pos.symbol)
            if price is None:
                continue

            cost_basis = pos.qty * pos.avg_entry_price
            current_value = pos.qty * price
            loss = cost_basis - current_value

            min_loss = max(self.tlh_loss_min, cost_basis * self.tlh_loss_pct)
            if loss < min_loss:
                continue

            substitute = self.sub_finder.get_substitute(pos.symbol)
            if not substitute:
                continue

            opportunities.append({
                "ticker": pos.symbol,
                "loss": loss,
                "qty": pos.qty,
                "substitute": substitute,
            })

        return opportunities

    def _compute_current_weights(
        self,
        positions: list[Position],
        prices: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, float]:
        weights = {}
        for pos in positions:
            price = prices.get(pos.symbol, 0.0)
            if price > 0 and portfolio_value > 0:
                weights[pos.symbol] = (pos.qty * price) / portfolio_value
        return weights

    def _get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        for ticker in tickers:
            bars = self.alpaca.get_bars(ticker, limit=5)
            if bars:
                latest = bars[-1]
                if hasattr(latest, 'close'):
                    prices[ticker] = float(latest.close)
        return prices

    # -------------------------------------------------------------------------
    # 31-day cycle
    # -------------------------------------------------------------------------

    def needs_rebalance(self) -> bool:
        if self.last_rebalance is None:
            return True
        return (date.today() - self.last_rebalance).days >= self.rebalance_days

    def get_days_since_rebalance(self) -> int:
        if self.last_rebalance is None:
            return 999
        return (date.today() - self.last_rebalance).days

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        today = date.today()
        try:
            positions = self.alpaca.get_positions()
            account = self.alpaca.get_account()
            portfolio_value = account.portfolio_value
            cash = account.cash
        except Exception:
            portfolio_value = 0.0
            cash = 0.0
            positions = []

        return {
            "portfolio_value": portfolio_value,
            "num_positions": len(positions),
            "last_rebalance": (
                self.last_rebalance.isoformat()
                if self.last_rebalance
                else None
            ),
            "days_since_rebalance": self.get_days_since_rebalance(),
            "needs_rebalance": self.needs_rebalance(),
            "wash_sales_active": len(self.lot_tracker.get_open_wash_sales(today)),
            "cash": cash,
        }
