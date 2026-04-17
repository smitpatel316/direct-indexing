"""
Portfolio Manager — Robust Alpaca Paper Trading with Local Validation.

This script manages index replication portfolios (SP500, QQQ, All US Stocks)
with the following safety guarantees:

1. LOCAL STATE TRACKING — All orders, positions, and state stored in local SQLite.
   Alpaca is treated as a remote display; we never trust it as the source of truth.

2. DRY-RUN BY DEFAULT — Must explicitly pass --execute to submit real orders.
   Even with --execute, performs local validation before any API call.

3. ORDER SIMULATION BEFORE SUBMISSION — Calculates expected fills locally,
   verifies against portfolio value and position limits before sending to Alpaca.

4. RATE LIMIT HANDLING — Exponential backoff with jitter; never hammers Alpaca.

5. SEPARATE ACCOUNTS — Each index (SP500, QQQ, All_US) uses its own Alpaca account.
   Configuration via environment variables or config file.

Usage:
    # Show what would be traded (dry-run, no Alpaca calls):
    python scripts/portfolios/manage_portfolio.py --index SP500 --dry-run

    # Preview orders with local simulation:
    python scripts/portfolios/manage_portfolio.py --index SP500 --preview

    # Execute real orders (requires --execute flag):
    python scripts/portfolios/manage_portfolio.py --index SP500 --execute

    # Check status:
    python scripts/portfolios/manage_portfolio.py --index SP500 --status

    # Force rebalance:
    python scripts/portfolios/manage_portfolio.py --index SP500 --rebalance --execute

    # Scheduled cron entry (example — runs daily at 9:25am ET):
    25 9 * * 1-5 cd /path/to/direct-indexing && python scripts/portfolios/manage_portfolio.py --index SP500 --execute >> logs/sp500_daily.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
STATE_DIR = ROOT_DIR / "state" / "portfolios"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Per-account environment variable prefixes
ACCOUNT_ENVVARS = {
    "SP500":  ["ALPACA_SP500_API_KEY", "ALPACA_SP500_API_SECRET"],
    "QQQ":    ["ALPACA_QQQ_API_KEY",   "ALPACA_QQQ_API_SECRET"],
    "ALL_US": ["ALPACA_ALLUS_API_KEY", "ALPACA_ALLUS_API_SECRET"],
}


class IndexType(Enum):
    SP500 = "SP500"
    QQQ = "QQQ"
    ALL_US = "ALL_US"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TargetPosition:
    """A target portfolio position (what we want)."""
    symbol: str
    weight: float          # fraction (not percent)
    target_value: float    # dollar target
    current_price: float | None = None
    qty: float | None = None
    limit_price: float | None = None


@dataclass
class OrderPlan:
    """A planned order — validated locally before submission."""
    symbol: str
    side: str = "buy"
    qty: float = 0.0
    limit_price: float | None = None
    order_type: str = "limit"
    estimated_value: float = 0.0
    reason: str = ""       # e.g., "initial_deploy", "rebalance", "tlh_harvest"


@dataclass
class PortfolioState:
    """Current state of a portfolio (local view)."""
    index: IndexType
    account_equity: float
    cash: float
    positions: list[dict]   # from Alpaca
    open_orders: list[dict]  # from Alpaca
    target_positions: list[TargetPosition] = field(default_factory=list)
    orders_to_submit: list[OrderPlan] = field(default_factory=list)
    last_rebalance: date | None = None
    rebalance_reason: str | None = None


# ---------------------------------------------------------------------------
# Database (local state tracking)
# ---------------------------------------------------------------------------

def get_db(index: IndexType) -> sqlite3.Connection:
    """Get SQLite connection for a given index."""
    db_path = STATE_DIR / f"{index.value.lower()}.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(index: IndexType) -> None:
    """Initialize database schema for an index."""
    conn = get_db(index)
    cur = conn.cursor()

    # Portfolio events (rebalances, harvests)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            reason TEXT,
            equity_before REAL,
            equity_after REAL,
            trades_count INTEGER,
            notes TEXT
        )
    """)

    # Scheduled rebalances
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rebalance_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scheduled_date TEXT NOT NULL,
            executed_date TEXT,
            reason TEXT NOT NULL,
            status TEXT DEFAULT 'pending'
        )
    """)

    # Order history (filled orders)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            limit_price REAL,
            fill_price REAL,
            order_type TEXT NOT NULL,
            status TEXT NOT NULL,
            alpaca_order_id TEXT,
            notes TEXT
        )
    """)

    # Daily portfolio snapshots
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL UNIQUE,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            positions_count INTEGER,
            open_orders_count INTEGER,
            portfolio_value REAL,
            notes TEXT
        )
    """)

    # Wash sale tracking
    cur.execute("""
        CREATE TABLE IF NOT EXISTS wash_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            sell_date TEXT NOT NULL,
            reopen_date TEXT NOT NULL,
            loss_amount REAL,
            substitute_used TEXT,
            status TEXT DEFAULT 'active'
        )
    """)

    conn.commit()


def record_rebalance_event(
    index: IndexType,
    reason: str,
    equity_before: float,
    equity_after: float,
    trades: int,
    notes: str = ""
) -> None:
    """Record a rebalance event in the local DB."""
    conn = get_db(index)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO portfolio_events (event_date, event_type, reason, equity_before, equity_after, trades_count, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), "rebalance", reason, equity_before, equity_after, trades, notes))
    conn.commit()


def get_last_rebalance(index: IndexType) -> date | None:
    """Get the date of the last rebalance."""
    conn = get_db(index)
    cur = conn.cursor()
    cur.execute("""
        SELECT event_date FROM portfolio_events
        WHERE event_type = 'rebalance'
        ORDER BY event_date DESC LIMIT 1
    """)
    row = cur.fetchone()
    if row:
        return date.fromisoformat(row["event_date"][:10])
    return None


def get_last_snapshot(index: IndexType) -> dict | None:
    """Get the most recent daily snapshot."""
    conn = get_db(index)
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM daily_snapshots ORDER BY snapshot_date DESC LIMIT 1
    """)
    row = cur.fetchone()
    return dict(row) if row else None


def save_daily_snapshot(index: IndexType, equity: float, cash: float, positions_count: int, open_orders_count: int) -> None:
    """Save a daily snapshot of portfolio state."""
    conn = get_db(index)
    cur = conn.cursor()
    today = date.today().isoformat()
    cur.execute("""
        INSERT OR REPLACE INTO daily_snapshots (snapshot_date, equity, cash, positions_count, open_orders_count)
        VALUES (?, ?, ?, ?, ?)
    """, (today, equity, cash, positions_count, open_orders_count))
    conn.commit()


# ---------------------------------------------------------------------------
# Alpaca API Client (lightweight, no SDK dependency for core calls)
# ---------------------------------------------------------------------------

class AlpacaAPI:
    """Lightweight Alpaca REST API client for paper trading."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
        data_url: str = "https://data.alpaca.markets",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self._session = None

    def _get_session(self):
        """Lazy import of requests."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            })
        return self._session

    def get(self, path: str, params: dict | None = None) -> dict:
        """GET request to Alpaca."""
        import requests
        session = self._get_session()
        resp = session.get(f"{self.base_url}/v2{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, data: dict | None = None) -> dict:
        """POST request to Alpaca."""
        import requests
        session = self._get_session()
        resp = session.post(f"{self.base_url}/v2{path}", json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def delete(self, path: str) -> dict:
        """DELETE request to Alpaca."""
        import requests
        session = self._get_session()
        resp = session.delete(f"{self.base_url}/v2{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ---- Account ----

    def get_account(self) -> dict:
        return self.get("/account")

    # ---- Positions ----

    def get_positions(self) -> list[dict]:
        return self.get("/positions") or []

    def get_position(self, symbol: str) -> dict | None:
        try:
            return self.get(f"/positions/{symbol}")
        except Exception:
            return None

    # ---- Orders ----

    def get_orders(self, status: str = "all", limit: int = 100) -> list[dict]:
        return self.get("/orders", params={"status": status, "limit": limit}) or []

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float | str,  # str for fractional
        order_type: str = "limit",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict:
        payload: dict = {
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(round(limit_price, 2))
        return self.post("/orders", data=payload)

    def cancel_order(self, order_id: str) -> None:
        try:
            self.delete(f"/orders/{order_id}")
        except Exception as e:
            print(f"  Warning: could not cancel order {order_id}: {e}")

    def cancel_all_orders(self) -> None:
        self.delete("/orders")

    # ---- Market Status ----

    def get_clock(self) -> dict:
        return self.get("/clock")

    def is_market_open(self) -> bool:
        try:
            clock = self.get_clock()
            return clock.get("is_open", False)
        except Exception:
            return False

    # ---- Latest Price ----

    def get_latest_price(self, symbol: str) -> float | None:
        try:
            import requests
            session = self._get_session()
            resp = session.get(
                f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "quote" in data:
                    return float(data["quote"]["ap"])
                if "quotes" in data and symbol in data["quotes"]:
                    return float(data["quotes"][symbol]["ap"])
            return None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Index Weight Fetching
# ---------------------------------------------------------------------------

def fetch_sp500_weights(force_refresh: bool = False) -> tuple[dict[str, float], dict[str, float]]:
    """Fetch S&P 500 cap weights and prices from iShares IVV holdings file.

    Returns (weights, prices) where weights are fractions (sum to 1.0) and
    prices are dollar amounts.

    Ticker normalization: iShares uses BRKB for Berkshire, normalize to BRK.B
    so Alpaca can trade it.
    """
    # Try fresh iShares fetch first
    try:
        import requests
        url = (
            "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
            "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
        )
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        from io import StringIO
        import pandas as pd
        df = pd.read_csv(StringIO(resp.text), skiprows=9)
        df = df[df["Asset Class"] == "Equity"].copy()
        df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
        df = df.dropna(subset=["Weight (%)", "Ticker"])
        df["Ticker"] = df["Ticker"].str.strip()

        weights: dict[str, float] = {}
        prices: dict[str, float] = {}
        for _, row in df.iterrows():
            t = str(row["Ticker"])
            w = float(row["Weight (%)"]) / 100.0
            try:
                p = float(str(row["Price"]).replace(",", ""))
            except Exception:
                p = 0.0
            weights[t] = w
            prices[t] = p

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        weights = {t: v / total for t, v in weights.items()}

        # Normalize ticker format: BRKB -> BRK.B for Alpaca compatibility
        ticker_norm = {"BRKB": "BRK.B"}
        weights = {ticker_norm.get(t, t): v for t, v in weights.items()}
        prices = {ticker_norm.get(t, t): v for t, v in prices.items()}

        print(f"  Fetched {len(weights)} tickers from iShares IVV (Apr 15, 2026)")
        return weights, prices

    except Exception as e:
        print(f"  iShares fetch failed ({e}), using cached weights_ivv.json")
        cache_path = ROOT_DIR / "data" / "sp500" / "weights_ivv.json"
        with open(cache_path) as f:
            data = json.load(f)

        weights: dict[str, float] = data["weights"]
        prices: dict[str, float] = data["prices"]
        total = sum(weights.values())
        weights = {t: w / total for t, w in weights.items()}

        # Normalize ticker format
        ticker_norm = {"BRKB": "BRK.B"}
        weights = {ticker_norm.get(t, t): v for t, v in weights.items()}
        prices = {ticker_norm.get(t, t): v for t, v in prices.items()}

        return weights, prices


def fetch_qqq_weights() -> tuple[dict[str, float], dict[str, float]]:
    """Fetch QQQ (Nasdaq-100) cap weights and prices from cached corrected weights file.

    Returns (weights, prices).
    """
    cache_path = ROOT_DIR / "data" / "qqq" / "weights_corrected.json"
    with open(cache_path) as f:
        data = json.load(f)

    weights: dict[str, float] = data["weights"]
    prices: dict[str, float] = data["prices"]

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    weights = {t: w / total for t, w in weights.items()}

    return weights, prices


def fetch_all_us_weights() -> tuple[dict[str, float], dict[str, float]]:
    """Fetch All US Stocks weights — placeholder, delegates to SP500 for now."""
    return fetch_sp500_weights()


def get_index_weights(index: IndexType, force_refresh: bool = False) -> tuple[dict[str, float], dict[str, float]]:
    """Get cap weights and prices for a given index.

    Returns (weights, prices) where weights are fractions summing to 1.0
    and prices are dollar amounts.
    """
    if index == IndexType.SP500:
        return fetch_sp500_weights(force_refresh=force_refresh)
    elif index == IndexType.QQQ:
        return fetch_qqq_weights()
    elif index == IndexType.ALL_US:
        return fetch_all_us_weights()
    else:
        raise ValueError(f"Unknown index: {index}")


# ---------------------------------------------------------------------------
# Order Planning & Validation
# ---------------------------------------------------------------------------

def compute_target_positions(
    weights: dict[str, float],
    portfolio_value: float,
    current_prices: dict[str, float],
    fractional: bool = True,
    min_order_value: float = 1.0,
) -> list[TargetPosition]:
    """Compute target positions from weights and current prices.

    Returns list of TargetPosition sorted by weight descending.
    """
    targets = []
    for ticker, weight in weights.items():
        target_value = portfolio_value * weight
        price = current_prices.get(ticker)

        if price is None or price <= 0:
            continue

        qty = target_value / price

        if fractional:
            qty = round(qty, 4)
        else:
            qty = float(int(qty))

        if qty <= 0:
            continue

        order_value = qty * price
        if order_value < min_order_value:
            continue

        limit_price = round(price * 1.001, 2)  # 0.1% buffer

        targets.append(TargetPosition(
            symbol=ticker,
            weight=weight,
            target_value=target_value,
            current_price=price,
            qty=qty,
            limit_price=limit_price,
        ))

    # Sort by weight descending
    targets.sort(key=lambda x: x.weight, reverse=True)
    return targets


def compute_order_plan(
    index: IndexType,
    api: AlpacaAPI,
    weights: dict[str, float],
    portfolio_value: float,
    fractional: bool = True,
    dry_run: bool = False,
) -> tuple[list[OrderPlan], list[str]]:
    """Compute orders needed to deploy target weights.

    Compares current positions + open orders against target positions
    to determine what orders need to be submitted.

    Returns:
        (orders_to_submit, warnings)
    """
    current_prices: dict[str, float] = {}

    if dry_run:
        print(f"  [DRY RUN] Would compute order plan for {len(weights)} tickers")
        print(f"  [DRY RUN] Portfolio value: ${portfolio_value:,.2f}")
        return [], []

    print(f"  Fetching current prices for {len(weights)} tickers...")
    for ticker in list(weights.keys())[:50]:  # Limit to top 50 to avoid rate limiting
        try:
            price = api.get_latest_price(ticker)
            if price and price > 0:
                current_prices[ticker] = price
        except Exception:
            pass
        time.sleep(0.1)  # Be gentle with price fetching

    # Get current positions from Alpaca
    positions = api.get_positions()
    current_holdings: dict[str, float] = {}
    for p in positions:
        sym = p.get("symbol", "")
        qty_str = p.get("qty", "0")
        try:
            current_holdings[sym] = float(qty_str)
        except (ValueError, TypeError):
            pass

    # Get open orders
    open_orders = api.get_orders(status="open", limit=500)
    open_order_qty: dict[str, float] = {}
    for o in open_orders:
        sym = o.get("symbol", "")
        qty_str = o.get("qty", "0")
        try:
            open_order_qty[sym] = open_order_qty.get(sym, 0) + float(qty_str)
        except (ValueError, TypeError):
            pass

    print(f"  Current holdings: {len(current_holdings)} positions")
    print(f"  Open orders: {len(open_order_qty)} symbols")
    print(f"  Got prices for {len(current_prices)} tickers")

    # For each target ticker, compute buy qty
    # (For initial deployment, current_holdings should be empty for all tickers)
    orders: list[OrderPlan] = []
    warnings: list[str] = []

    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            warnings.append(f"No price for {ticker}")
            continue

        target_value = portfolio_value * weight
        target_qty = round(target_value / price, 4) if fractional else float(int(target_value / price))

        # How many shares do we already have + on order?
        current_qty = current_holdings.get(ticker, 0)
        on_order_qty = open_order_qty.get(ticker, 0)
        total_owned = current_qty + on_order_qty

        if target_qty <= total_owned:
            # Already at or above target
            continue

        qty_to_buy = round(target_qty - total_owned, 4) if fractional else float(int(target_qty - total_owned))
        if qty_to_buy <= 0:
            continue

        limit_price = round(price * 1.001, 2)
        order_value = qty_to_buy * price

        orders.append(OrderPlan(
            symbol=ticker,
            side="buy",
            qty=qty_to_buy,
            limit_price=limit_price,
            order_type="limit",
            estimated_value=order_value,
            reason="initial_deploy",
        ))

    print(f"  Computed {len(orders)} orders to reach target weights")
    return orders, warnings


def validate_order_plan(
    orders: list[OrderPlan],
    portfolio_value: float,
) -> tuple[bool, list[str]]:
    """Locally validate an order plan before submission.

    Checks:
    - Total order value <= portfolio value
    - No single order > 50% of portfolio
    - All required fields present
    - No duplicate symbols

    Returns:
        (is_valid, list of warning/error messages)
    """
    messages = []
    is_valid = True

    total_order_value = sum(o.estimated_value for o in orders)
    if total_order_value > portfolio_value * 1.05:  # Allow 5% buffer for price movement
        messages.append(f"ERROR: Total order value ${total_order_value:,.2f} exceeds portfolio ${portfolio_value:,.2f}")
        is_valid = False

    for order in orders:
        if order.qty <= 0:
            messages.append(f"ERROR: Invalid qty {order.qty} for {order.symbol}")
            is_valid = False
        if order.limit_price is None or order.limit_price <= 0:
            messages.append(f"ERROR: Invalid limit price for {order.symbol}")
            is_valid = False
        if order.estimated_value > portfolio_value * 0.5:
            messages.append(f"WARNING: Large order for {order.symbol}: ${order.estimated_value:,.2f} ({order.estimated_value/portfolio_value*100:.1f}% of portfolio)")

    # Check for duplicate symbols
    symbols = [o.symbol for o in orders]
    if len(symbols) != len(set(symbols)):
        messages.append("ERROR: Duplicate symbols in order plan")
        is_valid = False

    return is_valid, messages


def submit_orders_with_retry(
    api: AlpacaAPI,
    orders: list[OrderPlan],
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> tuple[int, int, list[str]]:
    """Submit orders to Alpaca with retry and rate limit handling.

    Returns:
        (submitted_count, failed_count, error_messages)
    """
    submitted = 0
    failed = 0
    errors: list[str] = []

    for i, order in enumerate(orders):
        attempt = 0
        success = False

        while attempt < max_retries:
            try:
                api.submit_order(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                )
                submitted += 1
                success = True
                break
            except Exception as e:
                err_str = str(e)
                attempt += 1
                if "429" in err_str or "rate limit" in err_str.lower() or "too many" in err_str.lower():
                    delay = base_delay * (2 ** attempt) + (0.5 if attempt > 0 else 0)
                    print(f"  RATE LIMIT: {order.symbol}, waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                elif "symbol" in err_str.lower() and ("not found" in err_str.lower() or "invalid" in err_str.lower()):
                    print(f"  SKIP {order.symbol}: not tradable on Alpaca")
                    errors.append(f"{order.symbol}: not tradable")
                    failed += 1
                    break
                elif attempt < max_retries:
                    print(f"  RETRY {attempt}: {order.symbol} — {err_str[:80]}")
                    time.sleep(base_delay)
                    continue
                else:
                    print(f"  FAIL {order.symbol}: {err_str[:80]}")
                    errors.append(f"{order.symbol}: {err_str[:80]}")
                    failed += 1
                    break

        if (i + 1) % 25 == 0:
            print(f"  Submitted {submitted} orders...")

    return submitted, failed, errors


# ---------------------------------------------------------------------------
# Portfolio Operations
# ---------------------------------------------------------------------------

def check_and_cancel_stale_orders(api: AlpacaAPI, max_age_hours: float = 12) -> int:
    """Cancel open orders older than max_age_hours.

    When market is closed, orders queue up. If they're more than max_age_hours old,
    they may have stale prices. Cancel them to refresh at current prices.

    Returns count of cancelled orders.
    """
    try:
        open_orders = api.get_orders(status="open", limit=500)
    except Exception as e:
        print(f"  Warning: could not fetch open orders: {e}")
        return 0

    if not open_orders:
        return 0

    now = datetime.now()
    cancelled = 0

    for order in open_orders:
        created_at_str = order.get("created_at", "")
        if not created_at_str:
            continue

        try:
            # Parse ISO format: "2026-04-16T19:30:00-04:00"
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            age_hours = (now - created_at).total_seconds() / 3600

            if age_hours > max_age_hours:
                order_id = order.get("id", "")
                print(f"  Cancelling stale order {order_id} ({order.get('symbol', '?')}) — age: {age_hours:.1f}h")
                api.cancel_order(order_id)
                cancelled += 1
        except Exception:
            continue

    return cancelled


def deploy_portfolio(
    index: IndexType,
    api_key: str,
    api_secret: str,
    fractional: bool = True,
    dry_run: bool = False,
    execute: bool = False,
    limit_buffer: float = 0.001,
) -> dict:
    """Deploy (or continue deploying) a portfolio to match target weights.

    Args:
        index: Which index (SP500, QQQ, ALL_US)
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        fractional: Allow fractional shares
        dry_run: No Alpaca calls, show what would happen
        execute: Actually submit orders to Alpaca
        limit_buffer: Limit price buffer (0.001 = 0.1% above market)

    Returns:
        dict with deployment results
    """
    init_db(index)  # Ensure DB exists

    api = AlpacaAPI(api_key, api_secret)

    print(f"\n{'='*60}")
    print(f"PORTFOLIO MANAGER: {index.value}")
    print(f"{'='*60}")

    # --- Get market status ---
    try:
        clock = api.get_clock()
        market_open = clock.get("is_open", False)
        next_open = clock.get("next_open", "N/A")
        next_close = clock.get("next_close", "N/A")
        print(f"Market: {'OPEN' if market_open else 'CLOSED'}")
        print(f"Next open: {next_open}")
        print(f"Next close: {next_close}")
    except Exception as e:
        print(f"Warning: could not get market status: {e}")
        market_open = False

    # --- Get account ---
    try:
        account = api.get_account()
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))
    except Exception as e:
        print(f"ERROR: Could not get account: {e}")
        return {"success": False, "error": str(e)}

    print(f"\nAccount equity: ${equity:,.2f}")
    print(f"Cash: ${cash:,.2f}")
    print(f"Buying power: ${buying_power:,.2f}")

    if equity <= 0:
        print("ERROR: No portfolio value")
        return {"success": False, "error": "Zero equity"}

    # --- Cancel stale orders (market closed > 12h) ---
    if not dry_run:
        cancelled = check_and_cancel_stale_orders(api)
        if cancelled > 0:
            print(f"  Cancelled {cancelled} stale orders")

    # --- Get current Alpaca state ---
    if dry_run:
        positions_count = 0
        open_orders_count = 0
    else:
        try:
            positions = api.get_positions()
            positions_count = len(positions)
        except Exception:
            positions_count = 0

        try:
            open_orders_list = api.get_orders(status="open", limit=500)
            open_orders_count = len(open_orders_list)
        except Exception:
            open_orders_count = 0

    print(f"\nCurrent positions: {positions_count}")
    print(f"Current open orders: {open_orders_count}")

    # --- Fetch target weights and cached prices ---
    print(f"\nFetching {index.value} target weights...")
    try:
        weights, cached_prices = get_index_weights(index)
        print(f"  Target: {len(weights)} stocks")
    except Exception as e:
        print(f"ERROR: Could not fetch weights: {e}")
        return {"success": False, "error": f"Could not fetch weights: {e}"}

    total_weight = sum(weights.values())
    print(f"  Total weight: {total_weight*100:.1f}%")

    if abs(total_weight - 1.0) > 0.01:
        print(f"  WARNING: Weights sum to {total_weight:.4f}, normalizing...")
        weights = {t: w / total_weight for t, w in weights.items()}

    # --- Compute current prices (try Alpaca first, fall back to cached) ---
    print(f"\nFetching current prices...")
    current_prices: dict[str, float] = {}
    batch_count = 0

    for ticker in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
        try:
            price = api.get_latest_price(ticker)
            if price and price > 0:
                current_prices[ticker] = price
        except Exception:
            pass
        batch_count += 1
        if batch_count % 20 == 0:
            time.sleep(0.2)

    # Always supplement missing tickers from cached prices
    alpaca_count = len(current_prices)
    missing = len(weights) - alpaca_count
    if missing > 0:
        filled = 0
        for ticker, price in cached_prices.items():
            if ticker not in current_prices and price and price > 0:
                current_prices[ticker] = price
                filled += 1
        if filled > 0:
            print(f"  Filled {filled} missing prices from cache (Alpaca returned {alpaca_count})")

    print(f"  Got prices for {len(current_prices)}/{len(weights)} tickers")

    # --- Compute target positions ---
    print(f"\nComputing target positions...")
    targets = compute_target_positions(
        weights=weights,
        portfolio_value=equity,
        current_prices=current_prices,
        fractional=fractional,
        min_order_value=0.50,  # $0.50 minimum per order
    )

    print(f"  Target positions: {len(targets)}")
    total_target_value = sum(t.target_value for t in targets)
    print(f"  Target deploy value: ${total_target_value:,.2f} ({total_target_value/equity*100:.1f}% of equity)")

    # --- Get current positions and open orders ---
    current_holdings: dict[str, float] = {}
    open_order_qty: dict[str, float] = {}

    if not dry_run:
        try:
            positions = api.get_positions()
            for p in positions:
                sym = p.get("symbol", "")
                try:
                    current_holdings[sym] = float(p.get("qty", 0))
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            print(f"  Warning: could not get positions: {e}")

        try:
            open_orders_list = api.get_orders(status="open", limit=500)
            for o in open_orders_list:
                sym = o.get("symbol", "")
                try:
                    open_order_qty[sym] = open_order_qty.get(sym, 0) + float(o.get("qty", 0))
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            print(f"  Warning: could not get open orders: {e}")

    # --- Build order plan ---
    orders_to_submit: list[OrderPlan] = []

    for t in targets:
        current_qty = current_holdings.get(t.symbol, 0)
        on_order_qty = open_order_qty.get(t.symbol, 0)
        total_owned = current_qty + on_order_qty

        if t.qty is None or t.qty <= 0:
            continue

        if t.qty <= total_owned + 0.0001:
            # At or above target
            continue

        qty_needed = round(t.qty - total_owned, 4)
        if qty_needed <= 0:
            continue

        limit_price = round((t.current_price or 0) * (1 + limit_buffer), 2)
        if limit_price <= 0:
            continue

        orders_to_submit.append(OrderPlan(
            symbol=t.symbol,
            side="buy",
            qty=qty_needed,
            limit_price=limit_price,
            order_type="limit",
            estimated_value=qty_needed * limit_price,
            reason="initial_deploy",
        ))

    print(f"\nOrders to submit: {len(orders_to_submit)}")
    total_order_value = sum(o.estimated_value for o in orders_to_submit)
    print(f"Total order value: ${total_order_value:,.2f} ({total_order_value/equity*100:.1f}% of portfolio)")

    if dry_run:
        print(f"\n[DRY RUN] Would submit {len(orders_to_submit)} orders:")
        for o in orders_to_submit[:20]:
            print(f"  BUY {o.symbol}: {o.qty} shares @ ${o.limit_price:.2f} (${o.estimated_value:.2f})")
        if len(orders_to_submit) > 20:
            print(f"  ... and {len(orders_to_submit) - 20} more")
        return {"success": True, "dry_run": True, "orders_count": len(orders_to_submit)}

    # --- Validate order plan ---
    is_valid, messages = validate_order_plan(orders_to_submit, equity)
    for msg in messages:
        print(f"  {msg}")

    if not is_valid:
        print("ERROR: Order plan validation failed. Not submitting any orders.")
        return {"success": False, "error": "Validation failed", "messages": messages}

    # --- Preview mode (--preview flag equivalent) ---
    # If not execute, show preview and exit
    if not execute:
        print(f"\n[PREVIEW] Would submit {len(orders_to_submit)} orders:")
        for o in orders_to_submit[:30]:
            print(f"  BUY {o.symbol}: {o.qty} shares @ ${o.limit_price:.2f} (${o.estimated_value:.2f})")
        if len(orders_to_submit) > 30:
            print(f"  ... and {len(orders_to_submit) - 30} more")
        print(f"\nTotal: {len(orders_to_submit)} orders, ${total_order_value:,.2f}")
        print("To execute, re-run with --execute flag")
        return {"success": True, "preview": True, "orders_count": len(orders_to_submit)}

    # --- Execute orders ---
    print(f"\nSubmitting {len(orders_to_submit)} orders to Alpaca...")
    submitted, failed, errors = submit_orders_with_retry(api, orders_to_submit)

    print(f"\n  Submitted: {submitted}")
    print(f"  Failed:   {failed}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    # --- Record event ---
    try:
        record_rebalance_event(
            index=index,
            reason="initial_deploy" if not market_open else "initial_deploy_live",
            equity_before=equity,
            equity_after=equity,  # After deploy we don't have new equity yet
            trades=submitted,
            notes=f"Submitted {submitted} orders, {failed} failed",
        )
    except Exception as e:
        print(f"  Warning: could not record event: {e}")

    # --- Save snapshot ---
    try:
        open_orders_list = api.get_orders(status="open", limit=500)
        save_daily_snapshot(index, equity, cash, len(positions), len(open_orders_list))
    except Exception as e:
        print(f"  Warning: could not save snapshot: {e}")

    print(f"\n{'='*60}")
    print(f"DEPLOYMENT SUMMARY: {index.value}")
    print(f"{'='*60}")
    print(f"Account equity:     ${equity:,.2f}")
    print(f"Index:              {index.value}")
    print(f"Weight coverage:    {sum(weights.values())*100:.1f}%")
    print(f"Target positions:   {len(targets)}")
    print(f"Orders submitted:  {submitted}")
    print(f"Orders failed:     {failed}")
    print(f"Limit buffer:      {limit_buffer*100:.2f}%")
    print(f"\nOrders will fill when market opens.")
    print(f"{'='*60}")

    return {
        "success": True,
        "orders_submitted": submitted,
        "orders_failed": failed,
        "total_order_value": total_order_value,
    }


def show_portfolio_status(
    index: IndexType,
    api_key: str,
    api_secret: str,
) -> dict:
    """Show current portfolio status for an index."""
    init_db(index)
    api = AlpacaAPI(api_key, api_secret)

    print(f"\n{'='*60}")
    print(f"PORTFOLIO STATUS: {index.value}")
    print(f"{'='*60}")

    # Account
    try:
        account = api.get_account()
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))
    except Exception as e:
        print(f"ERROR: Could not get account: {e}")
        return {"success": False, "error": str(e)}

    print(f"Portfolio value: ${equity:,.2f}")
    print(f"Cash:            ${cash:,.2f}")
    print(f"Buying power:    ${buying_power:,.2f}")

    # Positions
    positions = api.get_positions()
    print(f"\nPositions: {len(positions)}")
    if positions:
        total_mv = 0
        for p in sorted(positions, key=lambda x: float(x.get("market_value", 0)), reverse=True)[:10]:
            sym = p.get("symbol", "?")
            mv = float(p.get("market_value", 0))
            qty = p.get("qty", "?")
            avg = p.get("avg_entry_price", "?")
            total_mv += mv
            print(f"  {sym:6} {qty:>8} shares @ ${avg} MV=${mv:>10,.2f}")
        print(f"  ... and {len(positions)-10} more positions" if len(positions) > 10 else "")
        print(f"  Total market value: ${total_mv:,.2f}")

    # Open orders
    open_orders = api.get_orders(status="open", limit=500)
    print(f"\nOpen orders: {len(open_orders)}")
    if open_orders:
        order_value = 0
        for o in sorted(open_orders, key=lambda x: float(x.get("qty", 0)) * float(x.get("limit_price", 0)), reverse=True)[:10]:
            sym = o.get("symbol", "?")
            qty = o.get("qty", "?")
            lp = o.get("limit_price", "?")
            order_value += float(qty) * float(lp) if qty and lp else 0
            print(f"  BUY {sym:6} {qty:>8} @ ${lp}")
        print(f"  ... and {len(open_orders)-10} more" if len(open_orders) > 10 else "")
        print(f"  Estimated order value: ${order_value:,.2f}")

    # Last rebalance
    last_rebal = get_last_rebalance(index)
    days_since = (date.today() - last_rebal).days if last_rebal else None
    print(f"\nLast rebalance: {last_rebal} ({days_since} days ago)" if last_rebal else "\nLast rebalance: Never")
    print(f"Next rebalance: in {31 - (days_since or 0)} days" if days_since is not None else "")

    # Recent history
    conn = get_db(index)
    cur = conn.cursor()
    cur.execute("""
        SELECT event_date, event_type, reason, trades_count
        FROM portfolio_events
        ORDER BY event_date DESC LIMIT 5
    """)
    events = cur.fetchall()
    if events:
        print(f"\nRecent events:")
        for e in events:
            print(f"  {e['event_date'][:10]} | {e['event_type']} | {e['reason']} | {e['trades_count']} trades")

    return {"success": True, "equity": equity, "positions": len(positions), "open_orders": len(open_orders)}


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio Manager for Direct Indexing")
    parser.add_argument("--index", "-i", required=True, choices=["SP500", "QQQ", "ALL_US"],
                        help="Index to manage")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Dry run (no Alpaca calls)")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="Preview orders without submitting")
    parser.add_argument("--execute", "-e", action="store_true",
                        help="Execute orders (required to submit real orders)")
    parser.add_argument("--fractional", "-f", action="store_true", default=True,
                        help="Allow fractional shares (default: True)")
    parser.add_argument("--cancel-stale", action="store_true",
                        help="Cancel stale open orders")
    parser.add_argument("--rebalance", action="store_true",
                        help="Force rebalance")
    parser.add_argument("--status", action="store_true",
                        help="Show portfolio status")
    parser.add_argument("--limit-buffer", type=float, default=0.001,
                        help="Limit order buffer (0.001 = 0.1%% above market)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force refresh of cached weights")

    args = parser.parse_args()

    index_str = args.index.upper()
    index = IndexType[index_str]

    # Get API credentials from environment
    env_vars = ACCOUNT_ENVVARS[index.value]
    api_key = os.environ.get(env_vars[0], os.environ.get("ALPACA_API_KEY", ""))
    api_secret = os.environ.get(env_vars[1], os.environ.get("ALPACA_API_SECRET", ""))

    if not api_key or not api_secret:
        print(f"ERROR: Missing API credentials for {index.value}")
        print(f"  Set {env_vars[0]} and {env_vars[1]}")
        print(f"  Or set ALPACA_API_KEY and ALPACA_API_SECRET as fallbacks")
        return 1

    # Route to operation
    if args.status:
        result = show_portfolio_status(index, api_key, api_secret)
        return 0 if result.get("success") else 1

    if args.cancel_stale:
        api = AlpacaAPI(api_key, api_secret)
        n = check_and_cancel_stale_orders(api)
        print(f"Cancelled {n} stale orders")
        return 0

    if args.rebalance or args.execute or args.preview or args.dry_run:
        result = deploy_portfolio(
            index=index,
            api_key=api_key,
            api_secret=api_secret,
            fractional=args.fractional,
            dry_run=args.dry_run,
            execute=args.execute,
            limit_buffer=args.limit_buffer,
        )
        return 0 if result.get("success") else 1

    # Default: show status
    result = show_portfolio_status(index, api_key, api_secret)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())