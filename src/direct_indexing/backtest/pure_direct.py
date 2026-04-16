"""
Pure Direct Indexing — backtest module.

No tax-loss harvesting. Rebalance to S&P 500 cap weights every 31 days.
Goal: replicate S&P 500 performance with less tax drag than annual rebalancing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd


# ---------------------------------------------------------------------------
# S&P 500 top holdings (approximate weights as of ~2024)
# ---------------------------------------------------------------------------
SP500_TOP_HOLDINGS = {
    "AAPL": 0.0720,
    "MSFT": 0.0630,
    "NVDA": 0.0280,
    "AMZN": 0.0220,
    "GOOGL": 0.0220,
    "GOOG": 0.0200,
    "META": 0.0210,
    "LLY": 0.0140,
    "AVGO": 0.0130,
    "JPM": 0.0130,
    "BRK-B": 0.0160,
    "V": 0.0100,
    "XOM": 0.0100,
    "UNH": 0.0090,
    "MA": 0.0090,
    "JNJ": 0.0090,
    "PG": 0.0080,
    "HD": 0.0080,
    "CVX": 0.0080,
    "ABBV": 0.0070,
    "MRK": 0.0070,
    "COST": 0.0060,
    "PEP": 0.0060,
    "KO": 0.0050,
    "CRM": 0.0050,
    "WMT": 0.0050,
    "BAC": 0.0050,
    "TMO": 0.0040,
    "MCD": 0.0040,
    "CSCO": 0.0040,
}

# Normalize to sum to 1.0
_total = sum(SP500_TOP_HOLDINGS.values())
SP500_WEIGHTS = {t: w / _total for t, w in SP500_TOP_HOLDINGS.items()}
ALL_TICKERS = list(SP500_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PureDIResult:
    """Result of a Pure Direct Indexing backtest."""
    start_date: date
    end_date: date
    initial_portfolio: float
    strategy_return_percent: float = 0.0
    benchmark_return_percent: float = 0.0
    alpha: float = 0.0
    num_rebalances: int = 0
    total_trades: int = 0
    max_drift_percent: float = 0.0
    final_portfolio: float = 0.0
    final_benchmark: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.start_date} → {self.end_date}: "
            f"Strategy {self.strategy_return_percent:+.1f}% | "
            f"Benchmark (VOO) {self.benchmark_return_percent:+.1f}% | "
            f"Alpha {self.alpha:+.1f}% | "
            f"Rebalances: {self.num_rebalances} | Trades: {self.total_trades}"
        )


# ---------------------------------------------------------------------------
# Price fetcher
# ---------------------------------------------------------------------------

def _extract_close(df: pd.DataFrame) -> pd.Series:
    """Extract close price series from yfinance DataFrame."""
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # Single ticker: first column
        close = close.iloc[:, 0]
    return close


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PureDirectIndexer:
    """Pure direct indexing: cap-weighted, rebalanced every 31 days, no TLH."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/backtest")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._prices: dict[str, dict[str, float]] = {}

    async def run(
        self,
        start: date | str,
        end: date | str,
        initial: float = 100_000.0,
        rebalance_days: int = 31,
    ) -> PureDIResult:
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        result = PureDIResult(start_date=start, end_date=end, initial_portfolio=initial)

        # Build ticker list (portfolio stocks + VOO benchmark)
        tickers_to_fetch = ALL_TICKERS + ["VOO"]

        # Fetch prices
        print(f"Fetching {len(tickers_to_fetch)} securities...")
        self._fetch_prices(tickers_to_fetch, start, end)
        print(f"Loaded {len(self._prices)} securities")

        # Initialize portfolio
        portfolio: dict[str, dict] = {}
        for ticker, weight in SP500_WEIGHTS.items():
            price = self._get_first_price(ticker, start)
            if price is None:
                continue
            shares = (initial * weight) / price
            portfolio[ticker] = {"shares": shares, "weight": weight}

        # Initialize VOO benchmark
        voo_price = self._get_first_price("VOO", start)
        if voo_price is None:
            raise ValueError("Could not get VOO price")
        voo_shares = initial / voo_price
        print(f"VOO initial price: ${voo_price:.2f}")

        # Backtest loop
        current = start
        last_rebalance = start
        num_rebalances = 0
        total_trades = 0
        max_drift = 0.0

        print(f"Backtesting {start} → {end}...")

        while current <= end:
            if current.weekday() < 5:  # Trading day
                # Portfolio value
                total_value = 0.0
                for ticker, p in portfolio.items():
                    price = self._get_price(ticker, current)
                    if price is not None:
                        total_value += p["shares"] * price

                # Track max drift
                for ticker, p in portfolio.items():
                    price = self._get_price(ticker, current)
                    if price is None or total_value == 0:
                        continue
                    current_weight = (p["shares"] * price) / total_value
                    drift = abs(current_weight - p["weight"])
                    max_drift = max(max_drift, drift * 100)

                # Rebalance check
                if (current - last_rebalance).days >= rebalance_days and total_value > 0:
                    trades = self._do_rebalance(portfolio, total_value, current)
                    total_trades += len(trades)
                    num_rebalances += 1
                    last_rebalance = current

            current += timedelta(days=1)

        # Final values
        voo_final = self._get_price("VOO", end) or 0
        result.final_benchmark = voo_shares * voo_final
        result.benchmark_return_percent = (
            (result.final_benchmark - initial) / initial * 100
        )

        final_value = 0.0
        for ticker, p in portfolio.items():
            price = self._get_price(ticker, end)
            if price is not None:
                final_value += p["shares"] * price

        result.final_portfolio = final_value
        result.strategy_return_percent = (
            (final_value - initial) / initial * 100
        )
        result.alpha = result.strategy_return_percent - result.benchmark_return_percent
        result.num_rebalances = num_rebalances
        result.total_trades = total_trades
        result.max_drift_percent = max_drift

        print(f"\n{result.summary()}")
        return result

    def _get_price(self, ticker: str, as_of: date) -> float | None:
        """Get closing price on or before as_of date."""
        if ticker not in self._prices:
            return None
        date_str = as_of.isoformat()
        candidates = {d: p for d, p in self._prices[ticker].items() if d <= date_str}
        if not candidates:
            return None
        return candidates[max(candidates.keys())]

    def _get_first_price(self, ticker: str, on_or_after: date) -> float | None:
        """Get first available price on or after on_or_after."""
        if ticker not in self._prices:
            return None
        date_str = on_or_after.isoformat()
        candidates = {d: p for d, p in self._prices[ticker].items() if d >= date_str}
        if not candidates:
            return None
        return candidates[min(candidates.keys())]

    def _fetch_prices(self, tickers: list[str], start: date, end: date) -> None:
        """Fetch prices for all tickers, using cache when available."""
        for ticker in tickers:
            cache_file = self.cache_dir / f"pdi_{ticker.replace('-', '_')}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    self._prices[ticker] = json.load(f)
                    continue

            try:
                df: pd.DataFrame = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=(end + timedelta(days=5)).isoformat(),
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    print(f"  Warning: {ticker} returned no data")
                    continue

                close = _extract_close(df)
                prices = {}
                for dt, val in close.items():
                    if pd.notna(val):
                        d = pd.Timestamp(dt).date().isoformat()
                        prices[d] = float(val)

                if prices:
                    self._prices[ticker] = prices
                    with open(cache_file, "w") as f:
                        json.dump(prices, f)
            except Exception as e:
                print(f"  Warning: {ticker} failed: {e}")

    def _do_rebalance(
        self,
        portfolio: dict,
        total_value: float,
        as_of: date,
    ) -> list[tuple[str, float, float]]:
        """Rebalance to target weights. Returns list of (ticker, old_shares, new_shares)."""
        trades = []
        for ticker, weight in SP500_WEIGHTS.items():
            price = self._get_price(ticker, as_of)
            if price is None:
                continue

            target_shares = (total_value * weight) / price
            old_shares = portfolio.get(ticker, {}).get("shares", 0.0)
            portfolio[ticker] = {"shares": target_shares, "weight": weight}

            if abs(old_shares - target_shares) > 0.001:
                trades.append((ticker, old_shares, target_shares))

        return trades


async def run_backtest(
    start: date | str,
    end: date | str,
    initial: float = 100_000.0,
    rebalance_days: int = 31,
) -> PureDIResult:
    indexer = PureDirectIndexer()
    return await indexer.run(start, end, initial, rebalance_days)


if __name__ == "__main__":
    import sys
    start_str = sys.argv[1] if len(sys.argv) > 1 else "2021-01-01"
    end_str = sys.argv[2] if len(sys.argv) > 2 else "2023-12-31"
    initial = float(sys.argv[3]) if len(sys.argv) > 3 else 100_000.0

    result = asyncio.run(run_backtest(start_str, end_str, initial))

    print("\n=== Pure Direct Indexing Backtest ===")
    print(f"Period:           {result.start_date} → {result.end_date}")
    print(f"Initial:          ${result.initial_portfolio:,.2f}")
    print(f"Final Portfolio:  ${result.final_portfolio:,.2f}")
    print(f"Final Benchmark:  ${result.final_benchmark:,.2f}")
    print(f"Strategy Return:  {result.strategy_return_percent:+.2f}%")
    print(f"Benchmark (VOO): {result.benchmark_return_percent:+.2f}%")
    print(f"Alpha:            {result.alpha:+.2f}%")
    print(f"Rebalances:       {result.num_rebalances}")
    print(f"Total Trades:     {result.total_trades}")
    print(f"Max Drift:        {result.max_drift_percent:.1f}%")
