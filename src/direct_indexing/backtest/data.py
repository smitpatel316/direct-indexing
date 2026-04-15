"""
Backtest data management: S&P 500 composition + historical prices.

Fetches and caches:
1. S&P 500 composition by date (fja05680/sp500 GitHub repo)
2. Historical price data via yfinance

Usage:
    data_mgr = BacktestDataManager(cache_dir=Path("data/backtest"))
    await data_mgr.load_composition()
    prices = await data_mgr.get_prices(
        ["AAPL", "MSFT"], start="2019-01-01", end="2024-12-31"
    )
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import yfinance as yf

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SP500Composition:
    """S&P 500 composition snapshot for a single date."""
    date: date
    tickers: list[str]


@dataclass
class PriceHistory:
    """Historical price data for a set of tickers."""
    prices: dict[str, dict[str, float]]  # ticker -> {date -> close_price}
    start_date: date
    end_date: date

    def get_price(self, ticker: str, as_of: date) -> float | None:
        """Get the closing price for ticker on as_of date."""
        date_str = as_of.isoformat()
        return self.prices.get(ticker, {}).get(date_str)


# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------


@dataclass
class BacktestDataManager:
    """Manages backtest data: S&P 500 composition + yfinance prices."""

    cache_dir: Path = field(default_factory=lambda: Path("data/backtest"))
    _composition: dict[str, list[str]] = field(default_factory=dict)
    _prices: dict[str, dict[str, float]] = field(default_factory=dict)

    COMPOSITION_URL = (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv"
    )

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._comp_cache = self.cache_dir / "sp500_composition.json"
        self._price_cache = self.cache_dir / "prices"

    # -------------------------------------------------------------------------
    # S&P 500 Composition
    # -------------------------------------------------------------------------

    async def load_composition(self, force_refresh: bool = False) -> None:
        """Load S&P 500 composition data from GitHub or cache.

        The composition CSV has format: date,tickers (comma-separated ticker list)
        """
        if self._composition and not force_refresh:
            return

        # Try cache first
        if self._comp_cache.exists() and not force_refresh:
            with open(self._comp_cache) as f:
                raw = json.load(f)
            self._composition = raw
            return

        # Fetch from GitHub
        import urllib.request

        print("Fetching S&P 500 composition data from GitHub...")
        req = urllib.request.Request(
            self.COMPOSITION_URL,
            headers={"Accept": "text/csv"},
        )
        with urllib.request.urlopen(req) as resp:
            content = resp.read().decode("utf-8")

        # Parse CSV: date,tickers
        reader = csv.reader(StringIO(content))
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0].strip()
            tickers_str = row[1].strip()
            if date_str and tickers_str:
                tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
                self._composition[date_str] = tickers

        # Save to cache
        with open(self._comp_cache, "w") as f:
            json.dump(self._composition, f)

        print(f"Loaded {len(self._composition)} composition records")

    def get_tickers_for_date(self, as_of: date | str) -> list[str]:
        """Return the S&P 500 tickers that were in the index on as_of date."""
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        date_str = as_of.isoformat()

        # Exact match
        if date_str in self._composition:
            return self._composition[date_str]

        candidates = sorted(
            [d for d in self._composition if d <= date_str],
            key=lambda d: datetime.fromisoformat(d),
            reverse=True,
        )
        if candidates:
            return self._composition[candidates[0]]

        return []

    def get_composition_range(
        self,
        start: date,
        end: date,
    ) -> dict[str, list[str]]:
        """Return composition snapshots for all dates in [start, end] range."""
        result = {}
        current = start
        while current <= end:
            date_str = current.isoformat()
            if date_str in self._composition:
                result[date_str] = self._composition[date_str]
            current = self._next_trading_day(current)
        return result

    @staticmethod
    def _next_trading_day(d: date) -> date:
        """Return the next trading day (skip weekends)."""
        from datetime import timedelta
        next_d = d + timedelta(days=1)
        while next_d.weekday() >= 5:  # Sat=5, Sun=6
            next_d += timedelta(days=1)
        return next_d

    # -------------------------------------------------------------------------
    # Historical prices via yfinance
    # -------------------------------------------------------------------------

    async def get_prices(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
        force_refresh: bool = False,
    ) -> PriceHistory:
        """Fetch historical close prices for tickers via yfinance.

        Results are cached per ticker to avoid re-downloading.
        """
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        all_prices: dict[str, dict[str, float]] = {}
        tickers_to_fetch: list[str] = []

        # Check cache for each ticker
        for ticker in tickers:
            ticker_cache = self._price_cache / f"{ticker}.json"
            if ticker_cache.exists() and not force_refresh:
                with open(ticker_cache) as f:
                    cached = json.load(f)
                    # Check if cache covers our date range
                    dates = sorted(cached.keys())
                    if dates:
                        cached_start = dates[0]
                        cached_end = dates[-1]
                        in_range = (
                            cached_start <= start.isoformat()
                            and cached_end >= end.isoformat()
                        )
                        if in_range:
                            all_prices[ticker] = cached
                            continue
            tickers_to_fetch.append(ticker)

        if tickers_to_fetch:
            print(f"Fetching {len(tickers_to_fetch)} ticker(s) from yfinance...")
            # Batch download — yfinance handles this efficiently
            for ticker in tickers_to_fetch:
                await self._fetch_ticker_prices(ticker, start, end)
                all_prices[ticker] = self._prices.get(ticker, {})

        # Return merged
        return PriceHistory(
            prices=all_prices,
            start_date=start,
            end_date=end,
        )

    async def _fetch_ticker_prices(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> None:
        """Fetch and cache prices for a single ticker."""
        ticker_cache = self._price_cache / f"{ticker}.json"
        self._price_cache.mkdir(parents=True, exist_ok=True)

        # Check existing cache
        existing: dict[str, float] = {}
        if ticker_cache.exists():
            with open(ticker_cache) as f:
                existing = json.load(f)

        # Fetch new data from yfinance
        try:
            import pandas as pd

            # Use module-level download (correct for yfinance >= 0.2)
            df: pd.DataFrame = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return

            close_col = df["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            new_prices: dict[str, float] = {
                pd.Timestamp(d).date().isoformat(): float(v)
                for d, v in close_col.items()
                if pd.notna(v)
            }

            # Merge with existing (existing takes precedence for overlap)
            merged = {**new_prices, **existing}
            self._prices[ticker] = merged

            # Save to cache
            with open(ticker_cache, "w") as f:
                json.dump(merged, f)
        except Exception as e:
            print(f"  Warning: failed to fetch {ticker}: {e}")
            # Fall back to existing cache
            if existing:
                self._prices[ticker] = existing

    # -------------------------------------------------------------------------
    # Convenience: get prices for all S&P 500 tickers in a date range
    # -------------------------------------------------------------------------

    async def get_prices_for_composition(
        self,
        start: date,
        end: date,
        sample_tickers: int = 0,
    ) -> PriceHistory:
        """Get prices for all tickers in S&P 500 during [start, end].

        Args:
            start: Start date
            end: End date
            sample_tickers: If > 0, only fetch this many randomly sampled tickers
                           (for quick testing without fetching all 500)

        Returns:
            PriceHistory with prices for all or sampled tickers.
        """
        import random

        composition = self.get_composition_range(start, end)
        all_tickers: set[str] = set()
        for tickers in composition.values():
            all_tickers.update(tickers)

        ticker_list = list(all_tickers)
        if sample_tickers > 0 and sample_tickers < len(ticker_list):
            ticker_list = random.sample(ticker_list, sample_tickers)

        return await self.get_prices(ticker_list, start, end)

    # -------------------------------------------------------------------------
    # S&P 500 returns (benchmark)
    # -------------------------------------------------------------------------

    async def get_spy_prices(
        self,
        start: date | str,
        end: date | str,
    ) -> PriceHistory:
        """Get SPY historical prices as benchmark."""
        return await self.get_prices(["SPY"], start, end)
