"""
Tax-Loss Harvesting Substitute Finder.

For each S&P 500 ticker, finds a substitute ticker from the same GICS
sub-industry that can replace it during the wash sale window.

Substitutes are chosen by:
1. Same GICS sub-industry (more precise than sector)
2. Highest correlation over trailing 252 trading days (>0.90 required)
3. Not the same ticker obviously
4. In the S&P 500 universe

The substitute map is computed once and cached, refreshed monthly.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from .sp500 import get_sp500, SP500Data


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubstituteCandidate:
    """A potential TLH substitute."""
    ticker: str
    original_ticker: str
    correlation: float
    sub_industry: str
    correlation_source: str  # "252d" | "sector_avg"


# ---------------------------------------------------------------------------
# Substitute Finder
# ---------------------------------------------------------------------------

class SubstituteFinder:
    """
    Finds and caches TLH substitute tickers.

    Strategy:
    1. Group all S&P 500 tickers by GICS sub-industry
    2. For each sub-industry group, compute pairwise price correlations
    3. For each ticker, select the highest-correlation peer in same sub-industry
       that has correlation > 0.90
    4. Store as: {original_ticker: {substitute: ticker, correlation: x, sub_industry: y}}
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/substitutes")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "substitute_map.json"
        self._sub_map: dict[str, dict] = {}  # {original: {substitute, correlation, sub_industry}}
        self._sp500 = get_sp500()
        self._last_refresh: Optional[date] = None

    def get_substitute(self, ticker: str) -> Optional[str]:
        """Get the substitute ticker for a given ticker."""
        self._ensure_loaded()
        entry = self._sub_map.get(ticker)
        return entry.get("substitute") if entry else None

    def get_substitute_info(self, ticker: str) -> Optional[SubstituteCandidate]:
        """Get full substitute info for a ticker."""
        self._ensure_loaded()
        entry = self._sub_map.get(ticker)
        if not entry:
            return None
        return SubstituteCandidate(
            ticker=entry["substitute"],
            original_ticker=ticker,
            correlation=entry["correlation"],
            sub_industry=entry["sub_industry"],
            correlation_source=entry.get("source", "252d"),
        )

    def _ensure_loaded(self) -> None:
        """Load from cache if fresh, otherwise compute."""
        today = date.today()
        if self._sub_map and self._last_refresh == today:
            return

        # Check if cache exists and is less than 30 days old
        if self._cache_file.exists():
            with open(self._cache_file) as f:
                cache = json.load(f)
            cached_date = date.fromisoformat(cache.get("computed_date", "2000-01-01"))
            if (today - cached_date).days < 30:
                self._sub_map = cache.get("substitutes", {})
                self._last_refresh = today
                return

        self._compute_and_cache()

    def _compute_and_cache(self) -> None:
        """Compute substitute map and save to cache."""
        print("Computing TLH substitute map...")
        sub_industry_groups = self._sp500.get_sub_industry_groups()
        tickers = self._sp500.get_constituents()

        # Download 252-day price history for all tickers
        print(f"Fetching 252-day price history for {len(tickers)} tickers...")
        prices = self._fetch_prices(tickers, days=252)

        if not prices:
            print("Warning: no price data fetched, using sector-based fallback")
            self._sub_map = self._fallback_substitutes()
            self._save_cache()
            return

        # Compute correlation matrix
        print("Computing correlations...")
        returns = self._compute_returns(prices)

        # For each sub-industry, compute intra-group correlations
        substitutes = {}
        for sub_industry, group_tickers in sub_industry_groups.items():
            if len(group_tickers) < 2:
                continue
            # Filter to tickers we have price data for
            group = [t for t in group_tickers if t in returns.columns]
            if len(group) < 2:
                continue

            # Compute pairwise correlations
            corr_matrix = returns[group].corr()

            for ticker in group:
                # Find best peer (excluding self)
                row = corr_matrix[ticker].drop(ticker)
                if row.empty:
                    continue
                best_peer = row.idxmax()
                best_corr = row.max()

                if best_corr >= 0.90:
                    substitutes[ticker] = {
                        "substitute": best_peer,
                        "correlation": round(float(best_corr), 4),
                        "sub_industry": sub_industry,
                        "source": "252d",
                    }

        # Fill gaps with sector-level fallback
        filled = self._fill_gaps(substitutes, sub_industry_groups)
        self._sub_map = filled
        self._last_refresh = date.today()
        self._save_cache()
        print(f"Substitute map: {len(substitutes)} high-corr pairs, "
              f"{len(filled) - len(substitutes)} filled from sector fallback")

    def _fetch_prices(self, tickers: list[str], days: int) -> pd.DataFrame:
        """Fetch daily close prices for tickers."""
        end = date.today()
        start = end - timedelta(days=days + 30)  # buffer for lookback

        # Batch fetch via yfinance
        tickers_valid = [t for t in tickers if t]  # remove empty
        try:
            df = yf.download(
                tickers_valid,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=True,
                auto_adjust=True,
                threads=True,
            )
            if df.empty:
                return pd.DataFrame()
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                return close.dropna(how="all")
            return pd.DataFrame()
        except Exception as e:
            print(f"Warning: price fetch failed: {e}")
            return pd.DataFrame()

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute daily returns from prices."""
        return prices.pct_change().dropna()

    def _fallback_substitutes(self) -> dict[str, dict]:
        """
        When correlation data isn't available, fall back to same GICS sector
        with largest market cap (weight).
        """
        sp500 = self._sp500
        sectors = {}
        for ticker, (sector, sub) in sp500.get_sector_map().items():
            if sector not in sectors:
                sectors[sector] = {}
            sectors[sector][ticker] = sp500.get_weights().get(ticker, 0.0)

        sub_map = {}
        for ticker, (sector, sub) in sp500.get_sector_map().items():
            peers = [t for t in sectors.get(sector, {}) if t != ticker]
            if not peers:
                continue
            best_peer = max(peers, key=lambda t: sectors[sector][t])
            sub_map[ticker] = {
                "substitute": best_peer,
                "correlation": 0.85,  # conservative estimate
                "sub_industry": sub,
                "source": "sector_cap",
            }
        return sub_map

    def _fill_gaps(
        self,
        substitutes: dict,
        sub_industry_groups: dict[str, list[str]],
    ) -> dict[str, dict]:
        """Fill gaps in substitutes using sector-level fallback."""
        all_tickers = set(self._sp500.get_constituents())
        filled = dict(substitutes)

        for ticker in all_tickers:
            if ticker in filled:
                continue
            # Find ticker in sub_industry_groups
            for sub, group in sub_industry_groups.items():
                if ticker in group:
                    peers = [t for t in group if t != ticker and t in substitutes]
                    if peers:
                        # Use any peer as substitute (correlation unknown)
                        filled[ticker] = {
                            "substitute": peers[0],
                            "correlation": 0.80,  # conservative
                            "sub_industry": sub,
                            "source": "sub_industry_any",
                        }
                    break

        return filled

    def _save_cache(self) -> None:
        """Save computed map to cache."""
        cache = {
            "computed_date": date.today().isoformat(),
            "substitutes": self._sub_map,
        }
        with open(self._cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    def get_map(self) -> dict[str, dict]:
        """Get the full substitute map."""
        self._ensure_loaded()
        return self._sub_map.copy()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[SubstituteFinder] = None

def get_substitute_finder() -> SubstituteFinder:
    """Get the singleton SubstituteFinder instance."""
    global _instance
    if _instance is None:
        _instance = SubstituteFinder()
    return _instance
