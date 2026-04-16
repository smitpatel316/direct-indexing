"""
S&P 500 Constituent Data and Weights.

Sources:
- Current constituents: datasets/s-and-p-500-companies (GitHub)
- Historical composition: fja05680/sp500 (GitHub)
- Current weights: estimated from market cap data

This module provides:
- get_constituents() — current S&P 500 ticker list
- get_weights() — current cap weights (approximated)
- get_sector_mapping() — GICS sector/sub-industry per ticker
- get_historical_tickers() — tickers for a given date (from fja05680)
"""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Constituent:
    """A single S&P 500 constituent."""
    ticker: str
    name: str
    sector: str          # GICS Sector
    sub_industry: str    # GICS Sub-Industry
    weight: float = 0.0  # Cap weight (fraction, not percent)
    shares: float = 0.0  # Shares outstanding (for reference)


# ---------------------------------------------------------------------------
# Static current constituents (top ~50 by weight, enough for most use cases)
# Full 503-ticker list fetched dynamically from GitHub on first load
# ---------------------------------------------------------------------------

# Top ~50 S&P 500 constituents by weight (approximate, as of 2024)
# These cover ~60% of index market cap by weight
CURRENT_TOP_CONSTITUENTS: dict[str, dict] = {
    "AAPL":  {"name": "Apple Inc.",                  "sector": "Information Technology", "sub_industry": "Technology Hardware Storage",         "weight": 0.0720},
    "MSFT":  {"name": "Microsoft Corp.",              "sector": "Information Technology", "sub_industry": "Systems Software",                   "weight": 0.0630},
    "NVDA":  {"name": "NVIDIA Corp.",                "sector": "Information Technology", "sub_industry": "Semiconductors",                      "weight": 0.0280},
    "AMZN":  {"name": "Amazon.com Inc.",             "sector": "Consumer Discretionary", "sub_industry": "Broadline Retail",                    "weight": 0.0220},
    "GOOGL": {"name": "Alphabet Inc. CL A",          "sector": "Communication Services", "sub_industry": "Interactive Media",                   "weight": 0.0220},
    "GOOG":  {"name": "Alphabet Inc. CL C",          "sector": "Communication Services", "sub_industry": "Interactive Media",                   "weight": 0.0200},
    "META":  {"name": "Meta Platforms Inc.",         "sector": "Communication Services", "sub_industry": "Interactive Media",                   "weight": 0.0210},
    "LLY":   {"name": "Eli Lilly and Co.",           "sector": "Health Care",            "sub_industry": "Pharmaceuticals",                     "weight": 0.0140},
    "AVGO":  {"name": "Broadcom Inc.",               "sector": "Information Technology", "sub_industry": "Semiconductors",                      "weight": 0.0130},
    "JPM":   {"name": "JPMorgan Chase & Co.",        "sector": "Financials",             "sub_industry": "Diversified Banks",                    "weight": 0.0130},
    "BRK-B": {"name": "Berkshire Hathaway CL B",    "sector": "Financials",             "sub_industry": "Multi-Sector Holdings",               "weight": 0.0160},
    "V":     {"name": "Visa Inc.",                   "sector": "Financials",             "sub_industry": "Transaction Processing",              "weight": 0.0100},
    "XOM":   {"name": "Exxon Mobil Corp.",           "sector": "Energy",                 "sub_industry": "Integrated Oil & Gas",                 "weight": 0.0100},
    "UNH":   {"name": "UnitedHealth Group Inc.",     "sector": "Health Care",            "sub_industry": "Managed Health Care",                 "weight": 0.0090},
    "MA":   {"name": "Mastercard Inc.",             "sector": "Financials",             "sub_industry": "Transaction Processing",              "weight": 0.0090},
    "JNJ":   {"name": "Johnson & Johnson",           "sector": "Health Care",            "sub_industry": "Pharmaceuticals",                     "weight": 0.0090},
    "PG":    {"name": "Procter & Gamble Co.",         "sector": "Consumer Staples",       "sub_industry": "Household Products",                  "weight": 0.0080},
    "HD":    {"name": "Home Depot Inc.",              "sector": "Consumer Discretionary", "sub_industry": "Home Improvement Retail",             "weight": 0.0080},
    "CVX":   {"name": "Chevron Corp.",               "sector": "Energy",                  "sub_industry": "Integrated Oil & Gas",                 "weight": 0.0080},
    "ABBV":  {"name": "AbbVie Inc.",                 "sector": "Health Care",            "sub_industry": "Biotechnology",                       "weight": 0.0070},
    "MRK":   {"name": "Merck & Co. Inc.",            "sector": "Health Care",            "sub_industry": "Pharmaceuticals",                      "weight": 0.0070},
    "COST":  {"name": "Costco Wholesale Corp.",      "sector": "Consumer Staples",       "sub_industry": "Consumer Staples Merchandising",      "weight": 0.0060},
    "PEP":   {"name": "PepsiCo Inc.",                "sector": "Consumer Staples",       "sub_industry": "Soft Drinks",                          "weight": 0.0060},
    "KO":    {"name": "Coca-Cola Co.",               "sector": "Consumer Staples",       "sub_industry": "Soft Drinks",                          "weight": 0.0050},
    "CRM":   {"name": "Salesforce Inc.",             "sector": "Information Technology", "sub_industry": "Application Software",                 "weight": 0.0050},
    "WMT":   {"name": "Walmart Inc.",                "sector": "Consumer Staples",       "sub_industry": "Consumer Staples Merchandising",      "weight": 0.0050},
    "BAC":   {"name": "Bank of America Corp.",       "sector": "Financials",             "sub_industry": "Diversified Banks",                    "weight": 0.0050},
    "TMO":   {"name": "Thermo Fisher Scientific",    "sector": "Health Care",            "sub_industry": "Life Sciences Tools",                  "weight": 0.0040},
    "MCD":   {"name": "McDonald's Corp.",            "sector": "Consumer Discretionary", "sub_industry": "Restaurants",                          "weight": 0.0040},
    "CSCO":  {"name": "Cisco Systems Inc.",          "sector": "Information Technology", "sub_industry": "Communications Equipment",             "weight": 0.0040},
    "ACN":   {"name": "Accenture PLC CL A",          "sector": "Information Technology", "sub_industry": "IT Consulting",                         "weight": 0.0035},
    "ABT":   {"name": "Abbott Laboratories",         "sector": "Health Care",            "sub_industry": "Health Care Equipment",               "weight": 0.0035},
    "DHR":   {"name": "Danaher Corp.",               "sector": "Health Care",            "sub_industry": "Health Care Equipment",               "weight": 0.0035},
    "CMCSA": {"name": "Comcast Corp. CL A",          "sector": "Communication Services", "sub_industry": "Cable & Satellite",                    "weight": 0.0035},
    "NEE":   {"name": "NextEra Energy Inc.",         "sector": "Utilities",              "sub_industry": "Electric Utilities",                  "weight": 0.0035},
    "AMD":   {"name": "Advanced Micro Devices",      "sector": "Information Technology", "sub_industry": "Semiconductors",                      "weight": 0.0030},
    "ORCL":  {"name": "Oracle Corp.",                "sector": "Information Technology", "sub_industry": "Application Software",                 "weight": 0.0030},
    "PM":    {"name": "Philip Morris Intl.",         "sector": "Consumer Staples",       "sub_industry": "Tobacco",                              "weight": 0.0030},
    "LIN":   {"name": "Linde PLC",                   "sector": "Materials",              "sub_industry": "Industrial Gases",                     "weight": 0.0030},
    "AMAT":  {"name": "Applied Materials",           "sector": "Information Technology", "sub_industry": "Semiconductor Equipment",              "weight": 0.0030},
    "RTX":   {"name": "RTX Corp.",                  "sector": "Industrials",           "sub_industry": "Aerospace & Defense",                  "weight": 0.0030},
    "SPGI":  {"name": "S&P Global Inc.",             "sector": "Financials",             "sub_industry": "Financial Exchanges",                  "weight": 0.0030},
    "INTU":  {"name": "Intuit Inc.",                "sector": "Information Technology", "sub_industry": "Application Software",                 "weight": 0.0030},
    "TXN":   {"name": "Texas Instruments",           "sector": "Information Technology", "sub_industry": "Semiconductors",                       "weight": 0.0030},
    "AMGN":  {"name": "Amgen Inc.",                 "sector": "Health Care",            "sub_industry": "Biotechnology",                        "weight": 0.0030},
    "BKNG":  {"name": "Booking Holdings",            "sector": "Consumer Discretionary", "sub_industry": "Hotels & Leisure",                    "weight": 0.0030},
    "QCOM":  {"name": "Qualcomm Inc.",              "sector": "Information Technology", "sub_industry": "Semiconductors",                       "weight": 0.0030},
    "UBER":  {"name": "Uber Technologies",          "sector": "Consumer Discretionary", "sub_industry": "Ride Hailing",                         "weight": 0.0025},
    "ADP":   {"name": "Automatic Data Processing",  "sector": "Information Technology", "sub_industry": "Human Resource Services",              "weight": 0.0025},
    "VRTX":  {"name": "Vertex Pharmaceuticals",     "sector": "Health Care",            "sub_industry": "Biotechnology",                        "weight": 0.0025},
    "LOW":   {"name": "Lowe's Companies",           "sector": "Consumer Discretionary", "sub_industry": "Home Improvement Retail",             "weight": 0.0025},
    "NFLX":  {"name": "Netflix Inc.",               "sector": "Communication Services", "sub_industry": "Movies & Entertainment",               "weight": 0.0025},
    "CAT":   {"name": "Caterpillar Inc.",           "sector": "Industrials",            "sub_industry": "Construction Machinery",               "weight": 0.0025},
    "MS":    {"name": "Morgan Stanley",             "sector": "Financials",             "sub_industry": "Investment Banking",                   "weight": 0.0025},
    "GS":    {"name": "Goldman Sachs",              "sector": "Financials",             "sub_industry": "Investment Banking",                   "weight": 0.0025},
    "BLK":   {"name": "BlackRock Inc.",             "sector": "Financials",             "sub_industry": "Asset Management",                    "weight": 0.0025},
    "DE":    {"name": "Deere & Co.",                "sector": "Industrials",            "sub_industry": "Agricultural Machinery",              "weight": 0.0025},
}

# Normalize weights to sum to 1.0
_total_weight = sum(v["weight"] for v in CURRENT_TOP_CONSTITUENTS.values())
for ticker, info in CURRENT_TOP_CONSTITUENTS.items():
    info["weight"] /= _total_weight


# ---------------------------------------------------------------------------
# S&P 500 Composition URLs (historical from fja05680)
# ---------------------------------------------------------------------------

COMPOSITION_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv"
)

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/"
    "data/constituents.csv"
)


# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------

class SP500Data:
    """
    S&P 500 data provider.

    Provides:
    - get_constituents() — all current tickers
    - get_constituents_with_weights() — current tickers + cap weights
    - get_historical_tickers(date) — tickers that were in S&P 500 on a given date
    - get_sector_map() — {ticker: (sector, sub_industry)}
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/sp500")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._comp_cache = self.cache_dir / "composition.json"
        self._constituents_cache = self.cache_dir / "constituents.csv"
        self._composition: dict[str, list[str]] = {}
        self._sectors: dict[str, tuple[str, str]] = {}  # ticker → (sector, sub_industry)
        self._weights: dict[str, float] = {}  # ticker → weight

    def load(self, force_refresh: bool = False) -> None:
        """Load S&P 500 data from cache or GitHub."""
        self._load_sectors(force_refresh)
        self._load_composition(force_refresh)
        self._load_weights()

    def _load_sectors(self, force_refresh: bool = False) -> None:
        """Load GICS sector/sub-industry mapping."""
        if self._sectors and not force_refresh:
            return

        # Try cache
        sector_cache = self.cache_dir / "sectors.json"
        if sector_cache.exists() and not force_refresh:
            with open(sector_cache) as f:
                raw = json.load(f)
            self._sectors = {t: tuple(v) for t, v in raw.items()}
            return

        # Fetch from GitHub
        try:
            req = urllib.request.Request(
                CONSTITUENTS_URL,
                headers={"Accept": "text/csv"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")

            reader = csv.DictReader(StringIO(content))
            for row in reader:
                ticker = row.get("Symbol", "").strip()
                sector = row.get("GICS Sector", "").strip()
                sub = row.get("GICS Sub-Industry", "").strip()
                if ticker and sector:
                    self._sectors[ticker] = (sector, sub)

            # Fill in missing from our static top-50 data
            for ticker, info in CURRENT_TOP_CONSTITUENTS.items():
                if ticker not in self._sectors:
                    self._sectors[ticker] = (info["sector"], info["sub_industry"])

            # Save cache
            with open(sector_cache, "w") as f:
                json.dump({t: list(v) for t, v in self._sectors.items()}, f)
            print(f"Loaded {len(self._sectors)} constituents with sector data")
        except Exception as e:
            print(f"Warning: failed to fetch S&P 500 constituents: {e}")
            # Fall back to static data
            for ticker, info in CURRENT_TOP_CONSTITUENTS.items():
                self._sectors[ticker] = (info["sector"], info["sub_industry"])

    def _load_composition(self, force_refresh: bool = False) -> None:
        """Load historical S&P 500 composition (which tickers on which dates)."""
        if self._composition and not force_refresh:
            return

        if self._comp_cache.exists() and not force_refresh:
            with open(self._comp_cache) as f:
                self._composition = json.load(f)
            print(f"Loaded composition: {len(self._composition)} date records")
            return

        try:
            req = urllib.request.Request(
                COMPOSITION_URL,
                headers={"Accept": "text/csv"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")

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

            with open(self._comp_cache, "w") as f:
                json.dump(self._composition, f)
            print(f"Loaded {len(self._composition)} composition records")
        except Exception as e:
            print(f"Warning: failed to fetch composition: {e}")

    def _load_weights(self) -> None:
        """Load cap weights from our static top-50 + fill rest with equal weight."""
        # Use top-50 weights
        for ticker, info in CURRENT_TOP_CONSTITUENTS.items():
            self._weights[ticker] = info["weight"]

        # Remaining constituents get equal weight (we don't have their individual weights)
        all_tickers = set(self._sectors.keys()) | set(CURRENT_TOP_CONSTITUENTS.keys())
        known_weight = sum(self._weights.values())
        num_unknown = len(all_tickers) - len(self._weights)
        if num_unknown > 0:
            per_unknown = (1.0 - known_weight) / num_unknown
            for ticker in all_tickers:
                if ticker not in self._weights:
                    self._weights[ticker] = per_unknown

    def get_constituents(self) -> list[str]:
        """Return all known S&P 500 tickers."""
        return list(self._sectors.keys())

    def get_historical_tickers(self, as_of: date | str) -> list[str]:
        """Return tickers that were in S&P 500 on as_of date."""
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        date_str = as_of.isoformat()

        if date_str in self._composition:
            return self._composition[date_str]

        # Find closest prior date
        candidates = sorted(
            [d for d in self._composition if d <= date_str],
            key=lambda d: datetime.fromisoformat(d),
            reverse=True,
        )
        if candidates:
            return self._composition[candidates[0]]
        return []

    def get_weights(self) -> dict[str, float]:
        """Return current cap weights."""
        return self._weights.copy()

    def get_sector_map(self) -> dict[str, tuple[str, str]]:
        """Return {ticker: (sector, sub_industry)}."""
        return self._sectors.copy()

    def get_sub_industry_groups(self) -> dict[str, list[str]]:
        """Return {sub_industry: [tickers in same sub-industry]}."""
        groups: dict[str, list[str]] = {}
        for ticker, (sector, sub) in self._sectors.items():
            if sub not in groups:
                groups[sub] = []
            groups[sub].append(ticker)
        return groups


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[SP500Data] = None

def get_sp500() -> SP500Data:
    """Get the singleton SP500Data instance."""
    global _instance
    if _instance is None:
        _instance = SP500Data()
        _instance.load()
    return _instance
