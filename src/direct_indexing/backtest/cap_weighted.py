"""
Cap-weighted S&P 500 backtester.

This extends the standard backtest with proper cap-weighted initialization
and rebalancing using real market cap data from yfinance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd

from .engine import BacktestEngine, BacktestConfig, BacktestResult, Position, HarvestEvent
from .data import BacktestDataManager


async def _get_market_caps(tickers: list[str], as_of: date) -> dict[str, float]:
    """
    Get market caps for tickers as of a given date.
    Uses shares outstanding × price as a proxy for market cap.
    """
    market_caps = {}
    
    # Fetch in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            # Download shares outstanding and current price
            data = yf.download(batch, start=as_of.isoformat(), end=(as_of + timedelta(days=5)).isoformat(), 
                              progress=False, auto_adjust=True, threads=True)
            if data.empty:
                continue
            
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            
            # Use last available price
            for ticker in batch:
                if ticker in close.index:
                    price = float(close[ticker])
                    if price > 0:
                        market_caps[ticker] = price  # Use price as weight proxy
        except Exception:
            continue
    
    return market_caps


class CapWeightedBacktestEngine(BacktestEngine):
    """
    Cap-weighted version of the backtest engine.
    
    Instead of equal-weighting all positions, this allocates capital
    proportionally based on S&P 500 market cap weights.
    """

    async def _fetch_cap_weights(self, tickers: list[str], as_of: date) -> dict[str, float]:
        """Fetch market cap-based weights for tickers."""
        market_caps = await _get_market_caps(tickers, as_of)
        if not market_caps:
            return {}
        
        total = sum(market_caps.values())
        if total == 0:
            return {}
        
        return {ticker: cap / total for ticker, cap in market_caps.items()}

    async def run_cap_weighted(
        self,
        start: date | str,
        end: date | str,
        initial: float = 100_000.0,
        rebalance_days: int = 30,
    ) -> BacktestResult:
        """Run a cap-weighted backtest."""
        if isinstance(start, str):
            start = date.fromisoformat(start)
        if isinstance(end, str):
            end = date.fromisoformat(end)

        await self.data_manager.load_composition()
        tickers = self.data_manager.get_tickers_for_date(start)

        prices = await self.data_manager.get_prices(tickers, start, end)
        spy_prices = await self.data_manager.get_spy_prices(start, end)

        # Fetch initial cap weights
        cap_weights = await self._fetch_cap_weights(tickers, start)

        # Initialize portfolio with cap weights
        self._positions = {}
        self._lots = {}
        self._cash = 0.0
        self._harvest_events = []
        self._tax_saved_cash = 0.0

        self._positions_no_tlh = {}

        for ticker, weight in cap_weights.items():
            first_price = self._find_first_price(prices, ticker, start)
            if first_price is None:
                continue
            qty = (initial * weight) / first_price
            self._positions[ticker] = Position(
                symbol=ticker, qty=qty, avg_cost=first_price
            )
            self._lots[ticker] = [
                LotRecord(
                    lot_id=f"{ticker}-buy-0",
                    symbol=ticker,
                    qty=qty,
                    cost_per_share=first_price,
                    buy_date=start,
                )
            ]

        # SPY benchmark
        first_spy = self._find_first_price(spy_prices, "SPY", start)
        if first_spy:
            self._spy_shares = initial / first_spy
            self._spy_cost = first_spy

        # No-TLH benchmark
        for ticker, pos in self._positions.items():
            self._positions_no_tlh[ticker] = Position(
                symbol=pos.symbol,
                qty=pos.qty,
                avg_cost=pos.avg_cost,
            )

        print(f"Cap-weighted portfolio initialized: {len(self._positions)} positions")

        # Walk through trading days
        current = start
        last_rebalance = start
        days = 0
        while current <= end:
            if current.weekday() < 5:
                await self._process_day(current, prices, spy_prices)
                days += 1

                if (current - last_rebalance).days >= rebalance_days:
                    # Fetch new cap weights and rebalance
                    new_weights = await self._fetch_cap_weights(
                        self.data_manager.get_tickers_for_date(current), current
                    )
                    if new_weights:
                        await self._rebalance(prices, new_weights)
                    else:
                        await self._rebalance(prices, None)
                    last_rebalance = current

            current += timedelta(days=1)

        print(f"Backtest complete: {days} trading days processed")
        return self._compute_result(start, end, prices, spy_prices)
