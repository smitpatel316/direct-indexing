"""
Backtest engine for direct indexing TLH strategies.

Simulates TLH on historical S&P 500 stocks and compares against buy-and-hold.

Usage:
    import asyncio
    from backtest.engine import BacktestEngine
    from backtest.data import BacktestDataManager

    data_mgr = BacktestDataManager(Path("data/backtest"))
    engine = BacktestEngine(data_manager=data_mgr)

    result = await engine.run(
        start_date="2019-01-01",
        end_date="2024-12-31",
        initial_portfolio=1_000_000,
        num_positions=100,  # top 100 S&P 500 stocks
    )
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import BacktestDataManager, PriceHistory


# ---------------------------------------------------------------------------
# Tax rates (defaults — override via config)
# ---------------------------------------------------------------------------

# Smit's blended LTCG rate (2026): 32% federal + 3.8% NIIT + ~10.3% CA
DEFAULT_LTCG_RATE = 0.461
DEFAULT_STCG_RATE = 0.462  # ordinary income (position held < 1 year)


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A position in the simulated portfolio."""
    symbol: str
    qty: float
    avg_cost: float  # cost basis per share

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_cost

    def market_value(self, current_price: float) -> float:
        return self.qty * current_price

    def unrealized_gain(self, current_price: float) -> float:
        return self.qty * (current_price - self.avg_cost)


@dataclass
class LotRecord:
    """Record of a buy lot for FIFO tracking."""
    lot_id: str
    symbol: str
    qty: float
    cost_per_share: float
    buy_date: date
    realized_gain: float = 0.0
    sold_qty: float = 0.0


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------


@dataclass
class HarvestEvent:
    """Record of a single tax-loss harvest."""
    date: date
    symbol: str
    loss_amount: float
    swap_target: str  # ETF bought instead
    qty_sold: float
    tax_saved: float = 0.0  # estimated tax saved at LTCG rate


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    start_date: date
    end_date: date
    initial_portfolio: float
    trading_days: int = 0

    # Portfolio values
    final_portfolio: float = 0.0
    final_benchmark: float = 0.0

    # Returns
    strategy_return_percent: float = 0.0
    benchmark_return_percent: float = 0.0
    total_tax_saved: float = 0.0
    after_tax_alpha: float = 0.0  # strategy with TLH vs without TLH

    # Harvest stats
    num_harvests: int = 0
    total_harvested_loss: float = 0.0

    harvest_events: list[HarvestEvent] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        bench_pct = self.benchmark_return_percent
        strat_pct = self.strategy_return_percent
        alpha = strat_pct - bench_pct
        alpha_sign = "+" if alpha >= 0 else ""
        lines = [
            f"Backtest: {self.start_date} → {self.end_date}",
            f"Initial:  ${self.initial_portfolio:,.0f}",
            f"Final: ${self.final_portfolio:,.2f} vs ${self.final_benchmark:,.2f}",
            "          (buy-hold SPY benchmark)",
            f"Return: {strat_pct:.2f}% (strategy) vs {bench_pct:.2f}% (benchmark)",
            f"Alpha: {alpha_sign}{alpha:.2f}%",
            f"Harvests: {self.num_harvests} events, ${self.total_harvested_loss:,.0f}",
            f"Tax saved: ${self.total_tax_saved:,.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    initial_portfolio: float = 1_000_000.0
    num_positions: int = 100  # number of S&P 500 stocks to hold
    loss_threshold_percent: float = 5.0
    min_loss_amount: float = 100.0
    swap_etf: str = "VOO"  # replacement ETF after harvest (alias: replacement_etf)
    ltcg_rate: float = DEFAULT_LTCG_RATE  # long-term capital gains rate
    stcg_rate: float = DEFAULT_STCG_RATE  # short-term capital gains rate
    rebalance_frequency_days: int = 30


@dataclass
class BacktestEngine:
    """Simulates TLH strategy on historical data."""

    data_manager: BacktestDataManager
    config: BacktestConfig = field(default_factory=BacktestConfig)

    _positions: dict[str, Position] = field(default_factory=dict)
    _lots: dict[str, list[LotRecord]] = field(default_factory=dict)  # symbol -> lots
    _cash: float = 0.0
    _harvest_events: list[HarvestEvent] = field(default_factory=list)
    _spy_shares: float = 0.0  # benchmark: buy-and-hold SPY
    _spy_cost: float = 0.0

    # Tax-saved cash accumulated through TLH harvests (reinvested)
    _tax_saved_cash: float = 0.0  # tracks cumulative tax savings

    async def run(self) -> BacktestResult:
        """Run the backtest.

        Computes two parallel portfolio values:
        - Strategy (with TLH): equal-weight portfolio, harvesting losses,
          tax savings accumulated and reinvested
        - Benchmark (without TLH): same positions, no harvesting,
          compare after-tax alpha
        """
        start = date.fromisoformat(self.config.start_date)
        end = date.fromisoformat(self.config.end_date)

        # Load composition data
        print("Loading S&P 500 composition data...")
        await self.data_manager.load_composition()

        # Get the tickers for the start date
        initial_tickers = self.data_manager.get_tickers_for_date(start)
        if not initial_tickers:
            raise ValueError(f"No S&P 500 composition found for {start}")

        # Take a sample of tickers for the portfolio
        tickers = initial_tickers[: self.config.num_positions]
        print(f"Building portfolio with {len(tickers)} positions: {tickers[:5]}...")

        # Fetch historical prices for all tickers
        print("Fetching historical prices (this may take a minute)...")
        prices = await self.data_manager.get_prices(
            tickers, start, end
        )

        # Get SPY prices for benchmark
        spy_prices = await self.data_manager.get_spy_prices(start, end)

        # Initialize portfolio: equal-weight across tickers
        per_position = self.config.initial_portfolio / len(tickers)
        self._positions = {}
        self._lots = {}
        self._cash = 0.0
        self._harvest_events = []
        self._tax_saved_cash = 0.0  # accumulated tax savings from TLH

        # No-TLH benchmark portfolio (same positions, no harvesting)
        self._positions_no_tlh = {}

        for ticker in tickers:
            # Buy at first available price
            first_price = self._find_first_price(prices, ticker, start)
            if first_price is None:
                continue
            qty = per_position / first_price
            self._positions[ticker] = Position(
                symbol=ticker, qty=qty, avg_cost=first_price
            )
            # Record a buy lot
            self._lots[ticker] = [
                LotRecord(
                    lot_id=f"{ticker}-buy-0",
                    symbol=ticker,
                    qty=qty,
                    cost_per_share=first_price,
                    buy_date=start,
                )
            ]

        # Initialize SPY benchmark
        first_spy = self._find_first_price(spy_prices, "SPY", start)
        if first_spy:
            self._spy_shares = self.config.initial_portfolio / first_spy
            self._spy_cost = first_spy

        # Initialize no-TLH benchmark with same positions
        for ticker, pos in self._positions.items():
            self._positions_no_tlh[ticker] = Position(
                symbol=pos.symbol,
                qty=pos.qty,
                avg_cost=pos.avg_cost,
            )

        print(f"Portfolio initialized: {len(self._positions)} positions")
        print(f"Backtesting {start} → {end}...")

        # Walk through each trading day
        current = start
        last_rebalance = start
        days = 0
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                await self._process_day(current, prices, spy_prices)
                days += 1

                # Check for periodic rebalance
                if (current - last_rebalance).days >= self.config.rebalance_frequency_days:  # noqa: E501
                    await self._rebalance(prices)
                    last_rebalance = current

            current += timedelta(days=1)

        print(f"Backtest complete: {days} trading days processed")

        # Calculate final result
        return self._compute_result(start, end, prices, spy_prices)

    def _find_first_price(
        self, prices: PriceHistory, ticker: str, on_or_after: date
    ) -> float | None:
        """Find the first available price for ticker on or after on_or_after."""
        ticker_prices = prices.prices.get(ticker, {})
        for d_str in sorted(ticker_prices.keys()):
            d = date.fromisoformat(d_str)
            if d >= on_or_after:
                return ticker_prices[d_str]
        return None

    async def _process_day(
        self,
        day: date,
        prices: PriceHistory,
        spy_prices: PriceHistory,
    ) -> None:
        """Process a single trading day: check for harvests."""

        # Check each position for harvestable losses
        for ticker, position in list(self._positions.items()):
            ticker_prices = prices.prices.get(ticker, {})
            day_str = day.isoformat()
            if day_str not in ticker_prices:
                continue

            current_price = ticker_prices[day_str]
            if current_price <= 0:
                continue

            gain = position.unrealized_gain(current_price)
            cost_basis_total = position.cost_basis
            gain_pct = (gain / cost_basis_total) * 100 if cost_basis_total > 0 else 0
            loss_pct = abs(gain_pct)

            # Check for loss harvest opportunity
            threshold_pct = self.config.loss_threshold_percent
            min_amount = self.config.min_loss_amount

            if gain < -min_amount and loss_pct >= threshold_pct:
                # Harvest! Sell the position and buy swap ETF
                loss_amount = abs(gain)
                qty = position.qty

                # Tax saved = loss amount × LTCG rate (reinvested later)
                tax_saved = loss_amount * self.config.ltcg_rate

                # Record harvest event
                self._harvest_events.append(
                    HarvestEvent(
                        date=day,
                        symbol=ticker,
                        loss_amount=loss_amount,
                        swap_target=self.config.swap_etf,
                        qty_sold=qty,
                        tax_saved=tax_saved,
                    )
                )

                # Accumulate tax saved cash (will reinvest on next rebalance)
                self._tax_saved_cash += tax_saved

                # Close the position (sell at current price)
                position_value = qty * current_price
                self._cash += position_value

                # Record the loss for tax purposes (FIFO: reduce the lot)
                self._close_lots_fifo(ticker, qty, current_price, day)

                # Remove position
                del self._positions[ticker]

    def _close_lots_fifo(
        self, symbol: str, qty: float, price: float, sale_date: date
    ) -> None:
        """Close lots using FIFO and record gain/loss."""
        lots = self._lots.get(symbol, [])
        remaining = qty
        realized_gain = 0.0
        realized_loss = 0.0

        while remaining > 0 and lots:
            lot = lots[0]
            if lot.sold_qty >= lot.qty:
                lots.pop(0)
                continue

            sell_from_lot = min(remaining, lot.qty - lot.sold_qty)
            proceeds = sell_from_lot * price
            cost = sell_from_lot * lot.cost_per_share
            gain = proceeds - cost

            if gain >= 0:
                realized_gain += gain
            else:
                realized_loss += abs(gain)

            lot.sold_qty += sell_from_lot
            remaining -= sell_from_lot

        if remaining > 0:
            # This shouldn't happen if qty matches lots
            realized_loss += remaining * price

    async def _rebalance(self, prices: PriceHistory) -> None:
        """Rebalance portfolio to equal weight and reinvest tax-saved cash."""
        # Build current prices dict: symbol -> latest close price
        current_prices = {}
        for sym in self._positions:
            ticker_prices = prices.prices.get(sym, {})
            if ticker_prices:
                last_date = sorted(ticker_prices.keys())[-1]
                current_prices[sym] = ticker_prices[last_date]
            else:
                current_prices[sym] = 0.0

        position_value = sum(
            pos.market_value(current_prices.get(pos.symbol, 0))
            for pos in self._positions.values()
        )
        total_value = self._cash + self._tax_saved_cash + position_value
        if total_value <= 0:
            return

        # Reinvest tax-saved cash into equal-weight positions
        if self._tax_saved_cash > 0:
            per_new = self._tax_saved_cash / len(self._positions)
            for ticker, pos in self._positions.items():
                price = current_prices.get(ticker, 0)
                if price > 0:
                    additional_qty = per_new / price
                    # Add to existing position
                    new_total_qty = pos.qty + additional_qty
                    new_avg_cost = (
                        (pos.qty * pos.avg_cost + additional_qty * price)
                        / new_total_qty
                    )
                    pos.qty = new_total_qty
                    pos.avg_cost = new_avg_cost
            self._cash += self._tax_saved_cash
            self._tax_saved_cash = 0.0

        # Basic rebalancing: maintain equal weight
        # (simplified: just ensure we're close to equal, no aggressive selling)
        target_per_position = total_value / len(self._positions)
        for ticker, pos in list(self._positions.items()):
            price = current_prices.get(ticker, 0)
            if price <= 0:
                continue
            current_value = pos.qty * price
            diff = current_value - target_per_position
            # Don't rebalance if within 5% of target
            if abs(diff) / target_per_position < 0.05:
                continue

    def _compute_no_tlh_value(self, prices: PriceHistory) -> float:
        """Compute final value of portfolio WITHOUT TLH harvesting.

        Same positions as strategy portfolio, but no harvesting executed,
        no tax savings reinvested. This is the proper benchmark for
        measuring after-tax alpha from TLH.
        """
        total = 0.0
        for ticker, position in self._positions_no_tlh.items():
            ticker_prices = prices.prices.get(ticker, {})
            if not ticker_prices:
                continue
            last_date = sorted(ticker_prices.keys())[-1]
            last_price = ticker_prices[last_date]
            if isinstance(last_price, (int, float)) and last_price > 0:
                total += position.market_value(last_price)
        return total

    def _compute_result(
        self,
        start: date,
        end: date,
        prices: PriceHistory,
        spy_prices: PriceHistory,
    ) -> BacktestResult:
        """Compute the final backtest result."""
        # Final portfolio value
        final_value = self._cash
        for ticker, position in self._positions.items():
            ticker_prices = prices.prices.get(ticker, {})
            last_date = sorted(ticker_prices.keys())[-1] if ticker_prices else None
            if last_date:
                final_value += position.market_value(ticker_prices[last_date])

        # Benchmark: buy and hold SPY
        spy_ticker_prices = spy_prices.prices.get("SPY", {})
        if spy_ticker_prices:
            last_spy_date = sorted(spy_ticker_prices.keys())[-1]
            final_spy_value = self._spy_shares * spy_ticker_prices[last_spy_date]
        else:
            final_spy_value = self.config.initial_portfolio

        total_return_raw = (
            (final_value - self.config.initial_portfolio)
            / self.config.initial_portfolio
        )
        total_return_bench_raw = (
            (final_spy_value - self.config.initial_portfolio)
            / self.config.initial_portfolio
        )

        # Tax alpha: losses harvested × LTCG rate = estimated tax saved
        total_harvested_loss = sum(h.loss_amount for h in self._harvest_events)
        estimated_tax_saved = total_harvested_loss * self.config.ltcg_rate

        # Compute no-TLH benchmark final value
        no_tlh_value = self._compute_no_tlh_value(prices)

        # After-tax alpha: strategy with TLH vs without TLH
        after_tax_alpha = final_value - no_tlh_value

        # Compute trading days
        trading_days = (end - start).days

        return BacktestResult(
            start_date=start,
            end_date=end,
            trading_days=trading_days,
            initial_portfolio=self.config.initial_portfolio,
            final_portfolio=final_value,
            final_benchmark=final_spy_value,
            strategy_return_percent=total_return_raw * 100,
            benchmark_return_percent=total_return_bench_raw * 100,
            total_tax_saved=estimated_tax_saved,
            after_tax_alpha=after_tax_alpha,
            num_harvests=len(self._harvest_events),
            total_harvested_loss=total_harvested_loss,
            harvest_events=self._harvest_events,
        )
