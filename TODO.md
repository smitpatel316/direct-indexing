# Direct Indexing - Next Phase Tasks

All Vikunja tasks (di-1 through di-7, IDs 95-99) are complete. These are the next priorities.

## Backtest Improvements

### [ ] Improve benchmark: Strategy WITH TLH vs WITHOUT TLH
**Why:** Currently comparing against SPY buy-and-hold, which measures absolute return not tax alpha.
**Fix:** Track two parallel portfolios:
- Portfolio A: equal-weight S&P 500 WITH TLH harvesting (tax savings reinvested daily)
- Portfolio B: equal-weight S&P 500 WITHOUT TLH (same trades, no harvesting)
**After-tax alpha** = (A end value) - (B end value)
**Notes:**
- Tax savings from harvests accumulate in cash daily
- Cash gets reinvested at next rebalance
- This properly measures whether TLH adds value vs doing nothing

### [ ] Add transaction costs to backtest
**Why:** Equal-weight rebalancing generates more trades than cap-weighted.
**Fix:** Deduct 0.1% per trade as transaction cost, or use realistic bid-ask spread.

### [ ] Validate wash sale carryforward in backtest
**Why:** TLH engine has carryforward logic, but backtest engine doesn't track it.
**Fix:** Add _carryforward tracking to backtest. When a loss is harvested, record it. When replacement ETF is bought within 30 days, adjust cost basis.

## CLI Improvements

### [ ] `di backtest --output json` for programmatic use

### [ ] `di backtest --compare 2018-01,2023-12` to run multiple periods

## Documentation

### [ ] Document the backtest methodology in README
- Data source: fja05680/sp500 GitHub repo
- Price source: yfinance
- Equal-weight portfolio construction
- TLH thresholds and parameters
- Known limitations

## Performance

### [ ] Speed up price fetching for large backtests
**Issue:** Fetching 500 tickers × 6 years takes 5+ minutes.
**Fix:** Parallel batch downloads, pre-fetch composition changes only.

---

Created: 2026-04-15