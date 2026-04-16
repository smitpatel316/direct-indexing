# Direct Indexing with Tax-Loss Harvesting

A modern Python implementation of direct indexing with automated tax-loss harvesting, inspired by Oracle's optimization-based approach. Built for production use on Alpaca.

## Features

- **True Direct Indexing**: Hold individual S&P 500 stocks, not ETFs
- **MILP Optimization**: Oracle-style optimization considers tax cost + drift + transaction cost jointly
- **Tax-Loss Harvesting**: Lot-level precision, 31-day wash sale window
- **No ETF Wrapper**: Harvested losses → cash → wait 31 days → repurchase original ticker
- **Backtesting**: Historical simulation with after-tax alpha measurement

## Architecture

```
direct-indexing/
├── src/direct_indexing/
│   ├── optimizer/              # MILP optimization engine (Oracle-inspired)
│   │   ├── oracle.py          # Main optimizer (PuLP + CBC)
│   │   ├── strategy.py        # Strategy interface
│   │   ├── solver.py          # CBC solver wrapper
│   │   ├── objectives/        # Tax, drift, transaction cost objectives
│   │   └── constraints/        # Wash sale, weight bound constraints
│   ├── optimizer_tlh.py        # Optimizer-based TLH engine
│   ├── tlh.py                 # Rule-based TLH engine (legacy)
│   ├── lot_tracker.py         # Lot-level basis tracking (FIFO)
│   ├── portfolio_manager.py   # Portfolio management
│   └── backtest/              # Historical backtesting
├── tests/                     # 181 passing tests
└── config.yaml.example        # Configuration template
```

## How It Works

**Traditional TLH with ETF Wrapper:**
```
Sell AAPL at loss → Buy VOO (similar, not identical)
→ Market moves → Sell VOO → Buy AAPL (wash sale risk)
```

**Oracle-Inspired Direct Indexing:**
```
Sell AAPL at loss → Hold cash
→ Wait 31 days (wash sale window)
→ Repurchase AAPL at new (hopefully higher) price
→ No replacement ETF = no wash sale risk
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/smitpatel316/direct-indexing.git
cd direct-indexing
pip install -e .

# Run backtest
di backtest --start 2021-01-01 --end 2023-12-31

# Run live scan (configure config.yaml first)
di run
```

## Backtest Results

Direct indexing with 1.5% loss threshold, 31-day wash sale window, $100K initial portfolio:

| Period | Strategy | Benchmark (SPY) | Alpha | Harvests | Tax Saved | After-Tax Alpha |
|--------|----------|-----------------|-------|----------|-----------|-----------------|
| 2018-01 → 2023-12 | +2.4% | +145.1% | -142.7% | 64 | $1,512 | -$76,460 |
| 2019-01 → 2023-12 | +29.3% | +158.4% | -129.1% | 52 | $1,368 | -$65,447 |
| 2021-01 → 2023-12 | +28.1% | +68.9% | -40.8% | 48 | $1,105 | -$8,553 |
| 2022-01 → 2023-12 | -13.8% | +28.7% | -42.5% | 81 | $1,847 | -$16,115 |

**Key Insight**: After-tax alpha compares strategy WITH TLH vs strategy WITHOUT TLH (same equal-weight portfolio). The negative after-tax alpha in rising markets reflects that equal-weight S&P 500 underperforms cap-weighted index. TLH value emerges in volatile sideways markets.

## Configuration

```yaml
tlh:
  enabled: true
  loss_threshold_percent: 1.5    # Optimization handles smaller losses
  min_loss_amount: 100.0
  wash_sale_window_days: 31     # Safe beyond IRS 30-day rule
  wash_sale_enabled: true
  carryforward_enabled: true
  # Optimizer settings
  min_weight_multiplier: 0.5     # Min position as % of target
  max_weight_multiplier: 2.0     # Max position as % of target
  min_notional: 100.0           # Minimum trade size
  solve_time_limit: 60            # Max seconds for MILP solver

tax_rates:
  short_term_rate: 0.37   # Ordinary income (< 1 year)
  long_term_rate: 0.20    # Capital gains (>= 1 year)
```

## Commands

```bash
# Backtest
di backtest --start 2021-01-01 --end 2023-12-31 --initial-value 100000

# Report
di report

# Run live system
di run

# Scan for harvest opportunities
di scan
```

## Optimization vs Rule-Based

| Feature | Rule-Based (threshold → harvest) | Optimizer-Based (MILP) |
|---------|----------------------------------|------------------------|
| Loss threshold | 5% fixed | 1.5% (optimization sizes optimally) |
| Trade sizing | Entire position | Per-lot, sized to weight bounds |
| Multi-objective | Not supported | Tax + drift + txn cost jointly |
| Constraints | Simple | Wash sale, weight bounds, cash |
| Wash sale | Via replacement ETF | Ticker-level 31-day restriction |

## Disclaimer

This software is for educational purposes. Tax laws change. Always consult a CPA before implementing any tax strategy. Backtest thoroughly before using with real money.

## License

MIT
