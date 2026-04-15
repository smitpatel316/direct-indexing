# Direct Indexing with Tax-Loss Harvesting

A modern Python implementation of ETF replication with automated tax-loss harvesting, designed to run on a Raspberry Pi.

## Features

- **ETF Replication**: Replicate any ETF (S&P 500, Total Market, etc.) using Alpaca
- **Tax-Loss Harvesting**: Automated scanning and harvesting of losses
- **Wash Sale Tracking**: 30-day rule enforcement with carryforward ledger
- **Web Dashboard**: Real-time monitoring via FastAPI + React
- **Self-hosted**: Runs 100% on your infrastructure, no Google dependency

## Quick Start

```bash
# Clone the repo
git clone https://github.com/smitpatel316/ReplicateETFSheets.git
cd ReplicateETFSheets

# Install dependencies
pip install -e .

# Configure (edit config.yaml)
cp config.yaml.example config.yaml

# Run with Docker
docker-compose up -d

# Or run directly
python -m direct_indexing.cli run --config config.yaml
```

## Architecture

```
direct-indexing/
├── src/direct_indexing/     # Main package
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── alpaca_client.py    # Alpaca API wrapper
│   ├── portfolio.py        # Portfolio tracking
│   ├── tlh.py              # Tax-loss harvesting engine
│   ├── wash_sale.py        # Wash sale tracker
│   └── etf_replicator.py   # ETF replication logic
├── tests/                   # Unit tests
├── dashboard/              # Web dashboard (FastAPI + React)
├── scripts/                # Utility scripts
└── config.yaml.example     # Configuration template
```

## Commands

```bash
# Run the full system
python -m direct_indexing.cli run

# Run TLH scan only
python -m direct_indexing.cli scan --threshold 5

# Check portfolio status
python -m direct_indexing.cli status

# Run dashboard only
python -m direct_indexing.cli dashboard

# Setup ETF replication (e.g., S&P 500)
python -m direct_indexing.cli setup --etf SPY
```

## Configuration

Edit `config.yaml`:

```yaml
alpaca:
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  paper_trading: true  # Set false for live trading

tlh:
  enabled: true
  loss_threshold_percent: 5.0
  min_loss_amount: 100.0
  frequency: "daily"  # daily, weekly, monthly
  swap_etfs:
    - VOO
    - SPY
    - IVV

portfolio:
  rebalance_threshold: 0.02  # 2% drift triggers rebalance
  target_etf: "SPY"
```

## Deployment

### Raspberry Pi (Recommended)

```bash
# Install
pip install -e .

# Add to crontab for daily TLH scan at 4 PM ET
crontab -e
# 0 16 * * 1-5 /usr/bin/python3 -m direct_indexing.cli scan
```

### Docker

```bash
docker-compose up -d
```

## Disclaimer

This software is for educational purposes. Tax laws change. Always consult a CPA before implementing any tax strategy. Backtest thoroughly before using with real money.

## License

MIT