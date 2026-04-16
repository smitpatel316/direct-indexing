"""
Command-Line Interface for Direct Indexing
Modern CLI using argparse with subcommands.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .alpaca_client import AlpacaClient
from .config import AppConfig, ConfigManager
from .tlh import TLHEngine


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Direct Indexing with Tax-Loss Harvesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run the full system")
    run_parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch web dashboard"
    )

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Run TLH scan")
    scan_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Loss threshold percentage (overrides config)"
    )
    scan_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show results without executing trades"
    )

    # status command
    subparsers.add_parser("status", help="Show portfolio status")

    # setup command
    setup_parser = subparsers.add_parser("setup", help="Setup ETF replication")
    setup_parser.add_argument(
        "--etf", "-e",
        type=str,
        default="SPY",
        help="Target ETF to replicate"
    )

    # dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch dashboard only")
    dashboard_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Dashboard port"
    )

    # report command
    report_parser = subparsers.add_parser("report", help="Generate TLH report")
    report_parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Report format"
    )

    # backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "--start", "-s",
        type=str,
        default="2018-01-01",
        help="Start date (YYYY-MM-DD). Default: 2018-01-01"
    )
    backtest_parser.add_argument(
        "--end", "-e",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD). Default: 2024-12-31"
    )
    backtest_parser.add_argument(
        "--portfolio", "-p",
        type=str,
        default="equal_weight_sp500",
        help="Portfolio type: equal_weight_sp500. Default: equal_weight_sp500"
    )
    backtest_parser.add_argument(
        "--initial-value", "-i",
        type=float,
        default=100000.0,
        help="Initial portfolio value. Default: $100,000"
    )
    backtest_parser.add_argument(
        "--full", "-F",
        action="store_true",
        help="Run full backtest with all metrics (Sharpe, Sortino, Alpha, Beta, etc.)"
    )
    backtest_parser.add_argument(
        "--sensitivity", "-S",
        action="store_true",
        help="Run sensitivity analysis across parameter variations"
    )
    backtest_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file for results"
    )

    # pure-direct run command
    pure_run_parser = subparsers.add_parser(
        "run-pure",
        help="Run Pure Direct Indexer: 31-day rebalance + TLH with sector substitutes"
    )
    pure_run_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebalance even if < 31 days since last"
    )
    pure_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing trades"
    )
    pure_run_parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.0005,
        help="Drift threshold to trigger trade (default: 0.0005 = 0.05%%)"
    )
    pure_run_parser.add_argument(
        "--tlh-loss-min",
        type=float,
        default=10.0,
        help="Minimum loss $ to harvest (default: $10)"
    )

    # pure-direct status
    pure_status_parser = subparsers.add_parser(
        "status-pure",
        help="Show Pure Direct Indexer status"
    )

    # pure-direct rebalance
    pure_rebal_parser = subparsers.add_parser(
        "rebalance-pure",
        help="Force a Pure Direct Indexer rebalance"
    )
    pure_rebal_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing trades"
    )

    return parser


def cmd_run(args, config: AppConfig) -> int:
    """Run the full system."""
    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )

    # Check market status
    market = client.get_market_status()
    print(f"Market status: {'OPEN' if market.get('is_open') else 'CLOSED'}")
    print(f"Next open: {market.get('next_open', 'N/A')}")
    print(f"Next close: {market.get('next_close', 'N/A')}")
    print()

    # Run TLH scan
    tlh_engine = TLHEngine(client, config.tlh)
    results = tlh_engine.run_daily_scan()

    if results:
        print(f"TLH Scan Results: {len(results)} harvests executed")
        for r in results:
            status = "OK" if r.success else "FAIL"
            sym = r.symbol
            print(f"  {status} {sym}: -${r.loss_amount:.2f} ({r.loss_percent:.1f}%)")
    else:
        print("TLH Scan: No harvestable positions found")

    return 0


def cmd_scan(args, config: AppConfig) -> int:
    """Run TLH scan."""
    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )
    tlh_engine = TLHEngine(client, config.tlh)

    if args.threshold:
        tlh_engine.config.loss_threshold_percent = args.threshold

    # Update expired wash sales
    expired = tlh_engine.update_expired_wash_sales()
    if expired > 0:
        print(f"Updated {expired} expired wash sale entries")

    # Scan portfolio
    harvestable = tlh_engine.scan_portfolio()

    if not harvestable:
        print("No harvestable positions found")
        return 0

    print(f"\nFound {len(harvestable)} harvestable positions:")
    print("-" * 50)

    for i, pos in enumerate(harvestable, 1):
        print(f"{i}. {pos.symbol}")
        entry_str = f"Entry: ${pos.avg_entry_price:.2f}"
        curr_str = f"Current: ${pos.current_price:.2f}"
        print(f"   Qty: {pos.qty} | {entry_str} | {curr_str}")
        loss_str = f"Loss: ${abs(pos.loss_amount):.2f} ({abs(pos.loss_percent):.1f}%)"
        print(f"   {loss_str}")
        print()

    if args.dry_run:
        print("(Dry run - no trades executed)")
        return 0

    print("Executing harvests...")
    results = tlh_engine.run_daily_scan()

    for r in results:
        if r.success:
            print(f"✓ Harvested {r.symbol}: -${r.loss_amount:.2f}")
        else:
            print(f"✗ Failed to harvest {r.symbol}: {r.error}")

    return 0


def cmd_status(args, config: AppConfig) -> int:
    """Show portfolio status."""
    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )

    # Get account info
    account = client.get_account()

    print("=" * 50)
    print("DIRECT INDEXING PORTFOLIO STATUS")
    print("=" * 50)
    print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
    print(f"Cash: ${account.cash:,.2f}")
    print(f"Equity: ${account.equity:,.2f}")
    print(f"Buying Power: ${account.buying_power:,.2f}")
    print()

    # Get positions
    positions = client.get_positions()

    if not positions:
        print("No open positions")
        return 0

    print(f"Open Positions: {len(positions)}")
    print("-" * 50)

    for pos in positions:
        pl_char = "+" if pos.unrealized_pl >= 0 else ""
        print(f"{pos.symbol}")
        print(f"  Qty: {pos.qty} | Avg: ${pos.avg_entry_price:.2f}")
        pl_val = f"{pl_char}${pos.unrealized_pl:.2f}"
        pl_pct = f"{pl_char}{pos.unrealized_plpc:.2f}%"
        pl_str = f"P/L: {pl_val} ({pl_pct})"
        print(f"  {pl_str}")

    # TLH summary
    tlh_engine = TLHEngine(client, config.tlh)
    summary = tlh_engine.get_summary()

    print()
    print("TAX-LOSS HARVESTING STATUS")
    print("-" * 50)
    print(f"Carryforward Balance: ${summary['carryforward_balance']:.2f}")
    print(f"Active Wash Sales: {summary['active_wash_sales']}")
    print(f"Harvestable Positions: {summary['harvestable_positions']}")

    return 0


def cmd_setup(args, config: AppConfig) -> int:
    """Setup ETF replication."""
    print(f"Setting up ETF replication for: {args.etf}")
    print("This would download ETF constituents and configure the portfolio.")
    print("(Feature in development)")
    return 0


def cmd_dashboard(args, config: AppConfig) -> int:
    """Launch dashboard."""
    print("Starting dashboard...")
    print(f"Connect to http://localhost:{args.port}")
    print("(Dashboard feature in development)")
    return 0


def cmd_report(args, config: AppConfig) -> int:
    """Generate TLH report."""
    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )
    tlh_engine = TLHEngine(client, config.tlh)

    summary = tlh_engine.get_summary()
    history = tlh_engine.get_history()

    if args.format == "json":
        import json
        print(json.dumps({
            **summary,
            "harvest_history": history,
        }, indent=2, default=str))
        return 0

    print("=" * 60)
    print("TAX-LOSS HARVESTING REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    print(f"Carryforward Balance: ${summary['carryforward_balance']:.2f}")
    print(f"Active Wash Sales: {summary['active_wash_sales']}")
    print(f"Expired Wash Sales: {summary.get('expired_wash_sales', 0)}")
    print(f"Harvestable Positions: {summary['harvestable_positions']}")
    print()

    if summary["top_losses"]:
        print("Top Harvestable Losses:")
        for loss in summary["top_losses"]:
            sym = loss["symbol"]
            amt = loss["loss_amount"]
            pct = loss["loss_percent"]
            print(f"  {sym}: -${amt:.2f} ({pct:.1f}%)")
        print()

    if history:
        print(f"Harvest History ({len(history)} events):")
        print("-" * 60)
        total_saved = 0.0
        for ev in reversed(history[-20:]):
            dt = ev.get("date", "?")[:10]
            sym = ev.get("symbol", "?")
            loss = ev.get("loss_amount", 0)
            swap_sym = ev.get("swap_target", "N/A")
            saved = loss * config.tlh.ltcg_rate
            total_saved += saved
            print(f"  {dt} | {sym:6} | -${loss:8.2f} | → {swap_sym:4}")
            print(f"         saved ~${saved:.2f}")
        print()
        print(f"Estimated total tax saved (all time): ${total_saved:.2f}")
        print(f"(Based on LTCG rate: {config.tlh.ltcg_rate*100:.1f}%)")
    else:
        print("No harvest history yet.")
        print()
        print("Swap History (pending swaps):")
        swaps = tlh_engine.get_pending_swaps()
        if swaps:
            for s in swaps[:5]:
                orig = s.get("original_symbol", "?")
                tgt = s.get("target_etf", "?")
                amt = s.get("amount", 0)
                print(f"  {s.get('scheduled_date','?')} | {orig} → {tgt} | ${amt:.2f}")
        else:
            print("  No pending swaps.")

    return 0


def cmd_backtest(args, config: AppConfig) -> int:
    """Run backtest on historical data."""
    import asyncio

    from .backtest.data import BacktestDataManager
    from .backtest.engine import BacktestConfig, BacktestEngine

    print(f"Backtest: {args.portfolio}")
    print(f"Period: {args.start} → {args.end}")
    print(f"Initial value: ${args.initial_value:,.2f}")
    print()

    # Build backtest config from CLI args + tlh config
    tlh_cfg = config.tlh
    backtest_config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_portfolio=args.initial_value,
        loss_threshold_percent=tlh_cfg.loss_threshold_percent,
        min_loss_amount=tlh_cfg.min_loss_amount,
        ltcg_rate=tlh_cfg.ltcg_rate,
        swap_etf=tlh_cfg.swap_etfs[0] if tlh_cfg.swap_etfs else "VOO",
    )

    # Initialize data manager (caches to disk)
    cache_dir = Path.home() / ".cache" / "direct-indexing"
    data_mgr = BacktestDataManager(cache_dir=cache_dir)

    # Run backtest
    print("Loading S&P 500 composition data...")
    engine = BacktestEngine(backtest_config, data_mgr)

    print("Running backtest... (this may take a few minutes)")
    result = asyncio.run(engine.run())

    # Print summary
    print()
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period:           {result.start_date} → {result.end_date}")
    print(f"Trading days:     {result.trading_days}")
    print(f"Harvest events:   {len(result.harvest_events)}")
    print()
    print("Benchmark (SPY buy-and-hold):")
    print(f"  Start value:    ${result.initial_portfolio:,.2f}")
    print(f"  End value:      ${result.final_benchmark:,.2f}")
    print(f"  Total return:   {result.benchmark_return_percent:.2f}%")
    print()
    print("Strategy (equal-weight S&P 500 + TLH):")
    print(f"  Start value:    ${result.initial_portfolio:,.2f}")
    print(f"  End value:      ${result.final_portfolio:,.2f}")
    print(f"  Total return:   {result.strategy_return_percent:.2f}%")
    print()
    alpha = result.strategy_return_percent - result.benchmark_return_percent
    alpha_sign = "+" if alpha >= 0 else ""
    print(f"After-tax alpha vs benchmark: {alpha_sign}{alpha:.2f}%")
    print(f"Total tax saved (harvested):   ${result.total_tax_saved:.2f}")
    print()

    if result.harvest_events:
        print("Top 5 harvest events:")
        sorted_events = sorted(
            result.harvest_events,
            key=lambda e: e.loss_amount,
            reverse=True
        )[:5]
        for ev in sorted_events:
            dt = str(ev.harvest_date)[:10]
            saved = f"~${ev.tax_saved:.2f}" if ev.tax_saved else "$0.00"
            print(f"  {dt}: {ev.symbol} -${ev.loss_amount:.2f} (saved {saved})")

    return 0


def cmd_backtest_full(args, config: AppConfig) -> int:
    """Run comprehensive backtest with full metrics (Sharpe, Sortino, Alpha, etc.)."""
    import asyncio
    from .backtest.backtest_engine import BacktestEngine, BacktestConfig as FullConfig

    print(f"Comprehensive Backtest: Pure Direct Indexing")
    print(f"Period: {args.start} → {args.end}")
    print(f"Initial value: ${args.initial_value:,.2f}")
    print()

    cfg = FullConfig(
        start_date=args.start,
        end_date=args.end,
        initial_value=args.initial_value,
        rebalance_days=31,
        drift_threshold=0.0005,
        tlh_loss_min=10.0,
        tlh_loss_pct=0.01,
        slippage_bps=0.5,
        risk_free_rate=0.05,
    )

    engine = BacktestEngine(cfg)
    print("Running... (fetching price data, may take a few minutes)")
    results = asyncio.run(engine.run())

    print(engine.summary())

    if args.output:
        import pandas as pd
        df = pd.DataFrame([results])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

    return 0


def cmd_sensitivity(args, config: AppConfig) -> int:
    """Run sensitivity analysis across parameter variations."""
    from .backtest.backtest_engine import BacktestEngine, BacktestConfig as FullConfig, SensitivityAnalyzer

    print(f"Sensitivity Analysis: Pure Direct Indexing")
    print(f"Period: {args.start} → {args.end}")
    print(f"Initial value: ${args.initial_value:,.2f}")
    print()

    base = FullConfig(
        start_date=args.start,
        end_date=args.end,
        initial_value=args.initial_value,
        slippage_bps=0.5,
        risk_free_rate=0.05,
    )

    variations = {
        "rebalance_days": [21, 31, 45, 63, 91],
        "drift_threshold": [0.0001, 0.0005, 0.0025],
        "tlh_loss_min": [5.0, 10.0, 25.0, 50.0],
    }

    print(f"Running {21} backtest variations...")
    df = SensitivityAnalyzer.run(base, variations)

    # Show key columns
    cols = ["variant", "cagr_strategy", "sharpe_strategy", "max_drawdown_strategy",
            "tracking_error", "information_ratio", "tax_alpha_annual",
            "num_tlh_events", "total_tlh_harvested"]
    print()
    print(df[cols].to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nFull results saved to {args.output}")

    return 0


def cmd_run_pure(args, config: AppConfig) -> int:
    """Run Pure Direct Indexer."""
    from .direct_indexer import PureDirectIndexer, RebalanceReason
    import asyncio

    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )
    indexer = PureDirectIndexer(
        alpaca_client=client,
        drift_threshold=args.drift_threshold,
        tlh_loss_min=args.tlh_loss_min,
    )

    status = indexer.get_status()
    print("=== Pure Direct Indexer Status ===")
    print(f"Portfolio value: ${status['portfolio_value']:,.2f}")
    print(f"Positions:       {status['num_positions']}")
    print(f"Cash:            ${status['cash']:,.2f}")
    print(f"Last rebalance:  {status['last_rebalance'] or 'Never'}")
    print(f"Days since:      {status['days_since_rebalance']}")
    print(f"Needs rebalance: {status['needs_rebalance']}")
    print(f"Wash sales:      {status['wash_sales_active']} active")
    print()

    if not args.force and not indexer.needs_rebalance():
        days = indexer.get_days_since_rebalance()
        print(f"Not yet time to rebalance ({days}/{indexer.rebalance_days} days).")
        print("Use --force to rebalance anyway.")
        return 0

    if args.dry_run:
        print("[Dry run — showing plan without executing trades]")
        # TODO: implement dry-run plan generation
        print("Dry-run plan generation not yet implemented.")
        return 0

    plan = asyncio.run(indexer.rebalance(RebalanceReason.SCHEDULED))
    print(f"\nRebalance complete:")
    print(f"  Sells: {len(plan.sell_orders)}")
    print(f"  TLH sells: {len(plan.tlh_sells)}")
    print(f"  Buys: {len(plan.buy_orders)}")
    print(f"  TLH buys: {len(plan.tlh_buys)}")

    return 0


def cmd_status_pure(args, config: AppConfig) -> int:
    """Show Pure Direct Indexer status."""
    from .direct_indexer import PureDirectIndexer

    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )
    indexer = PureDirectIndexer(alpaca_client=client)
    status = indexer.get_status()

    print("=== Pure Direct Indexer ===")
    print(f"Portfolio value:  ${status['portfolio_value']:,.2f}")
    print(f"Cash:             ${status['cash']:,.2f}")
    print(f"Positions:        {status['num_positions']}")
    print(f"Last rebalance:   {status['last_rebalance'] or 'Never'}")
    print(f"Days since:       {status['days_since_rebalance']} / 31")
    print(f"Needs rebalance:  {status['needs_rebalance']}")
    print(f"Active wash sales:{status['wash_sales_active']}")
    print()

    # Show open wash sales
    today = date.today()
    from datetime import timedelta
    ws_list = indexer.lot_tracker.get_open_wash_sales(today)
    if ws_list:
        print("Open Wash Sale Restrictions:")
        for ws in ws_list[:10]:
            days_left = (ws.reopen_date - today).days
            print(f"  {ws.ticker}: restricted until {ws.reopen_date} ({days_left}d left) → {ws.substitute_used}")
    else:
        print("No active wash sale restrictions.")

    return 0


def cmd_rebalance_pure(args, config: AppConfig) -> int:
    """Force a Pure Direct Indexer rebalance."""
    from .direct_indexer import PureDirectIndexer, RebalanceReason
    import asyncio

    client = AlpacaClient(
        config.alpaca.api_key,
        config.alpaca.api_secret,
        paper=config.alpaca.paper_trading,
    )
    indexer = PureDirectIndexer(alpaca_client=client)

    if args.dry_run:
        print("[Dry run]")
        return 0

    plan = asyncio.run(indexer.rebalance(RebalanceReason.SCHEDULED))
    print(f"Rebalance complete:")
    print(f"  Sells: {len(plan.sell_orders)}")
    print(f"  TLH sells: {len(plan.tlh_sells)}")
    print(f"  Buys: {len(plan.buy_orders)}")
    print(f"  TLH buys: {len(plan.tlh_buys)}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.load()

        # Route to command handler
        if args.command == "run":
            return cmd_run(args, config)
        elif args.command == "scan":
            return cmd_scan(args, config)
        elif args.command == "status":
            return cmd_status(args, config)
        elif args.command == "setup":
            return cmd_setup(args, config)
        elif args.command == "dashboard":
            return cmd_dashboard(args, config)
        elif args.command == "report":
            return cmd_report(args, config)
        elif args.command == "backtest":
            if args.full:
                return cmd_backtest_full(args, config)
            elif args.sensitivity:
                return cmd_sensitivity(args, config)
            return cmd_backtest(args, config)
        elif args.command == "run-pure":
            return cmd_run_pure(args, config)
        elif args.command == "status-pure":
            return cmd_status_pure(args, config)
        elif args.command == "rebalance-pure":
            return cmd_rebalance_pure(args, config)
        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
