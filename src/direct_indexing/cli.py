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

    return parser


def cmd_run(args, config: AppConfig) -> int:
    """Run the full system."""
    client = AlpacaClient(config.alpaca)

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
    client = AlpacaClient(config.alpaca)
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
    client = AlpacaClient(config.alpaca)

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
    client = AlpacaClient(config.alpaca)
    tlh_engine = TLHEngine(client, config.tlh)

    summary = tlh_engine.get_summary()

    if args.format == "json":
        import json
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("=" * 50)
        print("TAX-LOSS HARVESTING REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 50)
        print()
        print(f"Carryforward Balance: ${summary['carryforward_balance']:.2f}")
        print(f"Active Wash Sales: {summary['active_wash_sales']}")
        print(f"Harvestable Positions: {summary['harvestable_positions']}")
        print()

        if summary["top_losses"]:
            print("Top Harvestable Losses:")
            for loss in summary["top_losses"]:
                sym = loss["symbol"]
                amt = loss["loss_amount"]
                pct = loss["loss_percent"]
                print(f"  {sym}: -${amt:.2f} ({pct:.1f}%)")

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
