"""
Tests for PortfolioManager — bridges Alpaca positions with ETF targets.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

from direct_indexing.portfolio_manager import (
    PortfolioManager,
    PortfolioPosition,
    PortfolioDriftReport,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def manager():
    return PortfolioManager(target_etf="VOO", drift_threshold=2.0)


@pytest.fixture
def mock_positions():
    """Two mock Alpaca positions."""
    aapl = MagicMock()
    aapl.symbol = "AAPL"
    aapl.qty = 100.0
    aapl.current_price = 180.0
    aapl.market_value = 18_000.0

    msft = MagicMock()
    msft.symbol = "MSFT"
    msft.qty = 50.0
    msft.current_price = 400.0
    msft.market_value = 20_000.0

    return [aapl, msft]


# =============================================================================
# Target loading
# =============================================================================

class TestTargetLoading:
    """Tests for loading ETF target constituents."""

    def test_set_targets_from_dict(self, manager):
        manager.set_targets_from_dict({"AAPL": 7.25, "MSFT": 5.50, "GOOGL": 4.0})
        targets = manager.get_target_constituents()
        assert targets["AAPL"] == 7.25
        assert targets["MSFT"] == 5.50
        assert targets["GOOGL"] == 4.0

    def test_set_targets_from_dict_uppercases_ticker(self, manager):
        manager.set_targets_from_dict({"aapl": 7.25, "msft": 5.50})
        targets = manager.get_target_constituents()
        assert "AAPL" in targets
        assert "aapl" not in targets

    def test_set_targets_from_csv(self, manager, tmp_path):
        csv_file = tmp_path / "constituents.csv"
        csv_file.write_text("ticker,weight\nAAPL,7.25\nMSFT,5.5\nGOOGL,4.0\n")
        count = manager.set_targets_from_csv(csv_file)
        assert count == 3
        assert manager.get_target_constituents()["AAPL"] == 7.25

    def test_get_target_constituents_returns_copy(self, manager):
        manager.set_targets_from_dict({"AAPL": 7.25})
        targets = manager.get_target_constituents()
        targets["AAPL"] = 99.0
        assert manager.get_target_constituents()["AAPL"] == 7.25  # Unchanged

    def test_empty_targets(self, manager):
        targets = manager.get_target_constituents()
        assert targets == {}


# =============================================================================
# Drift calculation
# =============================================================================

class TestDriftCalculation:
    """Tests for drift calculation against ETF targets."""

    def test_drift_empty_targets(self, manager, mock_positions):
        """No targets defined → no drift calculated, but positions flagged as non-ETF."""
        report = manager.calculate_drift(mock_positions)
        # All positions are "extra" (not in ETF targets)
        assert len(report.positions) == 2
        assert report.needs_rebalance is True  # 100% drift for each

    def test_drift_perfect_match(self, manager, mock_positions):
        """Portfolio weights match ETF targets exactly → zero drift."""
        # VOO is ~28% AAPL and ~17% MSFT. Set portfolio to match.
        # Total = $38,000. AAPL at 7.25% = $2,755. MSFT at 5.5% = $2,090.
        # But our mock positions are AAPL=$18k, MSFT=$20k, total=$38k.
        # AAPL weight: 18k/38k = 47.4%, target 7.25% → huge drift.
        manager.set_targets_from_dict({"AAPL": 47.4, "MSFT": 52.6})
        report = manager.calculate_drift(mock_positions)

        aapl = next(p for p in report.positions if p.ticker == "AAPL")
        msft = next(p for p in report.positions if p.ticker == "MSFT")

        # Current weights match target → drift near 0
        assert abs(aapl.drift_percent) < 0.5
        assert abs(msft.drift_percent) < 0.5
        assert report.needs_rebalance is False

    def test_drift_overweight(self, manager, mock_positions):
        """Overweight position shows positive drift."""
        # Set target very low for AAPL
        manager.set_targets_from_dict({"AAPL": 1.0, "MSFT": 1.0})
        report = manager.calculate_drift(mock_positions)

        aapl = next(p for p in report.positions if p.ticker == "AAPL")
        # AAPL: current ~47%, target 1% → drift ~+46%
        assert aapl.drift_percent > 0
        assert aapl.current_weight > aapl.target_weight

    def test_drift_underweight(self, manager, mock_positions):
        """Underweight position shows negative drift."""
        manager.set_targets_from_dict({"AAPL": 80.0, "MSFT": 80.0})  # Total > 100% impossible
        report = manager.calculate_drift(mock_positions)

        aapl = next(p for p in report.positions if p.ticker == "AAPL")
        # AAPL: current 47%, target 80% → drift -33%
        assert aapl.drift_percent < 0

    def test_drift_extra_position_not_in_etf(self, manager, mock_positions):
        """Position held that's not in ETF targets is flagged."""
        manager.set_targets_from_dict({"AAPL": 50.0})  # Only AAPL in ETF
        report = manager.calculate_drift(mock_positions)

        # MSFT is not in ETF targets
        msft = next((p for p in report.positions if p.ticker == "MSFT"), None)
        assert msft is not None
        assert msft.target_weight == 0.0
        assert msft.current_weight > 0.0
        assert msft.shares_to_trade > 0  # Should sell it

    def test_drift_missing_constituent(self, manager, mock_positions):
        """ETF has AAPL but portfolio doesn't → underweight."""
        manager.set_targets_from_dict({"AAPL": 50.0, "MSFT": 50.0})
        # Remove MSFT from positions
        aapl_only = [mock_positions[0]]  # Only AAPL
        report = manager.calculate_drift(aapl_only)

        msft = next(p for p in report.positions if p.ticker == "MSFT")
        # MSFT: current 0%, target 50% → drift = -50%
        assert msft.current_value == 0.0
        assert msft.drift_percent < 0

    def test_max_drift_threshold(self, manager, mock_positions):
        """needs_rebalance True only when max drift exceeds threshold."""
        manager.drift_threshold = 2.0
        manager.set_targets_from_dict({"AAPL": 47.4, "MSFT": 52.6})
        report = manager.calculate_drift(mock_positions)
        assert report.needs_rebalance is False  # Within 2% threshold

        # Make AAPL wildly overweight
        manager.set_targets_from_dict({"AAPL": 1.0})
        report = manager.calculate_drift(mock_positions)
        assert report.needs_rebalance is True  # >2% drift

    def test_total_drift_value(self, manager, mock_positions):
        """total_drift_value sums absolute drift across all positions."""
        manager.set_targets_from_dict({"AAPL": 47.4, "MSFT": 52.6})
        report = manager.calculate_drift(mock_positions)
        # Each drift is near 0 since weights match
        assert report.total_drift_value < 1000.0

    def test_drift_report_fields(self, manager, mock_positions):
        manager.set_targets_from_dict({"AAPL": 50.0})
        report = manager.calculate_drift(mock_positions)

        assert isinstance(report.timestamp, datetime)
        assert report.total_value == 38_000.0  # 18k + 20k
        assert report.target_etf == "VOO"
        assert len(report.positions) >= 1


# =============================================================================
# Rebalance recommendations
# =============================================================================

class TestRebalanceRecommendations:
    """Tests for rebalancing trade recommendations."""

    def test_recommendations_empty(self, manager):
        """No targets → no recommendations."""
        report = PortfolioDriftReport(
            timestamp=datetime.now(),
            total_value=0,
            target_etf="VOO",
            positions=[],
            max_drift_percent=0,
            total_drift_value=0,
            needs_rebalance=False,
        )
        recs = manager.rebalance_recommendations(report)
        assert recs == []

    def test_recommendations_sell_overweight(self, manager):
        """Overweight position → SELL recommendation."""
        positions = [
            PortfolioPosition(
                ticker="AAPL",
                target_weight=10.0,
                target_value=10_000.0,
                target_shares=55.0,
                current_shares=200.0,
                current_price=180.0,
                current_value=36_000.0,
                current_weight=60.0,
                drift_value=26_000.0,
                drift_percent=50.0,
                shares_to_trade=145.0,
            )
        ]
        report = PortfolioDriftReport(
            timestamp=datetime.now(),
            total_value=60_000.0,
            target_etf="VOO",
            positions=positions,
            max_drift_percent=50.0,
            total_drift_value=26_000.0,
            needs_rebalance=True,
        )

        recs = manager.rebalance_recommendations(report)
        assert len(recs) == 1
        assert recs[0]["action"] == "sell"
        assert recs[0]["ticker"] == "AAPL"

    def test_recommendations_buy_underweight(self, manager):
        """Underweight position → BUY recommendation."""
        positions = [
            PortfolioPosition(
                ticker="AAPL",
                target_weight=50.0,
                target_value=50_000.0,
                target_shares=278.0,
                current_shares=10.0,
                current_price=180.0,
                current_value=1_800.0,
                current_weight=3.0,
                drift_value=-48_200.0,
                drift_percent=-47.0,
                shares_to_trade=268.0,
            )
        ]
        report = PortfolioDriftReport(
            timestamp=datetime.now(),
            total_value=60_000.0,
            target_etf="VOO",
            positions=positions,
            max_drift_percent=47.0,
            total_drift_value=48_200.0,
            needs_rebalance=True,
        )

        recs = manager.rebalance_recommendations(report)
        assert len(recs) == 1
        assert recs[0]["action"] == "buy"
        assert recs[0]["ticker"] == "AAPL"

    def test_recommendations_within_threshold_skipped(self, manager):
        """Position within drift threshold → no recommendation."""
        positions = [
            PortfolioPosition(
                ticker="AAPL",
                target_weight=50.0,
                target_value=50_000.0,
                target_shares=278.0,
                current_shares=270.0,  # Close to target
                current_price=180.0,
                current_value=48_600.0,
                current_weight=48.6,
                drift_value=-1_400.0,
                drift_percent=-1.4,  # Within 2% threshold
                shares_to_trade=8.0,
            )
        ]
        report = PortfolioDriftReport(
            timestamp=datetime.now(),
            total_value=100_000.0,
            target_etf="VOO",
            positions=positions,
            max_drift_percent=1.4,
            total_drift_value=1_400.0,
            needs_rebalance=False,
        )

        recs = manager.rebalance_recommendations(report)
        assert recs == []

    def test_recommendations_small_shares_skipped(self, manager):
        """Position with < 1 share to trade → skipped."""
        positions = [
            PortfolioPosition(
                ticker="AAPL",
                target_weight=50.0,
                target_value=50_000.0,
                target_shares=278.0,
                current_shares=277.5,  # Very close
                current_price=180.0,
                current_value=49_950.0,
                current_weight=49.95,
                drift_value=-50.0,
                drift_percent=-0.05,
                shares_to_trade=0.28,  # < 1 share
            )
        ]
        report = PortfolioDriftReport(
            timestamp=datetime.now(),
            total_value=100_000.0,
            target_etf="VOO",
            positions=positions,
            max_drift_percent=0.05,
            total_drift_value=50.0,
            needs_rebalance=False,
        )

        recs = manager.rebalance_recommendations(report)
        assert recs == []
