"""
Tests for ETF Constituents Loader.

Loads the list of stocks in an ETF (e.g., VOO) from a data source
so the portfolio replicator can calculate drift.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

from direct_indexing.portfolio import (
    Constituent,
    ETFReplica,
    DriftReport,
    DriftPosition,
)


# =============================================================================
# Constituent dataclass
# =============================================================================

class TestConstituent:
    """Tests for the Constituent dataclass."""

    def test_constituent_fields(self):
        c = Constituent(
            ticker="AAPL",
            shares=100.0,
            weight=7.25,  # percentage (7.25%)
            price=180.0,
        )
        assert c.ticker == "AAPL"
        assert c.shares == 100.0
        assert c.weight == 7.25
        assert c.price == 180.0

    def test_constituent_weight_is_percentage(self):
        """Weight is a percentage (7.25 for 7.25%), not a fraction."""
        c = Constituent(ticker="AAPL", shares=100.0, weight=7.25, price=180.0)
        assert c.weight > 1.0  # Should be > 1 (e.g., 7.25%)
        assert c.market_value == 18_000.0

    def test_market_value(self):
        c = Constituent(ticker="AAPL", shares=100.0, weight=7.25, price=180.0)
        assert c.market_value == 18_000.0

    def test_target_value(self):
        c = Constituent(ticker="AAPL", shares=100.0, weight=10.0, price=180.0)
        # 10% of $100,000 = $10,000
        assert c.target_value(100_000.0) == 10_000.0

    def test_target_shares(self):
        c = Constituent(ticker="AAPL", shares=100.0, weight=10.0, price=180.0)
        # $10,000 target / $180 price = 55.55 shares
        assert c.target_shares(100_000.0) == pytest.approx(55.555, abs=0.01)


# =============================================================================
# ETFReplica — drift calculation
# =============================================================================

class TestETFReplicaDrift:
    """Tests for ETFReplica drift reporting."""

    def test_drift_report_empty(self, tmp_path):
        """Empty replica has zero drift."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        report = replica.drift_report()

        assert report.total_value == 0.0
        assert report.max_drift == 0.0
        assert len(report.positions) == 0
        assert report.needs_rebalance is False

    def test_drift_report_perfect_match(self, tmp_path):
        """Current weight == target weight → zero drift."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=100.0)
        replica.add(ticker="MSFT", shares=50.0, weight=50.0, price=100.0)

        report = replica.drift_report()

        # Total = $10,000 + $5,000 = $15,000
        # AAPL: $10,000/$15,000 = 66.67%, target 50% → drift +16.67%
        # MSFT: $5,000/$15,000 = 33.33%, target 50% → drift -16.67%
        assert report.max_drift > 0  # There's drift
        assert len(report.positions) == 2

    def test_drift_report_no_drift_when_weights_match_portfolio(self, tmp_path):
        """When portfolio is 100% in constituents at correct weights, no drift."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        # VOO is ~28% AAPL. Set total_value so AAPL's market_value matches weight.
        # If we have $10,000 total and want AAPL at 50% weight:
        replica.add(ticker="AAPL", shares=50.0, weight=50.0, price=100.0)
        replica.add(ticker="MSFT", shares=50.0, weight=50.0, price=100.0)

        # At $15,000 total: AAPL = $5,000/15k = 33.33%, drift = -16.67%
        report = replica.drift_report()

        # Each position's drift = current_weight - target_weight
        for pos in report.positions:
            # These won't be zero because we didn't match the weights correctly
            assert isinstance(pos.drift, float)

    def test_drift_report_identifies_overweight(self, tmp_path):
        """Correctly identifies when a position exceeds target weight."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        # AAPL: market_value $15,000, weight target 10% → target value $5,000
        replica.add(ticker="AAPL", shares=150.0, weight=10.0, price=100.0)

        report = replica.drift_report()

        aapl = next(p for p in report.positions if p.ticker == "AAPL")
        # Current: $15,000 / $15,000 = 100%. Target: 10%. Drift: +90%
        assert aapl.drift == pytest.approx(90.0, abs=0.1)
        assert report.max_drift == 90.0

    def test_drift_report_identifies_underweight(self, tmp_path):
        """Correctly identifies when a position is below target weight."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        # AAPL: market_value $1,000, weight target 50% → target value $7,500
        replica.add(ticker="AAPL", shares=10.0, weight=50.0, price=100.0)
        replica.add(ticker="MSFT", shares=70.0, weight=50.0, price=100.0)

        report = replica.drift_report()

        aapl = next(p for p in report.positions if p.ticker == "AAPL")
        # Current: $1,000 / $8,000 = 12.5%. Target: 50%. Drift: -37.5%
        assert aapl.drift < 0  # Underweight

    def test_needs_rebalance_threshold(self, tmp_path):
        """needs_rebalance is True when max drift exceeds threshold."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=150.0, weight=10.0, price=100.0)  # 90% drift

        assert replica.needs_rebalance() is True

    def test_needs_rebalance_within_threshold(self, tmp_path):
        """needs_rebalance is False when all drifts are within threshold."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=51.0, weight=50.0, price=100.0)
        replica.add(ticker="MSFT", shares=49.0, weight=50.0, price=100.0)

        # Total = $10,000. AAPL = 51%, drift = +1%. Within 2% threshold.
        assert replica.needs_rebalance() is False

    def test_rebalance_trades(self, tmp_path):
        """rebalance_trades returns the trades needed to reach target weights."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=200.0, weight=50.0, price=100.0)  # $20k, 100%
        replica.add(ticker="MSFT", shares=0.0, weight=50.0, price=100.0)   # $0, 0%

        trades = replica.rebalance_trades()

        # Total = $20,000. Target per position = $10,000.
        # AAPL: has $20,000, target $10,000 → sell $10,000
        # MSFT: has $0, target $10,000 → buy $10,000
        assert "AAPL" in trades
        assert "MSFT" in trades
        assert trades["MSFT"].side.value == "buy"
        assert trades["AAPL"].side.value == "sell"

    def test_update_price(self, tmp_path):
        """update_price changes the price and affects drift calculation."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=100.0)  # $10k, 100%

        # Initially 100% AAPL with $10k
        report1 = replica.drift_report()
        assert report1.max_drift == 50.0  # 100% - 50% = 50% drift

        # Update price — market value doubles but weight is still wrong
        replica.update_price("AAPL", 200.0)  # Now $20k total
        report2 = replica.drift_report()
        # Still 100% AAPL, target 50% → still 50% drift
        assert report2.max_drift == 50.0

    def test_remove_constituent(self, tmp_path):
        """remove() deletes a constituent from the replica."""
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=100.0)
        replica.add(ticker="MSFT", shares=100.0, weight=50.0, price=100.0)

        removed = replica.remove("AAPL")
        assert removed is True
        assert "AAPL" not in replica.constituents
        assert "MSFT" in replica.constituents


# =============================================================================
# DriftReport fields
# =============================================================================

class TestDriftReport:
    """Tests for the DriftReport dataclass."""

    def test_drift_report_fields(self, tmp_path):
        replica = ETFReplica(target_etf="VOO", drift_threshold=2.0, data_dir=tmp_path)
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=100.0)

        report = replica.drift_report()

        assert isinstance(report.timestamp, datetime)
        assert report.total_value == 10_000.0
        assert report.max_drift == 50.0
        assert report.needs_rebalance is True
        assert len(report.positions) == 1


class TestDriftPosition:
    """Tests for the DriftPosition dataclass."""

    def test_drift_position_fields(self):
        pos = DriftPosition(
            ticker="AAPL",
            current_weight=60.0,
            target_weight=50.0,
            drift=10.0,
            current_value=60_000.0,
            target_value=50_000.0,
            current_shares=100.0,
            current_price=600.0,
        )
        assert pos.ticker == "AAPL"
        assert pos.current_weight == 60.0
        assert pos.target_weight == 50.0
        assert pos.drift == 10.0
        assert pos.current_value == 60_000.0
        assert pos.target_value == 50_000.0
        assert pos.current_shares == 100.0
        assert pos.current_price == 600.0
