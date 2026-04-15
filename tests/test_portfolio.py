"""
Portfolio Replicator - ETF constituent tracking and portfolio construction.

TDD: Tests first, define expected behavior, then implement.
"""

import pytest
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# We'll build the real implementation after defining the spec
from src.direct_indexing.portfolio import (
    Constituent,
    ETFReplica,
    PositionSide,
    DriftReport,
)


class TestConstituent:
    """Tests for ETF constituent."""

    def test_constituent_creation(self):
        c = Constituent(
            ticker="AAPL",
            shares=100.0,
            weight=5.5,
            price=175.0,
        )
        assert c.ticker == "AAPL"
        assert c.shares == 100.0
        assert c.weight == 5.5
        assert c.price == 175.0

    def test_constituent_market_value(self):
        c = Constituent(
            ticker="AAPL",
            shares=100.0,
            weight=5.5,
            price=175.0,
        )
        # market_value = shares * price
        assert c.market_value == 17500.0

    def test_constituent_target_shares_from_weight(self):
        # Given a portfolio value and target weight, calculate shares
        c = Constituent(ticker="AAPL", shares=0, weight=5.5, price=175.0)
        portfolio_value = 500_000.0
        
        # target_value = weight/100 * portfolio_value
        # shares = target_value / price
        target_shares = (c.weight / 100 * portfolio_value) / c.price
        assert abs(target_shares - 157.14) < 0.01


class TestETFReplica:
    """Tests for ETF replica manager."""

    def test_replica_creation(self):
        replica = ETFReplica(target_etf="SPY")
        assert replica.target_etf == "SPY"
        assert replica.constituents == {}  # dict, not list
        assert replica.last_rebalance is None

    def test_add_constituent(self):
        replica = ETFReplica(target_etf="SPY")
        replica.add(ticker="AAPL", shares=100.0, weight=5.5, price=175.0)
        
        assert len(replica.constituents) == 1
        c = replica.constituents["AAPL"]
        assert c.ticker == "AAPL"
        assert c.shares == 100.0

    def test_add_constituent_updates_existing(self):
        replica = ETFReplica(target_etf="SPY")
        replica.add(ticker="AAPL", shares=100.0, weight=5.5, price=175.0)
        replica.add(ticker="AAPL", shares=150.0, weight=5.5, price=180.0)
        
        assert len(replica.constituents) == 1
        assert replica.constituents["AAPL"].shares == 150.0

    def test_remove_constituent(self):
        replica = ETFReplica(target_etf="SPY")
        replica.add(ticker="AAPL", shares=100.0, weight=5.5, price=175.0)
        replica.remove("AAPL")
        
        assert "AAPL" not in replica.constituents

    def test_total_market_value(self):
        replica = ETFReplica(target_etf="SPY")
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=175.0)
        replica.add(ticker="MSFT", shares=50.0, weight=50.0, price=400.0)
        
        # 100 * 175 + 50 * 400 = 17500 + 20000 = 37500
        assert replica.total_market_value == 37500.0

    def test_rebalance_needed_when_drift_exceeds_threshold(self):
        replica = ETFReplica(target_etf="SPY", drift_threshold=5.0)
        # Use prices that result in exactly equal weights (no drift)
        # AAPL: 100 shares at $200 = $20000
        # MSFT: 100 shares at $200 = $20000
        # Total = $40000, each at exactly 50%
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=200.0)
        replica.add(ticker="MSFT", shares=100.0, weight=50.0, price=200.0)
        
        # Equal weights = no drift
        assert not replica.needs_rebalance()
        
        # Raise AAPL price by 20% -> $200 -> $240
        # AAPL: 100 * 240 = 24000
        # MSFT: 100 * 200 = 20000
        # Total = 44000, AAPL = 54.55%, drift = 4.55% (within 5% threshold)
        replica.update_price("AAPL", 240.0)
        assert not replica.needs_rebalance()
        
        # Raise AAPL price by another 20% -> $240 -> $288
        # AAPL: 100 * 288 = 28800
        # MSFT: 100 * 200 = 20000
        # Total = 48800, AAPL = 59.02%, drift = 9.02% (> 5% threshold)
        replica.update_price("AAPL", 288.0)
        assert replica.needs_rebalance()

    def test_drift_report_generation(self):
        replica = ETFReplica(target_etf="SPY", drift_threshold=5.0)
        # Use equal prices to get exact 50/50 split
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=200.0)
        replica.add(ticker="MSFT", shares=100.0, weight=50.0, price=200.0)
        
        report = replica.drift_report()
        
        assert isinstance(report, DriftReport)
        assert report.total_value == 40000.0
        assert len(report.positions) == 2
        assert report.needs_rebalance is False
        assert report.max_drift == 0.0

    def test_rebalance_calculates_trades_to_target_weights(self):
        replica = ETFReplica(target_etf="SPY", drift_threshold=2.0)
        # Portfolio value = 37500
        replica.add(ticker="AAPL", shares=100.0, weight=50.0, price=175.0)
        replica.add(ticker="MSFT", shares=50.0, weight=50.0, price=400.0)
        
        # Drift AAPL to 60% by raising its price
        replica.update_price("AAPL", 233.33)
        
        # After drift: AAPL = 60%, MSFT = 40%
        # AAPL drift = +10%, MSFT drift = -10%
        
        trades = replica.rebalance_trades()
        
        # Should suggest selling AAPL and buying MSFT
        assert "AAPL" in trades
        assert "MSFT" in trades
        # AAPL is overweight → sell
        assert trades["AAPL"].side == PositionSide.SELL
        # MSFT is underweight → buy
        assert trades["MSFT"].side == PositionSide.BUY


class TestDriftReport:
    """Tests for drift report."""

    def test_drift_calculation(self):
        replica = ETFReplica(target_etf="SPY", drift_threshold=1.0)
        # 2 constituents at 50% each with equal prices
        replica.add(ticker="A", shares=100.0, weight=50.0, price=100.0)
        replica.add(ticker="B", shares=100.0, weight=50.0, price=100.0)
        # Total = $20000, each = 50%
        
        # Increase A's price by 50% -> $150
        # A = 100 * 150 = 15000
        # B = 100 * 100 = 10000
        # Total = 25000, A = 60%, B = 40%
        # A drift = +10%, B drift = -10%
        replica.update_price("A", 150.0)
        
        report = replica.drift_report()
        
        # A drifted +10%
        a_position = next(p for p in report.positions if p.ticker == "A")
        assert abs(a_position.drift - 10.0) < 0.1
        assert a_position.current_weight == 60.0
        assert report.needs_rebalance is True  # 10% drift > 1% threshold


class TestPositionSide:
    """Tests for PositionSide enum."""

    def test_position_side_values(self):
        assert PositionSide.BUY.value == "buy"
        assert PositionSide.SELL.value == "sell"
        assert PositionSide.HOLD.value == "hold"