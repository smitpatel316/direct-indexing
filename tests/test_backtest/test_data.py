"""
Tests for backtest data module.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest

from direct_indexing.backtest.data import (
    BacktestDataManager,
    PriceHistory,
)


class TestBacktestDataManager:
    """Tests for BacktestDataManager."""

    def test_get_tickers_for_date_exact_match(self):
        """Exact date match returns correct tickers."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = BacktestDataManager(cache_dir=Path(tmp))
            mgr._composition = {
                "2020-01-02": ["AAPL", "MSFT", "GOOGL"],
                "2020-01-03": ["AAPL", "MSFT", "AMZN"],
            }

            tickers = mgr.get_tickers_for_date(date(2020, 1, 2))
            assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_get_tickers_for_date_closest_prior(self):
        """Returns closest prior date when exact date not available."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = BacktestDataManager(cache_dir=Path(tmp))
            mgr._composition = {
                "2020-01-02": ["AAPL", "MSFT"],
                "2020-01-03": ["AAPL", "MSFT", "AMZN"],
            }

            # Jan 4 is a Saturday, Jan 5 is a Sunday — next trading day is Jan 6
            # But no composition for Jan 4/5/6, should return Jan 3
            tickers = mgr.get_tickers_for_date(date(2020, 1, 6))
            assert tickers == ["AAPL", "MSFT", "AMZN"]

    def test_get_tickers_for_date_empty(self):
        """Returns empty list when no composition data."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = BacktestDataManager(cache_dir=Path(tmp))
            mgr._composition = {}

            tickers = mgr.get_tickers_for_date(date(2020, 1, 1))
            assert tickers == []

    def test_next_trading_day_skips_weekends(self):
        """_next_trading_day returns Monday for Friday."""
        friday = date(2020, 1, 3)  # Friday
        monday = date(2020, 1, 6)  # Monday
        assert BacktestDataManager._next_trading_day(friday) == monday

    def test_next_trading_day_skips_saturday(self):
        """_next_trading_day returns Monday for Saturday."""
        saturday = date(2020, 1, 4)
        monday = date(2020, 1, 6)
        assert BacktestDataManager._next_trading_day(saturday) == monday

    def test_next_trading_day_skips_sunday(self):
        """_next_trading_day returns Monday for Sunday."""
        sunday = date(2020, 1, 5)
        monday = date(2020, 1, 6)
        assert BacktestDataManager._next_trading_day(sunday) == monday

    def test_get_composition_range(self):
        """get_composition_range returns all dates in range."""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = BacktestDataManager(cache_dir=Path(tmp))
            mgr._composition = {
                "2020-01-02": ["AAPL", "MSFT"],
                "2020-01-03": ["AAPL", "MSFT", "AMZN"],
                "2020-01-06": ["AAPL", "MSFT", "AMZN", "GOOGL"],
            }

            result = mgr.get_composition_range(
                date(2020, 1, 2), date(2020, 1, 6)
            )
            assert "2020-01-02" in result
            assert "2020-01-03" in result
            assert "2020-01-06" in result
            assert "2020-01-07" not in result  # weekend, no data


class TestPriceHistory:
    """Tests for PriceHistory."""

    def test_get_price_exact_match(self):
        """get_price returns price for exact date."""
        prices = PriceHistory(
            prices={
                "AAPL": {
                    "2020-01-02": 150.0,
                    "2020-01-03": 155.0,
                }
            },
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 3),
        )

        assert prices.get_price("AAPL", date(2020, 1, 2)) == 150.0
        assert prices.get_price("AAPL", date(2020, 1, 3)) == 155.0

    def test_get_price_missing_ticker(self):
        """get_price returns None for unknown ticker."""
        prices = PriceHistory(
            prices={"AAPL": {"2020-01-02": 150.0}},
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 3),
        )

        assert prices.get_price("MSFT", date(2020, 1, 2)) is None

    def test_get_price_missing_date(self):
        """get_price returns None for unknown date."""
        prices = PriceHistory(
            prices={"AAPL": {"2020-01-02": 150.0}},
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 3),
        )

        assert prices.get_price("AAPL", date(2020, 1, 5)) is None
