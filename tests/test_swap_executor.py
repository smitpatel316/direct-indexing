"""
Tests for SwapExecutor — executes replacement ETF buys after TLH harvests.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from direct_indexing.swap_executor import (
    SwapExecutor,
    SwapRecord,
    SwapStatus,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def executor(data_dir):
    mock_client = MagicMock()
    return SwapExecutor(alpaca_client=mock_client, data_dir=data_dir)


@pytest.fixture
def swap_file(data_dir):
    return data_dir / "pending_swaps.json"


# =============================================================================
# SwapRecord
# =============================================================================

class TestSwapRecord:
    """Tests for SwapRecord dataclass."""

    def test_from_dict(self):
        d = {
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 1000.0,
            "qty": 2,
            "scheduled_date": "2026-04-15",
            "status": "PENDING",
        }
        swap = SwapRecord.from_dict(d)
        assert swap.original_symbol == "AAPL"
        assert swap.target_etf == "VOO"
        assert swap.amount == 1000.0
        assert swap.qty == 2
        assert swap.status == SwapStatus.PENDING

    def test_to_dict_roundtrip(self):
        swap = SwapRecord(
            original_symbol="AAPL",
            target_etf="VOO",
            amount=1000.0,
            qty=2,
            scheduled_date="2026-04-15",
            status=SwapStatus.EXECUTED,
            executed_at="2026-04-15T10:00:00",
            order_id="order-123",
        )
        d = swap.to_dict()
        restored = SwapRecord.from_dict(d)
        assert restored.original_symbol == swap.original_symbol
        assert restored.status == swap.status
        assert restored.order_id == "order-123"


# =============================================================================
# SwapExecutor — persistence
# =============================================================================

class TestSwapExecutorPersistence:
    """Tests for loading and saving swaps."""

    def test_load_swaps_empty(self, executor, swap_file):
        swap_file.write_text("[]")
        assert executor._load_swaps() == []

    def test_load_swaps_multiple(self, executor, swap_file):
        swaps = [
            {
                "original_symbol": "AAPL",
                "target_etf": "VOO",
                "amount": 1000.0,
                "qty": 2,
                "scheduled_date": "2026-04-15",
                "status": "PENDING",
            },
            {
                "original_symbol": "MSFT",
                "target_etf": "SPY",
                "amount": 500.0,
                "qty": 1,
                "scheduled_date": "2026-04-15",
                "status": "PENDING",
            },
        ]
        swap_file.write_text(json.dumps(swaps))
        loaded = executor._load_swaps()
        assert len(loaded) == 2
        assert loaded[0].original_symbol == "AAPL"
        assert loaded[1].target_etf == "SPY"

    def test_save_and_load_roundtrip(self, executor, swap_file):
        swaps = [
            SwapRecord(
                original_symbol="AAPL",
                target_etf="VOO",
                amount=1000.0,
                qty=2,
                scheduled_date="2026-04-15",
                status=SwapStatus.PENDING,
            )
        ]
        executor._save_swaps(swaps)
        loaded = executor._load_swaps()
        assert len(loaded) == 1
        assert loaded[0].original_symbol == "AAPL"


# =============================================================================
# SwapExecutor — execution
# =============================================================================

class TestSwapExecutorExecution:
    """Tests for swap execution logic."""

    def test_execute_pending_none_due(self, executor, swap_file):
        """No swaps due today → nothing executed."""
        future = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        swap_file.write_text(json.dumps([{
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 1000.0,
            "qty": 2,
            "scheduled_date": future,
            "status": "PENDING",
        }]))

        results = executor.execute_pending()
        assert results == []

    def test_execute_pending_market_closed(self, executor, swap_file):
        """Market closed → swap fails."""
        swap_file.write_text(json.dumps([{
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 1000.0,
            "qty": 2,
            "scheduled_date": "2020-01-01",  # Always in the past
            "status": "PENDING",
        }]))
        executor.client.is_market_open.return_value = False

        results = executor.execute_pending()
        assert len(results) == 1
        assert results[0].status == SwapStatus.FAILED
        assert "Market closed" in results[0].error

    def test_execute_pending_insufficient_buying_power(self, executor, swap_file):
        """Not enough buying power → swap fails."""
        swap_file.write_text(json.dumps([{
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 10000.0,
            "qty": 20,
            "scheduled_date": "2020-01-01",
            "status": "PENDING",
        }]))
        executor.client.is_market_open.return_value = True
        mock_acct = MagicMock()
        mock_acct.buying_power = 100.0  # Less than needed
        executor.client.get_account.return_value = mock_acct

        results = executor.execute_pending()
        assert len(results) == 1
        assert results[0].status == SwapStatus.FAILED
        assert "Insufficient" in results[0].error

    def test_execute_pending_success(self, executor, swap_file):
        """Market open + sufficient buying power → swap executes."""
        swap_file.write_text(json.dumps([{
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 1000.0,
            "qty": 2,
            "scheduled_date": "2020-01-01",
            "status": "PENDING",
        }]))
        executor.client.is_market_open.return_value = True
        mock_acct = MagicMock()
        mock_acct.buying_power = 10000.0
        executor.client.get_account.return_value = mock_acct

        mock_order = MagicMock()
        mock_order.id = "order-abc123"
        executor.client.submit_order.return_value = mock_order

        results = executor.execute_pending()
        assert len(results) == 1
        assert results[0].status == SwapStatus.EXECUTED
        assert results[0].order_id == "order-abc123"
        assert results[0].executed_at is not None

        # Verify order was submitted correctly
        executor.client.submit_order.assert_called_once()
        call = executor.client.submit_order.call_args
        assert call.kwargs["symbol"] == "VOO"
        assert call.kwargs["qty"] == 2.0

    def test_execute_pending_already_executed_skipped(self, executor, swap_file):
        """Already executed swaps are skipped."""
        swap_file.write_text(json.dumps([{
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 1000.0,
            "qty": 2,
            "scheduled_date": "2020-01-01",
            "status": "EXECUTED",
            "executed_at": "2026-04-14T10:00:00",
            "order_id": "order-old",
        }]))

        results = executor.execute_pending()
        assert results == []  # Nothing executed

    def test_execute_pending_multiple_due(self, executor, swap_file):
        """Multiple swaps due today → all executed."""
        swaps = [
            {
                "original_symbol": "AAPL",
                "target_etf": "VOO",
                "amount": 1000.0,
                "qty": 2,
                "scheduled_date": "2020-01-01",
                "status": "PENDING",
            },
            {
                "original_symbol": "MSFT",
                "target_etf": "SPY",
                "amount": 500.0,
                "qty": 1,
                "scheduled_date": "2020-01-01",
                "status": "PENDING",
            },
        ]
        swap_file.write_text(json.dumps(swaps))
        executor.client.is_market_open.return_value = True
        mock_acct = MagicMock()
        mock_acct.buying_power = 100000.0
        executor.client.get_account.return_value = mock_acct
        executor.client.submit_order.return_value = MagicMock(id="order-xyz")

        results = executor.execute_pending()
        assert len(results) == 2

    def test_execute_pending_error_handling(self, executor, swap_file):
        """API error → swap marked as failed."""
        swap_file.write_text(json.dumps([{
            "original_symbol": "AAPL",
            "target_etf": "VOO",
            "amount": 1000.0,
            "qty": 2,
            "scheduled_date": "2020-01-01",
            "status": "PENDING",
        }]))
        executor.client.is_market_open.return_value = True
        mock_acct = MagicMock()
        mock_acct.buying_power = 10000.0
        executor.client.get_account.return_value = mock_acct
        executor.client.submit_order.side_effect = Exception("API rate limit")

        results = executor.execute_pending()
        assert len(results) == 1
        assert results[0].status == SwapStatus.FAILED
        assert "API rate limit" in results[0].error


# =============================================================================
# SwapExecutor — query methods
# =============================================================================

class TestSwapExecutorQuery:
    """Tests for swap query methods."""

    def test_get_pending_swaps(self, executor, swap_file):
        swaps = [
            {"original_symbol": "AAPL", "target_etf": "VOO", "amount": 1000.0,
             "qty": 2, "scheduled_date": "2026-04-20", "status": "PENDING"},
            {"original_symbol": "MSFT", "target_etf": "SPY", "amount": 500.0,
             "qty": 1, "scheduled_date": "2026-04-20", "status": "EXECUTED"},
        ]
        swap_file.write_text(json.dumps(swaps))
        pending = executor.get_pending_swaps()
        assert len(pending) == 1
        assert pending[0].original_symbol == "AAPL"

    def test_get_executed_swaps(self, executor, swap_file):
        swaps = [
            {"original_symbol": "AAPL", "target_etf": "VOO", "amount": 1000.0,
             "qty": 2, "scheduled_date": "2026-04-15", "status": "EXECUTED"},
            {"original_symbol": "MSFT", "target_etf": "SPY", "amount": 500.0,
             "qty": 1, "scheduled_date": "2026-04-15", "status": "PENDING"},
        ]
        swap_file.write_text(json.dumps(swaps))
        executed = executor.get_executed_swaps()
        assert len(executed) == 1
        assert executed[0].original_symbol == "AAPL"

    def test_get_swap_summary(self, executor, swap_file):
        swaps = [
            {"original_symbol": "AAPL", "target_etf": "VOO", "amount": 1000.0,
             "qty": 2, "scheduled_date": "2026-04-15", "status": "EXECUTED"},
            {"original_symbol": "MSFT", "target_etf": "SPY", "amount": 500.0,
             "qty": 1, "scheduled_date": "2026-04-16", "status": "PENDING"},
            {"original_symbol": "GOOGL", "target_etf": "IVV", "amount": 300.0,
             "qty": 1, "scheduled_date": "2026-04-14", "status": "FAILED"},
        ]
        swap_file.write_text(json.dumps(swaps))
        summary = executor.get_swap_summary()
        assert summary["total_swaps"] == 3
        assert summary["executed"] == 1
        assert summary["pending"] == 1
        assert summary["failed"] == 1
        assert summary["total_executed_amount"] == 1000.0
        assert summary["total_pending_amount"] == 500.0

    def test_cancel_pending(self, executor, swap_file):
        swaps = [
            {"original_symbol": "AAPL", "target_etf": "VOO", "amount": 1000.0,
             "qty": 2, "scheduled_date": "2026-04-20", "status": "PENDING"},
            {"original_symbol": "MSFT", "target_etf": "SPY", "amount": 500.0,
             "qty": 1, "scheduled_date": "2026-04-20", "status": "EXECUTED"},
        ]
        swap_file.write_text(json.dumps(swaps))
        cancelled = executor.cancel_pending()
        assert cancelled == 1
        pending = executor.get_pending_swaps()
        assert len(pending) == 0
