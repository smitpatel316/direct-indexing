"""
Tests for TLH (Tax-Loss Harvesting) Engine
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from direct_indexing.tlh import TLHEngine, HarvestResult, WashSaleEntry, CarryforwardEntry
from direct_indexing.config import TLHConfig
from direct_indexing.alpaca_client import Position


@pytest.fixture
def config():
    return TLHConfig(
        enabled=True,
        loss_threshold_percent=5.0,
        min_loss_amount=100.0,
        max_harvests_per_year=10,
        frequency="daily",
        swap_etfs=["VOO", "SPY", "IVV"],
        wash_sale_enabled=True,
        carryforward_enabled=True,
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_positions.return_value = [
        Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,  # $10 loss
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
            current_price=140.0,
            cost_basis=1500.0,
        ),
        Position(
            symbol="MSFT",
            qty=5.0,
            avg_entry_price=300.0,
            market_value=1600.0,  # $100 gain
            unrealized_pl=100.0,
            unrealized_plpc=6.67,
            current_price=320.0,
            cost_basis=1500.0,
        ),
    ]
    return client


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def engine(mock_client, config, temp_data_dir):
    return TLHEngine(mock_client, config, temp_data_dir)


class TestWashSaleTracking:
    """Tests for wash sale tracking."""
    
    def test_record_wash_sale(self, engine):
        entry = engine.record_wash_sale("AAPL", 100.0)
        
        assert entry.symbol == "AAPL"
        assert entry.sold_loss == 100.0
        assert entry.status == "ACTIVE"
        assert entry.wash_sale_end_date > datetime.now()
    
    def test_is_in_wash_sale_period(self, engine):
        engine.record_wash_sale("AAPL", 100.0)
        
        assert engine.is_in_wash_sale_period("AAPL") is True
        assert engine.is_in_wash_sale_period("MSFT") is False
    
    def test_wash_sale_expires(self, engine, temp_data_dir):
        # Create an expired wash sale
        expired_entry = WashSaleEntry(
            symbol="AAPL",
            sold_date=datetime.now() - timedelta(days=60),
            sold_loss=100.0,
            wash_sale_end_date=datetime.now() - timedelta(days=30),
            status="ACTIVE",
        )
        
        # Write directly to file
        wash_data = [{
            "symbol": expired_entry.symbol,
            "sold_date": expired_entry.sold_date.isoformat(),
            "sold_loss": expired_entry.sold_loss,
            "wash_sale_end_date": expired_entry.wash_sale_end_date.isoformat(),
            "status": "ACTIVE",
            "notes": "",
        }]
        
        with open(temp_data_dir / "wash_sales.json", "w") as f:
            json.dump(wash_data, f)
        
        # Reload engine
        engine2 = TLHEngine(engine.client, engine.config, temp_data_dir)
        
        # Update expired
        expired = engine2.update_expired_wash_sales()
        assert expired == 1
        
        # Check status is now EXPIRED
        assert engine2.get_wash_sales("ACTIVE") == []


class TestCarryforward:
    """Tests for carryforward ledger."""
    
    def test_add_to_carryforward(self, engine):
        entry = engine.add_to_carryforward(500.0, "AAPL harvest")
        
        assert entry.amount == 500.0
        assert entry.source == "AAPL harvest"
        assert entry.remaining == 500.0
        assert entry.utilized == 0.0
    
    def test_use_carryforward(self, engine):
        engine.add_to_carryforward(500.0, "AAPL harvest")
        
        used = engine.use_carryforward(200.0, "offset gains")
        
        assert used == 200.0
        assert engine.get_carryforward_balance() == 300.0
    
    def test_carryforward_fifo(self, engine):
        # Add two entries
        engine.add_to_carryforward(300.0, "AAPL harvest")
        engine.add_to_carryforward(200.0, "MSFT harvest")
        
        # Use 400 - should use all of first, 100 of second
        used = engine.use_carryforward(400.0, "offset gains")
        
        assert used == 400.0
        assert engine.get_carryforward_balance() == 100.0
    
    def test_carryforward_exhausted(self, engine):
        engine.add_to_carryforward(100.0, "AAPL harvest")
        
        # Try to use more than available
        used = engine.use_carryforward(200.0, "offset gains")
        
        assert used == 100.0
        assert engine.get_carryforward_balance() == 0.0


class TestPortfolioScanning:
    """Tests for portfolio scanning."""
    
    def test_scan_finds_harvestable(self, engine):
        positions = engine.scan_portfolio()
        
        # AAPL has 6.67% loss - should be harvestable
        assert len(positions) >= 1
        assert any(p.symbol == "AAPL" for p in positions)
    
    def test_scan_excludes_profitable(self, engine):
        positions = engine.scan_portfolio()
        
        # MSFT has gain - should not be harvestable
        assert not any(p.symbol == "MSFT" and p.loss_percent < 0 for p in positions)
    
    def test_scan_respects_min_loss(self, engine):
        # Configure higher minimum
        engine.config.min_loss_amount = 200.0
        
        positions = engine.scan_portfolio()
        
        # AAPL only has $100 loss - below threshold
        assert not any(p.symbol == "AAPL" for p in positions)
    
    def test_scan_respects_threshold(self, engine):
        # Configure higher threshold
        engine.config.loss_threshold_percent = 10.0
        
        positions = engine.scan_portfolio()
        
        # AAPL only has 6.67% loss - below threshold
        assert not any(p.symbol == "AAPL" for p in positions)


class TestHarvestExecution:
    """Tests for harvest execution."""
    
    def test_execute_harvest_success(self, engine, mock_client):
        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
            current_price=140.0,
            cost_basis=1500.0,
        )

        mock_client.submit_order.return_value = {"id": "order123"}

        result = engine.execute_harvest(position)


        assert result.success is True
        assert result.symbol == "AAPL"
        assert result.loss_amount == 100.0
    
    def test_execute_harvest_records_wash_sale(self, engine, mock_client):
        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
            current_price=140.0,
            cost_basis=1500.0,
        )

        mock_client.submit_order.return_value = {"id": "order123"}


        result = engine.execute_harvest(position)

        assert engine.is_in_wash_sale_period("AAPL") is True
    
    def test_execute_harvest_adds_to_carryforward(self, engine, mock_client):
        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
            current_price=140.0,
            cost_basis=1500.0,
        )

        mock_client.submit_order.return_value = {"id": "order123"}

        result = engine.execute_harvest(position)

        balance = engine.get_carryforward_balance()
        assert balance == 100.0
    
    def test_execute_harvest_failure(self, engine, mock_client):
        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
            current_price=140.0,
            cost_basis=1500.0,
        )

        mock_client.submit_order.side_effect = Exception("API Error")


        result = engine.execute_harvest(position)

        assert result.success is False
        assert "API Error" in result.error


class TestSummary:
    """Tests for TLH summary reporting."""
    
    def test_get_summary(self, engine):
        summary = engine.get_summary()
        
        assert "carryforward_balance" in summary
        assert "active_wash_sales" in summary
        assert "harvestable_positions" in summary
        assert "top_losses" in summary
    
    def test_get_ytd_harvested(self, engine):
        engine.add_to_carryforward(500.0, "AAPL harvest")
        engine.add_to_carryforward(300.0, "MSFT harvest")
        
        ytd = engine.get_ytd_harvested()
        
        assert ytd == 800.0