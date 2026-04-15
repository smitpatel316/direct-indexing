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

class TestSyncPositionsToLots:
    """Tests for position-to-lot bootstrap sync."""

    def test_sync_creates_lot_when_none_exist(self, mock_client, config, temp_data_dir):
        """When symbol is in Alpaca but not in lot tracker, bootstrap a lot."""
        # Only AAPL position exists in mock_client
        engine = TLHEngine(mock_client, config, temp_data_dir)
        engine._sync_positions_to_lots()

        lots = engine._lot_tracker.get_lots("AAPL")
        assert len(lots) == 1
        assert lots[0].qty == 10.0
        assert lots[0].cost_per_share == 150.0

    def test_sync_adds_supplemental_lot_for_delta(self, mock_client, config, temp_data_dir):
        """When existing lots are less than Alpaca position qty, add supplemental lot."""
        engine = TLHEngine(mock_client, config, temp_data_dir)
        engine._sync_positions_to_lots()

        # Record a partial sell externally (Alpaca still shows 10, lot tracker thinks 7)
        engine._lot_tracker.record_sell(
            symbol="AAPL",
            qty=3.0,
            current_price=140.0,
        )

        # Re-sync — should detect delta and add supplemental lot
        engine._sync_positions_to_lots()

        lots = engine._lot_tracker.get_lots("AAPL")
        total_qty = sum(l.qty for l in lots)
        assert total_qty == 10.0  # Alpaca position qty

    def test_sync_does_nothing_when_lots_match_position(self, mock_client, config, temp_data_dir):
        """When existing lots exactly match position qty, do nothing."""
        engine = TLHEngine(mock_client, config, temp_data_dir)
        engine._sync_positions_to_lots()

        initial_lot_count = len(engine._lot_tracker.get_lots("AAPL"))

        # Re-sync — lots match position, nothing should change
        engine._sync_positions_to_lots()

        final_lot_count = len(engine._lot_tracker.get_lots("AAPL"))
        assert final_lot_count == initial_lot_count


class TestHarvestQty:
    """Test that execute_harvest sells only harvestable qty, not entire position."""

    def test_execute_harvest_sells_only_losing_lots_qty(self, mock_client, config, temp_data_dir):
        """execute_harvest sells only harvestable losing-lot qty, not position.qty.

        TLHEngine.__init__ bootstraps a lot from the position (20@160).
        We add a test lot at 5@180. At harvest price $175:
          - Bootstrap lot: 20@160, FMV=3500, FMV-cost=+300 → GAIN, not harvestable
          - Test lot: 5@180, FMV=875, cost=900 → -25 → LOSS, harvestable
        Old bug: submit qty=position.qty=20. Fixed: submit qty=total_qty of losing lots=5.

        Note: engine fixture uses min_loss_amount=100, so to test with a $25 loss lot
        we create a fresh config with min_loss_amount=0.
        """
        # Override config with low min_loss_amount to capture the $25 loss lot
        low_threshold_config = TLHConfig(
            loss_threshold_percent=5.0,
            min_loss_amount=0.0,  # Allow any non-zero loss
            swap_etfs=["VOO"],
            wash_sale_enabled=True,
            carryforward_enabled=True,
        )
        engine = TLHEngine(mock_client, low_threshold_config, temp_data_dir)

        # Bootstrap lot: 20@160 (AAPL position from mock_client)
        # At price $175: 20@160 → +$300 GAIN → not harvestable
        # Add this lot: 5@180 → at $175: (175-180)*5 = -$25 LOSS → harvestable
        engine._lot_tracker.record_buy("AAPL", 5.0, 180.0, "loss-lot")

        mock_client.get_latest_price.return_value = 175.0

        # Create position with qty=20 (matching mock_client.get_positions bootstrap)
        # and current_price=0 so execute_harvest fetches 175 via get_latest_price
        position = Position(
            symbol="AAPL",
            qty=20.0,
            avg_entry_price=160.0,
            market_value=3500.0,
            unrealized_pl=+300.0,
            unrealized_plpc=+9.4,
            current_price=0,  # Forces get_latest_price($175) in execute_harvest
            cost_basis=3200.0,
        )
        mock_client.submit_order.return_value = {"id": "order-123"}

        result = engine.execute_harvest(position, replacement_etf="VOO")

        submit_call = mock_client.submit_order.call_args
        assert submit_call is not None, f"submit_order was not called. error={result.error}"
        submitted_qty = submit_call[1]["qty"]
        assert submitted_qty == 5.0, f"Should sell 5 (harvestable losing lot), got {submitted_qty}"
        assert result.success is True
        assert result.lots_harvested == 1


class TestGainHarvesting:
    """Tests for gain harvesting feature."""

    def test_scan_gain_positions_disabled_by_default(self, mock_client, config, temp_data_dir):
        """When max_gain_to_sell=0 (default), no gain positions are scanned."""
        engine = TLHEngine(mock_client, config, temp_data_dir)
        assert engine.config.max_gain_to_sell == 0.0
        result = engine.scan_gain_positions()
        assert result == []

    def test_scan_gain_positions_finds_large_gain(
        self, mock_client, config, temp_data_dir
    ):
        """Positions with gain above max_gain_to_sell are found."""
        gain_config = TLHConfig(
            loss_threshold_percent=5.0,
            min_loss_amount=0.0,
            max_gain_to_sell=20.0,  # Harvest gains > 20%
            min_gain_amount=100.0,
            swap_etfs=["VOO"],
            wash_sale_enabled=True,
            carryforward_enabled=True,
        )
        engine = TLHEngine(mock_client, gain_config, temp_data_dir)

        # Override get_latest_price to return $200 (big gain)
        mock_client.get_latest_price.return_value = 200.0

        # Position: AAPL 10@150, current=$200 → 33% gain, +$500 gain
        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=2000.0,
            unrealized_pl=+500.0,
            unrealized_plpc=+33.3,
            current_price=0,  # Forces get_latest_price
            cost_basis=1500.0,
        )
        mock_client.get_positions.return_value = [position]

        gain_positions = engine.scan_gain_positions()
        assert len(gain_positions) == 1
        assert gain_positions[0].symbol == "AAPL"

    def test_scan_gain_positions_excludes_small_gain(
        self, mock_client, config, temp_data_dir
    ):
        """Positions with gain below min_gain_amount are excluded."""
        gain_config = TLHConfig(
            loss_threshold_percent=5.0,
            min_loss_amount=0.0,
            max_gain_to_sell=50.0,
            min_gain_amount=1000.0,  # Need at least $1000 gain
            swap_etfs=["VOO"],
            wash_sale_enabled=True,
            carryforward_enabled=True,
        )
        engine = TLHEngine(mock_client, gain_config, temp_data_dir)

        mock_client.get_latest_price.return_value = 155.0  # Only $50 gain

        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1550.0,
            unrealized_pl=+50.0,
            unrealized_plpc=+3.3,
            current_price=0,
            cost_basis=1500.0,
        )
        mock_client.get_positions.return_value = [position]

        gain_positions = engine.scan_gain_positions()
        assert len(gain_positions) == 0

    def test_execute_gain_harvest_success(
        self, mock_client, config, temp_data_dir
    ):
        """execute_gain_harvest sells at gain and records new cost basis."""
        gain_config = TLHConfig(
            loss_threshold_percent=5.0,
            min_loss_amount=0.0,
            max_gain_to_sell=20.0,
            min_gain_amount=100.0,
            swap_etfs=["VOO"],
            wash_sale_enabled=True,
            carryforward_enabled=True,
        )
        engine = TLHEngine(mock_client, gain_config, temp_data_dir)

        mock_client.get_latest_price.return_value = 200.0

        # Position: 10@150, now $200 → $500 gain
        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=2000.0,
            unrealized_pl=+500.0,
            unrealized_plpc=+33.3,
            current_price=0,
            cost_basis=1500.0,
        )
        mock_client.get_positions.return_value = [position]
        mock_client.submit_order.return_value = {"id": "gain-order-123"}

        result = engine.execute_gain_harvest(position, replacement_etf="VOO")

        assert result.success is True
        assert result.symbol == "AAPL"
        assert result.gain_amount == 500.0
        assert result.qty_sold == 10.0
        assert result.new_cost_basis == 200.0
        assert result.swap_target == "VOO"

        # Verify sell order was submitted
        mock_client.submit_order.assert_called_once()
        call_kwargs = mock_client.submit_order.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["side"].value == "sell"
        assert call_kwargs["qty"] == 10.0

    def test_execute_gain_harvest_no_gain(self, mock_client, config, temp_data_dir):
        """When position has no gain, harvest fails gracefully."""
        gain_config = TLHConfig(
            loss_threshold_percent=5.0,
            min_loss_amount=0.0,
            max_gain_to_sell=20.0,
            min_gain_amount=100.0,
            swap_etfs=["VOO"],
            wash_sale_enabled=True,
            carryforward_enabled=True,
        )
        engine = TLHEngine(mock_client, gain_config, temp_data_dir)

        mock_client.get_latest_price.return_value = 120.0  # Below cost basis

        position = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1200.0,
            unrealized_pl=-300.0,
            unrealized_plpc=-20.0,
            current_price=0,
            cost_basis=1500.0,
        )
        mock_client.get_positions.return_value = [position]

        result = engine.execute_gain_harvest(position, replacement_etf="VOO")

        assert result.success is False
        assert "No gain lots found" in (result.error or "")
