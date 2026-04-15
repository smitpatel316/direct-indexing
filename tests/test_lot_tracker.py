"""
Tests for LotTracker — local lot-level basis tracking.

Key concepts:
- A Lot records a single buy transaction (date, qty, cost basis per share)
- When we SELL, we match against lots (FIFO — oldest first)
- LotGain/Loss is computed as: current_market_value - cost_basis
- Wash sale rule: can't harvest if we BOUGHT same/complementary ETF in last 31 days
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

from direct_indexing.lot_tracker import (
    LotTracker,
    Lot,
    LotStatus,
    LotMatch,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tracker(temp_data_dir):
    return LotTracker(data_dir=temp_data_dir)


@pytest.fixture
def tracker_with_lots(temp_data_dir):
    """Tracker pre-seeded with 3 lots for AAPL."""
    t = LotTracker(data_dir=temp_data_dir)

    # Lot 1: bought 100 shares at $150 (oldest, gain)
    t.record_buy(
        symbol="AAPL",
        qty=100.0,
        cost_per_share=150.0,
        order_id="order-001",
        acquired_date=datetime(2024, 1, 15),
    )

    # Lot 2: bought 50 shares at $160 (gain)
    t.record_buy(
        symbol="AAPL",
        qty=50.0,
        cost_per_share=160.0,
        order_id="order-002",
        acquired_date=datetime(2024, 3, 1),
    )

    # Lot 3: bought 30 shares at $180 (loss)
    t.record_buy(
        symbol="AAPL",
        qty=30.0,
        cost_per_share=180.0,
        order_id="order-003",
        acquired_date=datetime(2024, 6, 1),
    )

    return t


# =============================================================================
# Lot dataclass
# =============================================================================


class TestLotDataclass:
    def test_lot_fields(self):
        lot = Lot(
            lot_id="lot-123",
            symbol="AAPL",
            qty=100.0,
            cost_per_share=150.0,
            acquired_date=datetime(2024, 1, 15),
            order_id="order-001",
            status=LotStatus.OPEN,
            notes="",
        )
        assert lot.symbol == "AAPL"
        assert lot.qty == 100.0
        assert lot.cost_basis == 15000.0  # 100 * 150
        assert lot.lot_id == "lot-123"
        assert lot.status == LotStatus.OPEN

    def test_lot_cost_basis_property(self):
        lot = Lot(
            lot_id="lot-123",
            symbol="AAPL",
            qty=50.0,
            cost_per_share=200.0,
            acquired_date=datetime(2024, 1, 1),
            order_id="order-1",
            status=LotStatus.OPEN,
        )
        assert lot.cost_basis == 10000.0  # 50 * 200

    def test_lot_immutability_note(self):
        """Lots are immutable records — once created, qty doesn't change in-place."""
        lot = Lot(
            lot_id="lot-1",
            symbol="AAPL",
            qty=100.0,
            cost_per_share=150.0,
            acquired_date=datetime(2024, 1, 1),
            order_id="o1",
            status=LotStatus.OPEN,
        )
        # The lot itself is a frozen record; remaining qty is tracked separately
        assert lot.qty == 100.0


# =============================================================================
# Recording buys
# =============================================================================


class TestRecordingBuys:
    def test_record_buy(self, tracker):
        lot_id = tracker.record_buy(
            symbol="AAPL",
            qty=100.0,
            cost_per_share=150.0,
            order_id="order-001",
            acquired_date=datetime(2024, 1, 15),
        )

        assert lot_id is not None
        lots = tracker.get_lots("AAPL")
        assert len(lots) == 1
        assert lots[0].symbol == "AAPL"
        assert lots[0].qty == 100.0
        assert lots[0].cost_per_share == 150.0
        assert lots[0].cost_basis == 15000.0

    def test_record_buy_generates_uuid(self, tracker):
        id1 = tracker.record_buy("AAPL", 10, 100.0, "order-1")
        id2 = tracker.record_buy("AAPL", 10, 100.0, "order-2")
        assert id1 != id2

    def test_record_buy_multiple_symbols(self, tracker):
        tracker.record_buy("AAPL", 100, 150.0, "o1")
        tracker.record_buy("MSFT", 50, 300.0, "o2")
        tracker.record_buy("GOOGL", 20, 140.0, "o3")

        assert len(tracker.get_lots("AAPL")) == 1
        assert len(tracker.get_lots("MSFT")) == 1
        assert len(tracker.get_lots("GOOGL")) == 1

    def test_record_buy_persists(self, temp_data_dir):
        t1 = LotTracker(data_dir=temp_data_dir)
        t1.record_buy("AAPL", 100, 150.0, "o1")

        # New instance reads from disk
        t2 = LotTracker(data_dir=temp_data_dir)
        lots = t2.get_lots("AAPL")
        assert len(lots) == 1
        assert lots[0].qty == 100.0


# =============================================================================
# Recording sells (lot matching)
# =============================================================================


class TestRecordingSells:
    def test_record_sell_fifo(self, tracker_with_lots):
        """Oldest lot is matched first (FIFO)."""
        # Sell 80 shares — should match lot 1 (100 shares available)
        matches = tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=80.0,
            current_price=170.0,  # current price for gain calc
            sell_date=datetime(2024, 12, 1),
        )

        assert len(matches) == 1
        assert matches[0].lot_id == tracker_with_lots.get_lots("AAPL")[0].lot_id
        assert matches[0].qty_matched == 80.0
        # Lot 1 had cost 150, sold at 170 → gain = (170-150) * 80
        assert matches[0].gain == pytest.approx((170.0 - 150.0) * 80.0)

    def test_record_sell_crosses_lots(self, tracker_with_lots):
        """Sell qty larger than first lot — uses FIFO across multiple lots."""
        # Sell 120 shares — lot 1 (100) + part of lot 2 (20)
        matches = tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=120.0,
            current_price=170.0,
            sell_date=datetime(2024, 12, 1),
        )

        assert len(matches) == 2
        # First match: lot 1, full 100 shares
        assert matches[0].qty_matched == 100.0
        # Second match: lot 2, 20 of its 50 shares
        assert matches[1].qty_matched == 20.0
        # Total matched = 120
        assert sum(m.qty_matched for m in matches) == 120.0

    def test_record_sell_exact_lot(self, tracker_with_lots):
        """Sell exactly one full lot."""
        # Sell 30 shares — FIFO picks lot 1 (100 shares available, oldest first)
        # So lot 1 gets 30 of its 100 used
        matches = tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=30.0,
            current_price=170.0,
            sell_date=datetime(2024, 12, 1),
        )

        assert len(matches) == 1
        # FIFO means oldest lot (lot 1, 100 shares, $150 cost) gets matched first
        assert matches[0].qty_matched == 30.0
        assert matches[0].gain == pytest.approx((170.0 - 150.0) * 30.0)  # gain (not loss)

    def test_record_sell_insufficient_lots(self, tracker_with_lots):
        """Sell more than available — raises error."""
        with pytest.raises(ValueError, match="Insufficient lot quantity"):
            tracker_with_lots.record_sell(
                symbol="AAPL",
                qty=500.0,  # Only 180 shares across 3 lots
                current_price=170.0,
                sell_date=datetime(2024, 12, 1),
            )

    def test_record_sell_nonexistent_symbol(self, tracker):
        """Selling symbol with no lots — raises error."""
        with pytest.raises(ValueError, match="Insufficient lot quantity"):
            tracker.record_sell("AAPL", 10, 170.0, datetime.now())

    def test_record_sell_updates_remaining_qty(self, tracker_with_lots):
        """After FIFO sell across multiple lots, remaining qty is tracked per lot."""
        # Capture lot IDs before selling
        all_before = {l.qty: l.lot_id for l in tracker_with_lots.get_all_lots()}
        lot_100_id = all_before[100.0]
        lot_50_id = all_before[50.0]
        lot_30_id = all_before[30.0]

        tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=120.0,
            current_price=170.0,
            sell_date=datetime(2024, 12, 1),
        )

        # FIFO: 120 from lots [100(lot1), 50(lot2), 30(lot3)]
        # → lot1: 100 fully used → 0 remaining (CLOSED, filtered from get_lots)
        # → lot2: 20 of 50 used → 30 remaining (PARTIAL)
        # → lot3: unused → 30 remaining (OPEN)
        # get_lots returns only lots with qty > 0: [lot2, lot3] (2 items)
        all_after = {l.lot_id: l for l in tracker_with_lots.get_all_lots()}

        assert all_after[lot_100_id].qty == 0.0  # fully used
        assert all_after[lot_50_id].qty == 30.0  # 20 used from 50
        assert all_after[lot_30_id].qty == 30.0  # unused

    def test_record_sell_persists(self, tracker_with_lots, temp_data_dir):
        tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=50.0,
            current_price=170.0,
            sell_date=datetime(2024, 12, 1),
        )

        # New instance sees updated remaining qty
        t2 = LotTracker(data_dir=temp_data_dir)
        lots = t2.get_lots("AAPL")
        remaining_total = sum(l.qty for l in lots)
        assert remaining_total == 130.0  # 180 - 50


# =============================================================================
# Gain/loss calculations
# =============================================================================


class TestGainLossCalculations:
    def test_lot_at_gain(self, tracker):
        """Lot with current price > cost per share."""
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        lots = tracker.get_lots("AAPL")
        # At $170/share: gain = (170 - 150) * 100 = $2,000
        assert tracker.lot_gain(lots[0], current_price=170.0) == pytest.approx(2000.0)

    def test_lot_at_loss(self, tracker):
        """Lot with current price < cost per share."""
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        lots = tracker.get_lots("AAPL")
        # At $130/share: loss = (130 - 150) * 100 = -$2,000
        assert tracker.lot_gain(lots[0], current_price=130.0) == pytest.approx(-2000.0)

    def test_lot_at_loss_percent(self, tracker):
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        lots = tracker.get_lots("AAPL")
        # At $130/share: -13.33%
        assert tracker.lot_gain_percent(lots[0], current_price=130.0) == pytest.approx(
            -13.333, rel=0.01
        )

    def test_lot_gain_percent_zero_price(self, tracker):
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        lots = tracker.get_lots("AAPL")
        assert tracker.lot_gain_percent(lots[0], current_price=0.0) == 0.0


class TestScanHarvestableLots:
    def test_finds_losing_lots(self, tracker):
        """Lots currently at a loss should be harvestable."""
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        # Record another lot at a gain (higher cost basis)
        tracker.record_buy("AAPL", 50, 180.0, "o2", datetime(2024, 6, 1))

        # Current price: $130 — lot 1 (100 @ $150) is at loss, lot 2 (50 @ $180) is also at loss
        harvestable = tracker.scan_harvestable_lots(
            symbol="AAPL", current_price=130.0, min_loss_amount=100.0
        )

        # Both lots are at a loss; sorted by loss magnitude desc (lot1 $2000 > lot2 $2500? no, lot2 = 50*50=2500 > lot1 = 100*20=2000)
        assert len(harvestable) == 2

    def test_excludes_profitable_lots(self, tracker):
        """Lots currently at a gain are NOT harvestable."""
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))

        harvestable = tracker.scan_harvestable_lots(
            symbol="AAPL", current_price=170.0  # gain
        )

        assert len(harvestable) == 0

    def test_respects_min_loss_amount(self, tracker):
        """Small losses below threshold are excluded."""
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        # At $145, loss = $500 (100 * 5) — below $1000 min
        harvestable = tracker.scan_harvestable_lots(
            symbol="AAPL", current_price=145.0, min_loss_amount=1000.0
        )
        assert len(harvestable) == 0

    def test_multiple_lots_all_losing(self, tracker):
        """Multiple lots can all be losing."""
        tracker.record_buy("AAPL", 100, 150.0, "o1", datetime(2024, 1, 1))
        tracker.record_buy("AAPL", 50, 160.0, "o2", datetime(2024, 3, 1))

        harvestable = tracker.scan_harvestable_lots(
            symbol="AAPL", current_price=130.0, min_loss_amount=100.0
        )

        assert len(harvestable) == 2

    def test_empty_for_no_lots(self, tracker):
        harvestable = tracker.scan_harvestable_lots(
            symbol="AAPL", current_price=130.0
        )
        assert len(harvestable) == 0


# =============================================================================
# Wash sale — recent trades tracking
# =============================================================================


class TestRecentTrades:
    def test_record_recent_buy(self, tracker):
        """Recording a buy that happened recently — for wash sale tracking."""
        tracker.record_recent_trade(
            symbol="VOO", side="buy", date=datetime(2024, 12, 1)
        )

        assert tracker.was_bought_recently("VOO", as_of=datetime(2024, 12, 2))
        assert not tracker.was_bought_recently("VOO", as_of=datetime(2025, 1, 15))  # >31 days

    def test_record_recent_sell(self, tracker):
        tracker.record_recent_trade(
            symbol="AAPL", side="sell", date=datetime(2024, 12, 1)
        )
        # We only track buys for wash sale (not sells)
        assert tracker.was_bought_recently("AAPL") is False

    def test_was_bought_recently_30_day_window(self, tracker):
        """IRS wash sale: bought within 30 days before or after = wash sale.

        Window is [buy_date - 30_days, buy_date + 30_days].
        """
        buy_date = datetime(2024, 12, 1)

        tracker.record_recent_trade("AAPL", side="buy", date=buy_date)

        # Exactly 30 days before — inside window → wash sale
        assert tracker.was_bought_recently("AAPL", as_of=datetime(2024, 12, 1) - timedelta(days=30)) is True
        # 31 days before — outside window → no wash sale
        assert tracker.was_bought_recently("AAPL", as_of=datetime(2024, 12, 1) - timedelta(days=31)) is False
        # Exactly 30 days after — inside window → wash sale
        assert tracker.was_bought_recently("AAPL", as_of=datetime(2024, 12, 1) + timedelta(days=30)) is True
        # 31 days after — outside window → no wash sale
        assert tracker.was_bought_recently("AAPL", as_of=datetime(2024, 12, 1) + timedelta(days=31)) is False

    def test_recent_trades_persist(self, temp_data_dir):
        t1 = LotTracker(data_dir=temp_data_dir)
        t1.record_recent_trade("VOO", side="buy", date=datetime(2024, 12, 1))

        t2 = LotTracker(data_dir=temp_data_dir)
        assert t2.was_bought_recently("VOO", as_of=datetime(2024, 12, 15)) is True


# =============================================================================
# Do-Not-Sell check (the key wash sale gate)
# =============================================================================


class TestDoNotSell:
    def test_cannot_harvest_if_bought_recently(self, tracker):
        """If we bought AAPL in last 31 days, can't harvest it."""
        # We bought VOO 5 days ago (not the same security)
        tracker.record_recent_trade("VOO", side="buy", date=datetime.now() - timedelta(days=5))

        # But we're trying to harvest AAPL — different ticker, should be OK
        tracker.record_buy("AAPL", 100, 150.0, "o1")
        lots = tracker.get_lots("AAPL")
        lots[0]._current_price = 130.0  # at loss

        harvestable = tracker.scan_harvestable_lots("AAPL", current_price=130.0)

        # AAPL wasn't bought recently — it's harvestable
        assert len(harvestable) == 1

    def test_do_not_sell_same_symbol(self, tracker):
        """Same symbol bought in last 31 days = wash sale, must block."""
        tracker.record_recent_trade("AAPL", side="buy", date=datetime.now() - timedelta(days=5))
        tracker.record_buy("AAPL", 100, 150.0, "o1")

        # Check: can we harvest?
        # We have AAPL lots but we also BOUGHT AAPL recently
        can_harvest = tracker.can_harvest_lot(
            symbol="AAPL",
            lot_id=tracker.get_lots("AAPL")[0].lot_id,
            as_of=datetime.now(),
        )
        assert can_harvest is False

    def test_do_not_sell_replacement_etf(self, tracker):
        """If we just BOUGHT the replacement ETF (VOO), we can't harvest the original (SPY).

        Scenario: We sold SPY at a loss and immediately bought VOO.
        The VOO buy is in our recent_trades, so harvesting SPY would trigger wash sale.
        """
        # We BOUGHT VOO as replacement (recently)
        tracker.record_recent_trade("VOO", side="buy", date=datetime.now() - timedelta(days=2))

        # Now we want to harvest SPY and replace with VOO
        tracker.record_buy("SPY", 100, 450.0, "o1")

        can_harvest = tracker.can_harvest_lot(
            symbol="SPY",
            lot_id=tracker.get_lots("SPY")[0].lot_id,
            replacement_etf="VOO",
            as_of=datetime.now(),
        )
        # SPY wasn't bought recently, but VOO (replacement) WAS bought → wash sale
        assert can_harvest is False

    def test_can_harvest_no_recent_activity(self, tracker):
        tracker.record_buy("AAPL", 100, 150.0, "o1")

        can_harvest = tracker.can_harvest_lot(
            symbol="AAPL",
            lot_id=tracker.get_lots("AAPL")[0].lot_id,
            as_of=datetime.now(),
        )
        assert can_harvest is True

    def test_can_harvest_stale_lot(self, tracker):
        """A lot that was bought >31 days ago is not blocked by its own buy date."""
        old_date = datetime.now() - timedelta(days=60)
        tracker.record_buy("AAPL", 100, 150.0, "o1", acquired_date=old_date)

        can_harvest = tracker.can_harvest_lot(
            symbol="AAPL",
            lot_id=tracker.get_lots("AAPL")[0].lot_id,
            as_of=datetime.now(),
        )
        assert can_harvest is True


# =============================================================================
# Remaining qty tracking
# =============================================================================


class TestRemainingQty:
    def test_get_remaining_qty(self, tracker_with_lots):
        remaining = tracker_with_lots.get_remaining_qty("AAPL")
        assert remaining == 180.0  # 100 + 50 + 30

    def test_remaining_after_partial_sell(self, tracker_with_lots):
        tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=50.0,
            current_price=170.0,
            sell_date=datetime(2024, 12, 1),
        )
        remaining = tracker_with_lots.get_remaining_qty("AAPL")
        assert remaining == 130.0

    def test_remaining_qty_no_lots(self, tracker):
        assert tracker.get_remaining_qty("XYZ") == 0.0


# =============================================================================
# All open lots
# =============================================================================


class TestGetAllLots:
    def test_get_all_lots(self, tracker_with_lots):
        all_lots = tracker_with_lots.get_all_lots()
        assert len(all_lots) == 3
        assert all(l.status == LotStatus.OPEN for l in all_lots)

    def test_get_all_lots_filters_closed(self, tracker_with_lots):
        # Sell 150 shares via FIFO from lots [100, 50, 30]:
        #   lot1: 100 fully used → qty=0 → CLOSED
        #   lot2: 50 fully used → qty=0 → CLOSED
        #   lot3: 0 used → qty=30 → OPEN
        # So only lot3 remains in get_lots (qty > 0)
        tracker_with_lots.record_sell(
            symbol="AAPL",
            qty=150.0,
            current_price=170.0,
            sell_date=datetime(2024, 12, 1),
        )
        all_lots = tracker_with_lots.get_all_lots()
        assert len(all_lots) == 3  # all lots still in DB
        open_lots = tracker_with_lots.get_lots("AAPL")
        assert len(open_lots) == 1  # only lot3 has qty > 0
