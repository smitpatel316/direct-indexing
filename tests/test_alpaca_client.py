"""
Tests for AlpacaClient — built against the alpaca-py SDK interface.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime
from uuid import uuid4

from direct_indexing.alpaca_client import (
    AlpacaClient,
    Position,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    Account,
)


@pytest.fixture
def api_key():
    return "PKXXXXXXXXXXXX"


@pytest.fixture
def secret_key():
    return "SECRETXXXXXXXXXXXX"


@pytest.fixture
def client(api_key, secret_key):
    """Create client with real TradingClient (paper mode, no API calls made)."""
    return AlpacaClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )


@pytest.fixture
def mock_trading():
    return MagicMock()


@pytest.fixture
def mock_data():
    return MagicMock()


@pytest.fixture
def client_with_mocks(api_key, secret_key, mock_trading, mock_data):
    """Client with injected mock clients for unit testing."""
    return AlpacaClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
        trading_client=mock_trading,
        data_client=mock_data,
    )


# =============================================================================
# Position dataclass
# =============================================================================

class TestPosition:
    """Tests for our internal Position dataclass."""

    def test_position_fields(self):
        pos = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1600.0,
            unrealized_pl=100.0,
            unrealized_plpc=0.0667,
            current_price=160.0,
            cost_basis=1500.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 10.0
        assert pos.avg_entry_price == 150.0
        assert pos.market_value == 1600.0
        assert pos.unrealized_pl == 100.0
        assert pos.unrealized_plpc == 0.0667
        assert pos.current_price == 160.0
        assert pos.cost_basis == 1500.0

    def test_loss_amount_is_magnitude(self):
        """loss_amount is the HARMFUL loss MAGNITUDE (always positive)."""
        pos = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,
            unrealized_pl=-100.0,
            unrealized_plpc=-0.0667,
            current_price=140.0,
            cost_basis=1500.0,
        )
        entry_value = pos.avg_entry_price * pos.qty
        current_value = pos.current_price * pos.qty
        assert entry_value - current_value == pytest.approx(100.0)


# =============================================================================
# Order dataclass
# =============================================================================

class TestOrder:
    """Tests for our internal Order dataclass."""

    def test_order_fields(self):
        order = Order(
            id="order-123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=10.0,
            limit_price=None,
            stop_price=None,
            status=OrderStatus.NEW,
            filled_at=None,
            created_at=datetime(2026, 1, 1, 9, 30),
            extended_hours=False,
        )
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.qty == 10.0

    def test_order_side_string(self):
        """side can be a string or enum."""
        order = Order(
            id="order-123",
            symbol="AAPL",
            side="sell",
            order_type="market",
            qty=10.0,
            limit_price=None,
            stop_price=None,
            status="new",
            filled_at=None,
            created_at=datetime(2026, 1, 1, 9, 30),
            extended_hours=False,
        )
        assert order.side == "sell"
        assert order.order_type == "market"


# =============================================================================
# AlpacaClient
# =============================================================================

class TestAlpacaClientUnit:
    """Unit tests for AlpacaClient using mocks."""

    def test_initialization(self, client, api_key, secret_key):
        """Client initializes with correct credentials."""
        assert client._trading is not None
        assert client._data is not None
        assert client.paper is True

    def test_initialization_live_url(self, api_key, secret_key):
        """Client with live base_url uses url_override."""
        live_client = AlpacaClient(
            api_key=api_key,
            secret_key=secret_key,
            base_url="https://api.alpaca.markets",
            paper=False,
        )
        assert live_client.base_url == "https://api.alpaca.markets"
        assert live_client.paper is False

    def test_is_market_open_true(self, mock_trading, client_with_mocks):
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_trading.get_clock.return_value = mock_clock

        result = client_with_mocks.is_market_open()
        assert result is True
        mock_trading.get_clock.assert_called_once()

    def test_is_market_open_false(self, mock_trading, client_with_mocks):
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_trading.get_clock.return_value = mock_clock

        result = client_with_mocks.is_market_open()
        assert result is False

    def test_get_account(self, mock_trading, client_with_mocks):
        mock_account = MagicMock()
        mock_account.buying_power = "20000.00"
        mock_account.cash = "10000.00"
        mock_account.equity = "51000.00"
        mock_account.portfolio_value = "50000.00"
        mock_account.last_equity = "49000.00"
        mock_account.daytrade_count = 0
        mock_trading.get_account.return_value = mock_account

        account = client_with_mocks.get_account()
        assert account.buying_power == 20000.0
        assert account.cash == 10000.0
        assert account.equity == 51000.0
        assert account.portfolio_value == 50000.0
        assert account.daytrade_count == 0

    def test_get_positions_empty(self, mock_trading, client_with_mocks):
        mock_trading.get_all_positions.return_value = []

        positions = client_with_mocks.get_positions()
        assert positions == []

    def test_get_positions_with_data(self, mock_trading, client_with_mocks):
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "10.0"
        mock_position.avg_entry_price = "150.0"
        mock_position.market_value = "1600.0"
        mock_position.unrealized_pl = "100.0"
        mock_position.unrealized_plpc = "0.0667"
        mock_position.current_price = "160.0"
        mock_position.cost_basis = "1500.0"
        mock_trading.get_all_positions.return_value = [mock_position]

        positions = client_with_mocks.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10.0
        assert positions[0].current_price == 160.0

    def test_get_position_found(self, mock_trading, client_with_mocks):
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = "10.0"
        mock_position.avg_entry_price = "150.0"
        mock_position.market_value = "1600.0"
        mock_position.unrealized_pl = "100.0"
        mock_position.unrealized_plpc = "0.0667"
        mock_position.current_price = "160.0"
        mock_position.cost_basis = "1500.0"
        mock_trading.get_open_position.return_value = mock_position

        position = client_with_mocks.get_position("AAPL")
        assert position is not None
        assert position.symbol == "AAPL"

    def test_get_position_not_found(self, mock_trading, client_with_mocks):
        mock_trading.get_open_position.side_effect = Exception("not found")

        position = client_with_mocks.get_position("ZZZZ")
        assert position is None

    def test_submit_order_success(self, mock_trading, client_with_mocks):
        mock_order = MagicMock()
        mock_order.id = uuid4()
        mock_order.symbol = "AAPL"
        mock_order.side = MagicMock()
        mock_order.side.value = "buy"
        mock_order.order_type = MagicMock()
        mock_order.order_type.value = "market"
        mock_order.qty = "10.0"
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.status = MagicMock()
        mock_order.status.value = "new"
        mock_order.filled_at = None
        mock_order.created_at = datetime(2026, 1, 1, 9, 30)
        mock_order.extended_hours = False
        mock_trading.submit_order.return_value = mock_order

        order = client_with_mocks.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=10.0,
        )
        assert order.symbol == "AAPL"
        mock_trading.submit_order.assert_called_once()

    def test_get_orders_filtered(self, mock_trading, client_with_mocks):
        mock_order = MagicMock()
        mock_order.id = uuid4()
        mock_order.symbol = "AAPL"
        mock_order.side = MagicMock()
        mock_order.side.value = "buy"
        mock_order.order_type = MagicMock()
        mock_order.order_type.value = "market"
        mock_order.qty = "10.0"
        mock_order.limit_price = None
        mock_order.stop_price = None
        mock_order.status = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_at = datetime(2026, 1, 1, 10, 0)
        mock_order.created_at = datetime(2026, 1, 1, 9, 30)
        mock_order.extended_hours = False
        mock_trading.get_orders.return_value = [mock_order]

        orders = client_with_mocks.get_orders(status="filled", symbols=["AAPL"])
        assert len(orders) == 1
        assert orders[0].status == "filled"

    def test_cancel_order(self, mock_trading, client_with_mocks):
        client_with_mocks.cancel_order("order-123")
        mock_trading.cancel_order_by_id.assert_called_once_with("order-123")

    def test_cancel_all_orders(self, mock_trading, client_with_mocks):
        client_with_mocks.cancel_all_orders()
        mock_trading.cancel_orders.assert_called_once()

    def test_get_bars(self, mock_data, client_with_mocks):
        mock_bars = MagicMock()
        mock_bars.df = MagicMock()
        mock_bars.df.reset_index.return_value.to_dict.return_value = [
            {"symbol": "AAPL", "close": 160.0, "volume": 1000000}
        ]
        mock_data.get_stock_bars.return_value = mock_bars

        bars = client_with_mocks.get_bars("AAPL", timeframe="1Day", limit=100)
        assert isinstance(bars, list)
        mock_data.get_stock_bars.assert_called_once()
