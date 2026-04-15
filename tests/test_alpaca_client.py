"""
Tests for AlpacaClient
"""

import pytest
from unittest.mock import patch, MagicMock
from direct_indexing.alpaca_client import (
    AlpacaClient, Position, Order, OrderSide, 
    OrderType, TimeInForce, Account
)
from direct_indexing.config import AlpacaConfig


@pytest.fixture
def config():
    return AlpacaConfig(
        api_key="test_key",
        api_secret="test_secret",
        base_url="https://paper-api.alpaca.markets",
        data_url="https://data.alpaca.markets",
        paper_trading=True,
    )


@pytest.fixture
def client(config):
    return AlpacaClient(config)


class TestPosition:
    """Tests for Position dataclass."""
    
    def test_current_price_calculation(self):
        pos = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1600.0,
            unrealized_pl=100.0,
            unrealized_plpc=6.67,
        )
        assert pos.current_price == 160.0
    
    def test_loss_percent_positive(self):
        pos = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
        )
        assert pos.loss_percent == pytest.approx(-6.67, 0.01)
    
    def test_loss_amount_is_magnitude(self):
        # loss_amount is the HARMFUL loss MAGNITUDE (always positive)
        # It represents how much was lost, not the signed P&L
        pos = Position(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.0,
            market_value=1400.0,  # current = 140
            unrealized_pl=-100.0,
            unrealized_plpc=-6.67,
        )
        # (150 - 140) * 10 = 100 (positive magnitude)
        assert pos.loss_amount == 100.0
        assert pos.loss_amount == abs(pos.unrealized_pl)


class TestAlpacaClient:
    """Tests for AlpacaClient."""
    
    def test_initialization(self, client):
        assert client.config.api_key == "test_key"
        assert client.config.api_secret == "test_secret"
        assert "test_key" in client.headers["APCA-API-KEY-ID"]
    
    def test_is_market_open_success(self, client):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"is_open": True}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            result = client.is_market_open()
            assert result is True
    
    def test_is_market_open_closed(self, client):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"is_open": False}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            result = client.is_market_open()
            assert result is False
    
    def test_get_account(self, client):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "cash": "10000.00",
                "buying_power": "20000.00",
                "portfolio_value": "50000.00",
                "equity": "51000.00",
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            account = client.get_account()
            assert account.cash == 10000.0
            assert account.buying_power == 20000.0
            assert account.portfolio_value == 50000.0
            assert account.equity == 51000.0
    
    def test_get_positions_empty(self, client):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_get.return_value = mock_response
            
            positions = client.get_positions()
            assert positions == []
    
    def test_get_positions_with_data(self, client):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "symbol": "AAPL",
                    "qty": "10.0",
                    "avg_entry_price": "150.0",
                    "market_value": "1600.0",
                    "unrealized_pl": "100.0",
                    "unrealized_plpc": "0.0667",
                }
            ]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            positions = client.get_positions()
            assert len(positions) == 1
            assert positions[0].symbol == "AAPL"
            assert positions[0].qty == 10.0


class TestOrder:
    """Tests for Order dataclass."""
    
    def test_order_creation(self):
        order = Order(
            symbol="AAPL",
            qty=10.0,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
        )
        assert order.symbol == "AAPL"
        assert order.qty == 10.0
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
    
    def test_order_defaults(self):
        order = Order(symbol="AAPL")
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.time_in_force == TimeInForce.DAY
        assert order.qty is None
        assert order.limit_price is None