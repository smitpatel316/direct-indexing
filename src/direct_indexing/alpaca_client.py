"""
Alpaca API Client
Modern async/sync client for Alpaca trading API.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests  # type: ignore[import-untyped]

from .config import AlpacaConfig


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float

    @property
    def current_price(self) -> float:
        if self.qty > 0:
            return self.market_value / self.qty
        return 0.0

    @property
    def loss_percent(self) -> float:
        if self.avg_entry_price > 0:
            pct = (self.current_price - self.avg_entry_price) / self.avg_entry_price
            return pct * 100
        return 0.0

    @property
    def loss_amount(self) -> float:
        return (self.avg_entry_price - self.current_price) * self.qty


@dataclass
class Order:
    """Trading order."""
    symbol: str
    qty: float | None = None
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    extended_hours: bool = False


@dataclass
class Account:
    """Alpaca account info."""
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float


class AlpacaClient:
    """Modern Alpaca API client with sync requests."""

    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.base_url = config.base_url
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.api_secret,
        }

    @classmethod
    def from_config(cls, config: AlpacaConfig) -> "AlpacaClient":
        return cls(config)

    def get_account(self) -> Account:
        """Get account information."""
        response = requests.get(
            f"{self.base_url}/v2/account",
            headers=self.headers
        )
        response.raise_for_status()
        data = response.json()

        return Account(
            cash=float(data.get("cash", 0)),
            buying_power=float(data.get("buying_power", 0)),
            portfolio_value=float(data.get("portfolio_value", 0)),
            equity=float(data.get("equity", 0)),
        )

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        response = requests.get(
            f"{self.base_url}/v2/positions",
            headers=self.headers
        )

        if response.status_code == 204:
            return []

        response.raise_for_status()
        data = response.json()

        return [
            Position(
                symbol=pos["symbol"],
                qty=float(pos["qty"]),
                avg_entry_price=float(pos["avg_entry_price"]),
                market_value=float(pos["market_value"]),
                unrealized_pl=float(pos["unrealized_pl"]),
                unrealized_plpc=float(pos["unrealized_plpc"]) * 100,
            )
            for pos in data
        ]

    def get_position(self, symbol: str) -> Position | None:
        """Get a specific position."""
        response = requests.get(
            f"{self.base_url}/v2/positions/{symbol}",
            headers=self.headers
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        return Position(
            symbol=data["symbol"],
            qty=float(data["qty"]),
            avg_entry_price=float(data["avg_entry_price"]),
            market_value=float(data["market_value"]),
            unrealized_pl=float(data["unrealized_pl"]),
            unrealized_plpc=float(data["unrealized_plpc"]) * 100,
        )

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get real-time quote for a symbol."""
        response = requests.get(
            f"{self.config.data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=self.headers
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        return data.get("quote", {})  # type: ignore[no-any-return]

    def submit_order(self, order: Order) -> dict[str, Any]:
        """Submit a trading order."""
        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "side": (
                order.side.value
                if isinstance(order.side, OrderSide)
                else order.side
            ),
            "type": order.order_type.value if isinstance(
                order.order_type, OrderType
            ) else order.order_type,
            "time_in_force": order.time_in_force.value if isinstance(
                order.time_in_force, TimeInForce
            ) else order.time_in_force,
        }

        if order.qty is not None:
            payload["qty"] = str(order.qty)

        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)

        if order.stop_price is not None:
            payload["stop_price"] = str(order.stop_price)

        if order.extended_hours:
            payload["extended_hours"] = True

        response = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def get_orders(self, status: str = "all") -> list[dict[str, Any]]:
        """Get orders by status (open, closed, all)."""
        response = requests.get(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            params={"status": status}
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        response = requests.delete(
            f"{self.base_url}/v2/orders/{order_id}",
            headers=self.headers
        )
        if response.status_code != 204:
            response.raise_for_status()

    def get_bars(
        self, symbol: str, timeframe: str = "1Day", limit: int = 100
    ) -> list[dict]:
        """Get historical bars for a symbol."""
        response = requests.get(
            f"{self.config.data_url}/v2/stocks/{symbol}/bars",
            headers=self.headers,
            params={"timeframe": timeframe, "limit": limit}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("bars", [])  # type: ignore[no-any-return]

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        response = requests.get(
            f"{self.base_url}/v2/clock",
            headers=self.headers
        )
        response.raise_for_status()
        data = response.json()
        return data.get("is_open", False)  # type: ignore[no-any-return]

    def get_market_status(self) -> dict[str, Any]:
        """Get full market status."""
        response = requests.get(
            f"{self.base_url}/v2/clock",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
