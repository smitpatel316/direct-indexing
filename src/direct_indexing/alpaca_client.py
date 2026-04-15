"""
Alpaca client — thin wrapper around alpaca-py SDK.

We delegate to TradingClient for all API calls. This module exists to:
1. Provide a consistent internal interface across our modules
2. Map alpaca-py models to our own dataclasses where convenient
3. Avoid scattering API key configuration across modules

alpaca-py handles:
- Request/response validation via Pydantic v2
- Paper vs live mode via `paper=True` parameter
- Error handling, timeouts, retries (via underlying HTTP client)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.models import Clock as AlpacaClock
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.models import TradeAccount as AlpacaAccount
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Order time in force."""
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    EXT = "ext"


class OrderStatus(Enum):
    """Order status."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class Position:
    """Our internal position representation."""
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float = 0.0
    cost_basis: float = 0.0

    @property
    def loss_amount(self) -> float:
        """HARMFUL loss MAGNITUDE (always positive when in loss).


        This is the amount of unrealized loss you'd realize if you sold.
        Always positive or zero — never negative (a gain is 0, not negative).
        """
        if self.unrealized_pl >= 0:
            return 0.0
        return abs(self.unrealized_pl)

    @property
    def loss_percent(self) -> float:
        """Percent loss from entry price to current price."""
        if self.avg_entry_price <= 0 or self.current_price <= 0:
            return 0.0
        return (
            (self.current_price - self.avg_entry_price)
            / self.avg_entry_price
        ) * 100


@dataclass
class Order:
    """Our internal order representation."""
    id: str
    symbol: str
    side: OrderSide | str
    order_type: OrderType | str
    qty: float | None
    limit_price: float | None
    stop_price: float | None
    status: OrderStatus | str
    filled_at: datetime | None
    created_at: datetime
    extended_hours: bool


@dataclass
class Account:
    """Our internal account representation."""
    buying_power: float
    cash: float
    equity: float
    portfolio_value: float
    last_equity: float
    daytrade_count: int


class AlpacaClient:
    """
    Thin wrapper around alpaca-py TradingClient.

    Provides a stable internal interface while delegating to the official SDK.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        data_url: str = "https://data.alpaca.markets",
        paper: bool = True,
        trading_client: Any = None,
        data_client: Any = None,
    ):
        self.base_url = base_url
        self.data_url = data_url
        self.paper = paper

        # Allow injected clients for testing (dependency injection)
        if trading_client is not None:
            self._trading = trading_client
        elif base_url != "https://paper-api.alpaca.markets":
            self._trading = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=False,
                url_override=base_url,
            )
        else:
            self._trading = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            )

        if data_client is not None:
            self._data = data_client
        else:
            self._data = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    def get_account(self) -> Account:
        """Get account information."""
        raw: AlpacaAccount = self._trading.get_account()
        return Account(
            buying_power=float(raw.buying_power),
            cash=float(raw.cash),
            equity=float(raw.equity),
            portfolio_value=float(raw.portfolio_value),
            last_equity=float(raw.last_equity),
            daytrade_count=raw.daytrade_count,
        )

    # -------------------------------------------------------------------------
    # Positions
    # -------------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        raw_positions: list[AlpacaPosition] = self._trading.get_all_positions()
        return [self._map_position(p) for p in raw_positions]

    def get_position(self, symbol: str) -> Position | None:
        """Get a single position by symbol."""
        try:
            raw: AlpacaPosition = self._trading.get_open_position(symbol)
            return self._map_position(raw)
        except Exception:
            # Alpaca raises if position not found — treat as no position
            return None

    def _map_position(self, raw: AlpacaPosition) -> Position:
        """Map alpaca-py Position to our internal Position dataclass."""
        return Position(
            symbol=raw.symbol,
            qty=float(raw.qty),
            avg_entry_price=float(raw.avg_entry_price),
            market_value=float(raw.market_value or 0),
            unrealized_pl=float(raw.unrealized_pl or 0),
            unrealized_plpc=float(raw.unrealized_plpc or 0),
            current_price=float(raw.current_price or 0),
            cost_basis=float(raw.cost_basis),
        )

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        extended_hours: bool = False,
    ) -> Order:
        """Submit a trading order."""
        side_str = side.value if isinstance(side, OrderSide) else side

        if qty is not None:
            order_data: Any = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=AlpacaOrderSide.BUY if side_str == "buy" else AlpacaOrderSide.SELL,
                time_in_force=AlpacaTimeInForce.DAY,
                extended_hours=extended_hours,
            )
        else:
            raise ValueError("qty is required for market orders")

        raw: AlpacaOrder = self._trading.submit_order(order_data=order_data)
        return self._map_order(raw)

    def get_orders(
        self,
        status: str = "all",
        symbols: list[str] | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get orders, optionally filtered by status and symbols."""
        # Map our status strings to alpaca-py enum values
        status_map = {
            "open": "open",
            "closed": "closed",
            "all": "all",
            "filled": "closed",  # filled orders are in closed
            "new": "open",
            "partially_filled": "open",
        }
        api_status = status_map.get(status, None)
        filter_params = GetOrdersRequest(
            status=api_status,
            symbols=symbols,
            limit=limit,
        )
        raw_orders: list[AlpacaOrder] = self._trading.get_orders(filter=filter_params)
        return [self._map_order(o) for o in raw_orders]

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order by ID."""
        self._trading.cancel_order_by_id(order_id)

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self._trading.cancel_orders()

    def _map_order(self, raw: AlpacaOrder) -> Order:
        """Map alpaca-py Order to our internal Order dataclass."""
        return Order(
            id=str(raw.id),
            symbol=raw.symbol or "",
            side=raw.side.value if raw.side else "",
            order_type=raw.order_type.value if raw.order_type else "",
            qty=float(raw.qty) if raw.qty else None,
            limit_price=float(raw.limit_price) if raw.limit_price else None,
            stop_price=float(raw.stop_price) if raw.stop_price else None,
            status=raw.status.value if raw.status else "",
            filled_at=raw.filled_at,
            created_at=raw.created_at,
            extended_hours=raw.extended_hours or False,
        )

    # -------------------------------------------------------------------------
    # Market data
    # -------------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get historical bar data for a symbol."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            limit=limit,
        )
        bars = self._data.get_stock_bars(request_params)
        records = bars.df.reset_index().to_dict("records")
        return cast(list[dict[str, Any]], records)

    def get_latest_price(self, symbol: str) -> float | None:
        """Get the latest trade price for a symbol."""
        bars = self.get_bars(symbol, timeframe="1Min", limit=1)
        if not bars:
            return None
        return float(bars[0]["close"])

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock: AlpacaClock = self._trading.get_clock()
        return bool(clock.is_open)

    def get_market_status(self) -> dict[str, Any]:
        """Get full market clock status."""
        clock: AlpacaClock = self._trading.get_clock()
        return {
            "is_open": bool(clock.is_open),
            "next_open": clock.next_open.isoformat(),
            "next_close": clock.next_close.isoformat(),
        }
