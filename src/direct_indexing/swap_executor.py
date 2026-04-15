"""
Swap Execution Engine — executes replacement ETF buys after TLH harvests.

After harvesting a loss (selling a stock), we buy a similar ETF (VOO/SPY/IVV)
to maintain market exposure. This module handles that execution.

Wash sale compliance: we sell stock X at a loss and immediately buy ETF Y.
Since ETF Y is not "substantially identical" to stock X, this doesn't
trigger the wash sale rule (30-day repurchase restriction).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class SwapStatus(Enum):
    """Status of a scheduled swap."""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class SwapRecord:
    """A single swap record."""
    original_symbol: str
    target_etf: str
    amount: float
    qty: int
    scheduled_date: str
    status: SwapStatus
    executed_at: str | None = None
    order_id: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SwapRecord":
        return cls(
            original_symbol=d["original_symbol"],
            target_etf=d["target_etf"],
            amount=float(d["amount"]),
            qty=int(d["qty"]),
            scheduled_date=d["scheduled_date"],
            status=SwapStatus(d.get("status", "PENDING")),
            executed_at=d.get("executed_at"),
            order_id=d.get("order_id"),
            error=d.get("error"),
        )

    def to_dict(self) -> dict:
        return {
            "original_symbol": self.original_symbol,
            "target_etf": self.target_etf,
            "amount": self.amount,
            "qty": self.qty,
            "scheduled_date": self.scheduled_date,
            "status": self.status.value,
            "executed_at": self.executed_at,
            "order_id": self.order_id,
            "error": self.error,
        }


class SwapExecutor:
    """
    Executes replacement ETF buys after TLH harvests.

    Reads pending swaps from data/pending_swaps.json and executes them.

    Usage:
        executor = SwapExecutor(client, data_dir=Path("data"))
        results = executor.execute_pending()
        for result in results:
            print(f"{result.original_symbol} → {result.target_etf}: {result.status}")
    """

    def __init__(
        self,
        alpaca_client,  # AlpacaClient
        data_dir: Path = Path("data"),
    ):
        self.client = alpaca_client
        self.data_dir = data_dir
        self.swap_file = data_dir / "pending_swaps.json"

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load_swaps(self) -> list[SwapRecord]:
        """Load pending swaps from JSON file."""
        if not self.swap_file.exists():
            return []
        with open(self.swap_file) as f:
            data = json.load(f)
        return [SwapRecord.from_dict(d) for d in data]

    def _save_swaps(self, swaps: list[SwapRecord]) -> None:
        """Save swaps to JSON file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.swap_file, "w") as f:
            json.dump([s.to_dict() for s in swaps], f, indent=2)

    def _now_str(self) -> str:
        return datetime.now().isoformat()

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def execute_pending(self) -> list[SwapRecord]:
        """
        Execute all pending swaps that are due (scheduled_date <= today).

        Returns list of SwapRecords with updated status.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        swaps = self._load_swaps()
        updated = []

        for swap in swaps:
            if swap.status != SwapStatus.PENDING:
                continue
            if swap.scheduled_date > today:
                continue  # Not due yet

            result = self._execute_swap(swap)
            updated.append(result)

        if updated:
            # Merge updated records back into full list and save
            swap_by_key = {
                (s.original_symbol, s.scheduled_date): s for s in swaps
            }
            for u in updated:
                key = (u.original_symbol, u.scheduled_date)
                swap_by_key[key] = u
            self._save_swaps(list(swap_by_key.values()))

        return updated

    def _execute_swap(self, swap: SwapRecord) -> SwapRecord:
        """Execute a single swap."""
        try:
            # Check if market is open
            if not self.client.is_market_open():
                swap.status = SwapStatus.FAILED
                swap.error = "Market closed"
                return swap

            # Check buying power
            account = self.client.get_account()
            if account.buying_power < swap.amount:
                swap.status = SwapStatus.FAILED
                swap.error = (
                    f"Insufficient buying power"
                    f" (${account.buying_power:.2f} < ${swap.amount:.2f})"
                )
                return swap

            # Submit buy order for target ETF
            from .alpaca_client import OrderSide, OrderType
            order = self.client.submit_order(
                symbol=swap.target_etf,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=float(swap.qty),
            )

            swap.status = SwapStatus.EXECUTED
            swap.executed_at = self._now_str()
            swap.order_id = order.id if order else None
            return swap

        except Exception as e:
            swap.status = SwapStatus.FAILED
            swap.error = str(e)
            return swap

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def get_pending_swaps(self) -> list[SwapRecord]:
        """Return all pending (not yet executed) swaps."""
        return [s for s in self._load_swaps() if s.status == SwapStatus.PENDING]

    def get_executed_swaps(self) -> list[SwapRecord]:
        """Return all executed swaps."""
        return [s for s in self._load_swaps() if s.status == SwapStatus.EXECUTED]

    def get_swap_summary(self) -> dict:
        """Return a summary of swap activity."""
        swaps = self._load_swaps()
        pending = [s for s in swaps if s.status == SwapStatus.PENDING]
        executed = [s for s in swaps if s.status == SwapStatus.EXECUTED]
        failed = [s for s in swaps if s.status == SwapStatus.FAILED]

        total_executed = sum(s.amount for s in executed)
        total_pending = sum(s.amount for s in pending)

        return {
            "total_swaps": len(swaps),
            "pending": len(pending),
            "executed": len(executed),
            "failed": len(failed),
            "total_executed_amount": total_executed,
            "total_pending_amount": total_pending,
        }

    def cancel_pending(self) -> int:
        """Cancel all pending swaps. Returns count cancelled."""
        swaps = self._load_swaps()
        cancelled = 0
        for swap in swaps:
            if swap.status == SwapStatus.PENDING:
                swap.status = SwapStatus.CANCELLED
                cancelled += 1
        if cancelled:
            self._save_swaps(swaps)
        return cancelled
