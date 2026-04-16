"""
Wash sale constraint handling for tax-loss harvesting.
Tracks ticker-level restrictions during the 31-day wash sale window.
"""

from datetime import date, timedelta
from typing import Dict, Set, Optional
import pandas as pd


class WashSaleConstraints:
    """
    Manages wash sale restrictions for direct indexing.

    Wash sale rule (IRS Publication 550):
    A wash sale occurs when you sell a security at a loss and buy a
    "substantially identical" security within 30 days before or after
    the sale (61-day window: 30 days before + 30 days after + sale day).

    For direct indexing, we track:
    - Tickers we CANNOT buy (sold at loss recently)
    - Tickers we CANNOT sell (bought recently at loss)

    The 31-day restriction window: if we sell ticker X at a loss today,
    we cannot buy X again until tomorrow (31st day after sale).
    """

    def __init__(
        self,
        current_date: date,
        wash_window_days: int = 31,
    ):
        """
        Initialize wash sale constraints.

        Args:
            current_date: The date for which to evaluate restrictions
            wash_window_days: Days to restrict (default 31 = safe beyond wash period)
        """
        self.current_date = current_date
        self.wash_window_days = wash_window_days

        # Tickers restricted from buying (sold at loss within window)
        self._restricted_from_buying: Dict[str, date] = {}

        # Tickers restricted from selling (bought at loss within window)
        self._restricted_from_selling: Dict[str, date] = {}

    def add_sell_restriction(
        self,
        ticker: str,
        sell_date: date,
        restriction_end_date: Optional[date] = None,
    ) -> None:
        """
        Add a ticker to the buy restriction list after selling at a loss.

        Args:
            ticker: The ticker symbol
            sell_date: The date of the sale
            restriction_end_date: When the restriction ends (default: wash_window_days after)
        """
        if restriction_end_date is None:
            restriction_end_date = sell_date + timedelta(days=self.wash_window_days)
        self._restricted_from_buying[ticker] = restriction_end_date

    def add_buy_restriction(
        self,
        ticker: str,
        buy_date: date,
        restriction_end_date: Optional[date] = None,
    ) -> None:
        """
        Add a ticker to the sell restriction list after buying at a loss.

        Args:
            ticker: The ticker symbol
            buy_date: The date of the purchase
            restriction_end_date: When the restriction ends
        """
        if restriction_end_date is None:
            restriction_end_date = buy_date + timedelta(days=self.wash_window_days)
        self._restricted_from_selling[ticker] = restriction_end_date

    def is_restricted_from_buying(self, ticker: str) -> bool:
        """
        Check if ticker is restricted from being bought.

        Args:
            ticker: The ticker to check

        Returns:
            True if cannot buy, False otherwise
        """
        if ticker not in self._restricted_from_buying:
            return False
        return self.current_date <= self._restricted_from_buying[ticker]

    def is_restricted_from_selling(self, ticker: str) -> bool:
        """
        Check if ticker is restricted from being sold.

        Args:
            ticker: The ticker to check

        Returns:
            True if cannot sell, False otherwise
        """
        if ticker not in self._restricted_from_selling:
            return False
        return self.current_date <= self._restricted_from_selling[ticker]

    def get_buy_restricted_tickers(self) -> Set[str]:
        """Return set of all tickers currently restricted from buying."""
        return {
            t for t, end in self._restricted_from_buying.items()
            if self.current_date <= end
        }

    def get_sell_restricted_tickers(self) -> Set[str]:
        """Return set of all tickers currently restricted from selling."""
        return {
            t for t, end in self._restricted_from_selling.items()
            if self.current_date <= end
        }

    def add_constraints_to_problem(self, prob, buys, sells, tax_lots) -> None:
        """
        Add wash sale constraints to the optimization problem.

        For each ticker restricted from buying:
        - Set buy variable to 0

        For each ticker restricted from selling:
        - Set all sell variables for that ticker to 0

        Args:
            prob: PuLP problem
            buys: Dict mapping ticker -> LpVariable
            sells: Dict mapping tax_lot_id -> LpVariable
            tax_lots: DataFrame with tax lot info
        """
        # Can't buy restricted tickers
        for ticker in self.get_buy_restricted_tickers():
            if ticker in buys:
                prob += buys[ticker] == 0, f"wash_buy_{ticker}"

        # Can't sell restricted tickers
        for ticker in self.get_sell_restricted_tickers():
            ticker_lots = tax_lots[tax_lots["identifier"] == ticker]
            for lot_id in ticker_lots["tax_lot_id"]:
                if lot_id in sells:
                    prob += sells[lot_id] == 0, f"wash_sell_{lot_id}"

    def cleanup_expired(self) -> None:
        """Remove expired restrictions."""
        self._restricted_from_buying = {
            t: d for t, d in self._restricted_from_buying.items()
            if self.current_date <= d
        }
        self._restricted_from_selling = {
            t: d for t, d in self._restricted_from_selling.items()
            if self.current_date <= d
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export current restrictions as DataFrame."""
        rows = []
        for ticker, end in self._restricted_from_buying.items():
            rows.append({
                "identifier": ticker,
                "restriction_type": "BUY_RESTRICTED",
                "end_date": end,
                "active": self.current_date <= end,
            })
        for ticker, end in self._restricted_from_selling.items():
            rows.append({
                "identifier": ticker,
                "restriction_type": "SELL_RESTRICTED",
                "end_date": end,
                "active": self.current_date <= end,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["identifier", "restriction_type", "end_date", "active"]
        )
