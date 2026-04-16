"""
Weight bound constraints for portfolio optimization.
"""

import pulp
import pandas as pd
from typing import Dict


class WeightConstraints:
    """
    Manages weight bound constraints for portfolio optimization.

    Ensures positions stay within min/max weight bounds around target.
    For example, if AAPL target_weight = 7% and min_mult = 0.5:
    - Hard minimum: 3.5% (can't go below this)
    - Soft minimum: target - 90% * (target - hard_min) = 3.55%
    - Maximum: target * max_mult
    """

    def __init__(
        self,
        min_weight_multiplier: float = 0.5,
        max_weight_multiplier: float = 2.0,
        soft_limit_percentage: float = 0.90,
    ):
        """
        Args:
            min_weight_multiplier: Minimum weight as fraction of target (default 0.5)
            max_weight_multiplier: Maximum weight as fraction of target (default 2.0)
            soft_limit_percentage: How close to hard min before penalizing (default 0.90)
        """
        self.min_weight_multiplier = min_weight_multiplier
        self.max_weight_multiplier = max_weight_multiplier
        self.soft_limit_percentage = soft_limit_percentage

    def get_bounds(
        self,
        target_weight: float,
        current_weight: float,
    ) -> tuple[float, float, float]:
        """
        Calculate weight bounds for a position.

        Args:
            target_weight: Target weight for the position
            current_weight: Current weight for the position

        Returns:
            Tuple of (soft_min, hard_min, hard_max)
        """
        hard_min = target_weight * self.min_weight_multiplier
        weight_delta = target_weight - hard_min
        soft_min = target_weight - (self.soft_limit_percentage * weight_delta)
        hard_max = target_weight * self.max_weight_multiplier

        return soft_min, hard_min, hard_max

    def add_constraints(
        self,
        prob: pulp.LpProblem,
        sells: Dict[str, pulp.LpVariable],
        buys: Dict[str, pulp.LpVariable],
        tax_lots: pd.DataFrame,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
        total_value: float,
    ) -> None:
        """
        Add weight bound constraints to the optimization problem.

        Args:
            prob: PuLP problem
            sells: Dict mapping tax_lot_id -> LpVariable
            buys: Dict mapping identifier -> LpVariable
            tax_lots: DataFrame with current tax lots
            target_weights: DataFrame with [identifier, target_weight]
            prices: DataFrame with [identifier, price]
            total_value: Total portfolio value
        """
        # Build current holdings by ticker
        ticker_value: Dict[str, float] = {}
        for _, lot in tax_lots.iterrows():
            ticker = lot["identifier"]
            price_for_lot = prices[prices["identifier"] == ticker]["price"].iloc[0]
            val = lot.get("current_value", lot["quantity"] * price_for_lot)
            ticker_value[ticker] = ticker_value.get(ticker, 0.0) + val

        price_map = dict(zip(prices["identifier"], prices["price"]))
        target_map = dict(zip(target_weights["identifier"], target_weights["target_weight"]))

        for ticker, current_val in ticker_value.items():
            current_weight = current_val / total_value if total_value > 0 else 0.0
            target_weight = target_map.get(ticker, 0.0)
            price = price_map.get(ticker, 0.0)

            if target_weight == 0 or price == 0:
                continue

            soft_min, hard_min, hard_max = self.get_bounds(target_weight, current_weight)

            # Calculate current position variables
            ticker_lots = tax_lots[tax_lots["identifier"] == ticker]
            sell_contribution = pulp.lpSum(
                sells[lot.tax_lot_id] for lot in ticker_lots.itertuples()
                if lot.tax_lot_id in sells
            )
            buy_contribution = buys.get(ticker, 0.0)

            # New position value after trades
            new_value = current_val - sell_contribution * price + buy_contribution * price
            new_weight = new_value / total_value

            # Hard minimum constraint
            prob += new_weight >= hard_min, f"weight_min_{ticker}"
            # Hard maximum constraint
            prob += new_weight <= hard_max, f"weight_max_{ticker}"
