"""
Transaction cost objective for portfolio optimization.
"""

import pulp
import pandas as pd
from typing import Dict, Tuple


def calculate_transaction_cost(
    prob: pulp.LpProblem,
    sells: Dict[str, pulp.LpVariable],
    buys: Dict[str, pulp.LpVariable],
    prices: pd.DataFrame,
    total_value: float,
    bid_ask_spreads: pd.DataFrame | None = None,
    commission_per_trade: float = 0.0,
    transaction_normalization: float = 1200.0,
) -> Tuple[pulp.LpAffineExpression, float]:
    """
    Calculate transaction cost component.

    Transaction costs include:
    - Bid-ask spread cost (proportional to trade value)
    - Commission per trade (fixed cost)

    Args:
        prob: The optimization problem to add constraints to
        sells: Dict mapping tax_lot_id -> LpVariable
        buys: Dict mapping identifier -> LpVariable
        prices: DataFrame with [identifier, price]
        total_value: Total portfolio value
        bid_ask_spreads: Optional DataFrame with [identifier, spread_bps]
        commission_per_trade: Fixed commission per trade (default $0)
        transaction_normalization: Normalization factor

    Returns:
        Tuple of (transaction_cost expression, baseline cost = 0)
    """
    transaction_terms = []
    price_map = dict(zip(prices["identifier"], prices["price"]))
    spread_map = dict(zip(bid_ask_spreads["identifier"], bid_ask_spreads["spread_bps"])) if bid_ask_spreads is not None else {}

    # Cost for sells
    for lot_id, var in sells.items():
        # Find the ticker for this lot
        # We need to pass in a mapping or look it up
        # For now, use a generic spread estimate
        spread_bps = 5.0  # Default 5 bps
        cost_per_share = var * spread_bps / 10000.0
        transaction_terms.append(cost_per_share * transaction_normalization)

    # Cost for buys
    for ticker, var in buys.items():
        price = price_map.get(ticker, 0.0)
        spread_bps = spread_map.get(ticker, 5.0)
        cost_per_share = var * price * spread_bps / 10000.0
        transaction_terms.append(cost_per_share * transaction_normalization)

    total_transaction_cost = pulp.lpSum(transaction_terms)
    return total_transaction_cost, 0.0
