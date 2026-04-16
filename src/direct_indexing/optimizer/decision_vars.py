"""
Decision variable creation for portfolio optimization.
"""

import pulp
import pandas as pd
from typing import Dict


def create_decision_variables(
    prob: pulp.LpProblem,
    tax_lots: pd.DataFrame,
    target_tickers: list[str],
    prices: pd.DataFrame,
    cash: float,
) -> tuple[Dict[str, pulp.LpVariable], Dict[str, pulp.LpVariable], pulp.LpVariable]:
    """
    Create buy/sell decision variables for the optimization problem.

    Args:
        prob: The PuLP problem to add variables to
        tax_lots: DataFrame with columns [tax_lot_id, identifier, quantity]
        target_tickers: List of valid ticker symbols to trade
        prices: DataFrame with columns [identifier, price]
        cash: Available cash for buying

    Returns:
        Tuple of (sells, buys, cash_var):
            - sells: Dict mapping tax_lot_id -> LpVariable (quantity to sell)
            - buys: Dict mapping identifier -> LpVariable (quantity to buy)
            - cash_var: LpVariable for tracking cash usage
    """
    sells: Dict[str, pulp.LpVariable] = {}
    buys: Dict[str, pulp.LpVariable] = {}

    # Create sell variables for each tax lot
    for _, lot in tax_lots.iterrows():
        lot_id = lot["tax_lot_id"]
        ticker = lot["identifier"]
        max_qty = lot["quantity"]

        sells[lot_id] = pulp.LpVariable(
            f"sell_{lot_id}",
            lowBound=0,
            upBound=max_qty,
            cat="Continuous",
        )

    # Create buy variables for each target ticker
    for ticker in target_tickers:
        price_row = prices[prices["identifier"] == ticker]
        if price_row.empty:
            continue
        price = price_row["price"].iloc[0]
        if price <= 0:
            continue

        # Max quantity based on available cash
        max_qty = cash / price

        buys[ticker] = pulp.LpVariable(
            f"buy_{ticker}",
            lowBound=0,
            upBound=max_qty,
            cat="Continuous",
        )

    # Cash usage variable
    cash_var = pulp.LpVariable("cash_used", lowBound=0, upBound=cash)

    return sells, buys, cash_var
