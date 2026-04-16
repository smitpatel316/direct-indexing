"""
Drift objective for portfolio optimization.
Minimizes deviation from target weights.
"""

import pulp
import pandas as pd
from typing import Dict, Tuple


def calculate_drift_impact(
    prob: pulp.LpProblem,
    sells: Dict[str, pulp.LpVariable],
    buys: Dict[str, pulp.LpVariable],
    gain_loss: pd.DataFrame,
    target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    total_value: float,
    drift_normalization: float = 100.0,
) -> Tuple[pulp.LpAffineExpression, float]:
    """
    Calculate the drift impact — deviation from target weights.

    Drift is the difference between current weight and target weight,
    measured in portfolio value terms.

    Args:
        prob: The optimization problem to add constraints to
        sells: Dict mapping tax_lot_id -> LpVariable
        buys: Dict mapping identifier -> LpVariable
        gain_loss: DataFrame with current holdings
        target_weights: DataFrame with target weights per ticker
        prices: DataFrame with [identifier, price]
        total_value: Total portfolio value
        drift_normalization: Normalization factor for drift

    Returns:
        Tuple of (drift_impact expression, current_drift_score)
    """
    drift_terms = []
    current_drift = 0.0

    # Build current holdings by ticker
    ticker_holdings: Dict[str, dict] = {}
    for _, lot in gain_loss.iterrows():
        ticker = lot["identifier"]
        if ticker not in ticker_holdings:
            ticker_holdings[ticker] = {
                "quantity": 0.0,
                "value": 0.0,
                "target_weight": 0.0,
            }
        ticker_holdings[ticker]["quantity"] += lot["quantity"]
        ticker_holdings[ticker]["value"] += lot.get("current_value", 0.0)

    # Add target weights
    for _, row in target_weights.iterrows():
        ticker = row["identifier"]
        if ticker in ticker_holdings:
            ticker_holdings[ticker]["target_weight"] = row["target_weight"]

    price_map = dict(zip(prices["identifier"], prices["price"]))

    for ticker, holding in ticker_holdings.items():
        current_weight = holding["value"] / total_value if total_value > 0 else 0.0
        target_weight = holding.get("target_weight", 0.0)
        price = price_map.get(ticker, 0.0)

        # Calculate drift for sells
        ticker_lots = gain_loss[gain_loss["identifier"] == ticker]
        sell_contribution = pulp.lpSum(
            sells[lot.tax_lot_id] for lot in ticker_lots.itertuples()
            if lot.tax_lot_id in sells
        )

        # Calculate drift for buys
        buy_contribution = buys.get(ticker, 0.0)

        # Net position change in value terms
        position_change = (buy_contribution - sell_contribution) * price

        # New weight
        new_weight = (holding["value"] - sell_contribution * price + position_change) / total_value

        # Drift from target
        drift = new_weight - target_weight
        drift_terms.append(drift * drift_normalization)

        # Current drift for baseline
        current_drift += abs(current_weight - target_weight)

    # Penalize drift
    total_drift_impact = pulp.lpSum(drift_terms)
    return total_drift_impact, current_drift
