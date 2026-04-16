"""
Tax cost objective for portfolio optimization.
"""

import pulp
import pandas as pd
from typing import Dict, Tuple


def calculate_tax_impact(
    prob: pulp.LpProblem,
    sells: Dict[str, pulp.LpVariable],
    gain_loss: pd.DataFrame,
    total_value: float,
    tax_normalization: float = 800.0,
    enforce_wash_sale_prevention: bool = True,
) -> Tuple[pulp.LpAffineExpression, float]:
    """
    Calculate the tax impact component of the objective function.

    For each tax lot, we track:
    - Current unrealized gain/loss
    - Realized tax when selling (loss = tax savings, gain = tax owed)
    - Wash sale adjustment (reduce negative tax liability by 1/5th)

    Args:
        prob: The optimization problem to add constraints to
        sells: Dictionary mapping tax_lot_id -> LpVariable
        gain_loss: DataFrame with tax lot gain/loss info
            Required columns: [tax_lot_id, identifier, quantity, cost_basis,
                             current_value, per_share_tax_liability, gain_type]
        total_value: Total portfolio value for normalization
        tax_normalization: Normalization factor for tax impact
        enforce_wash_sale_prevention: If True, reduce tax savings by 1/5th
            to make TLH less attractive (conservative estimate)

    Returns:
        Tuple of:
        - Tax impact expression (sum of realized taxes)
        - Current unrealized tax score (for baseline comparison)
    """
    tax_impacts = []
    current_tax_score = 0.0

    for _, lot in gain_loss.iterrows():
        lot_id = lot["tax_lot_id"]
        if lot_id not in sells:
            continue

        per_share_tax = lot.get("per_share_tax_liability", 0.0)

        # For losses, reduce tax savings conservatively (1/5th) to account for
        # potential wash sale complications
        if per_share_tax < 0 and enforce_wash_sale_prevention:
            per_share_tax = per_share_tax / 5.0
        elif per_share_tax < 0:
            per_share_tax = 0.0

        if per_share_tax == 0:
            continue

        # Current tax liability for this lot
        qty = lot["quantity"]
        current_lot_tax = qty * per_share_tax
        current_tax_score += current_lot_tax / total_value

        # Realized tax variable
        tax_realized = pulp.LpVariable(f"tax_realized_{lot_id}")

        # Constraint: realized tax = reduction in tax liability from selling
        # If we sell x shares: new_tax = (qty - x) * per_share_tax
        # realized_tax = current_tax - new_tax = x * per_share_tax
        prob += (
            tax_realized == sells[lot_id] * per_share_tax / total_value,
            f"tax_realized_{lot_id}",
        )

        tax_impacts.append(tax_realized * tax_normalization)

    total_tax_impact = pulp.lpSum(tax_impacts)
    return total_tax_impact, current_tax_score
