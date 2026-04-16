"""
MILP Solver wrapper using PuLP + CBC.
"""

import pulp


def solve_optimization_problem(
    prob: pulp.LpProblem,
    time_limit: int = 60,
    gap_rel: float = 0.01,
    warm_start: bool = True,
) -> tuple[int, float | None]:
    """
    Solve a PuLP optimization problem using the CBC solver.

    Args:
        prob: The PuLP optimization problem to solve
        time_limit: Time limit in seconds (default 60)
        gap_rel: Relative optimality gap tolerance (default 0.01 = 1%)
        warm_start: Whether to use warm start (default True)

    Returns:
        Tuple of (status, objective_value):
            - status: Solver status code (pulp.LpStatus)
            - objective_value: The optimal objective value, or None if not solved
    """
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        warmStart=warm_start,
        options=[
            "allowableGap", str(gap_rel),
            "maxSolutions", "1",
            "maxNodes", "10000",
        ],
    )

    status = prob.solve(solver)
    objective_value = pulp.value(prob.objective) if status == pulp.LpStatusOptimal else None

    return status, objective_value
