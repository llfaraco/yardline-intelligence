"""Regression tests for rank metrics on vectorized moving-average baselines."""

import pandas as pd

from yardline_intelligence.evaluation.eval_vectorized import evaluate_baseline_vectorized


def test_baseline_evaluator_returns_spearman():
    rows = []
    for player, scale in [("WR1", 1.0), ("WR2", 2.0), ("WR3", 3.0)]:
        for week, points in enumerate([1.0, 2.0, 3.0], start=1):
            rows.append(
                {
                    "sleeper_id": player,
                    "week": week,
                    "half_ppr_points": points * scale,
                }
            )
        rows.append(
            {
                "sleeper_id": player,
                "week": 4,
                "half_ppr_points": 4.0 * scale,
            }
        )

    result = evaluate_baseline_vectorized(
        pd.DataFrame(rows),
        validate_weeks=[4],
        window=3,
    )

    assert result["n"] == 3
    assert result["spearman"] == 1.0
