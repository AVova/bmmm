"""Unit tests for the budget optimiser (deterministic, no model needed)."""

from __future__ import annotations

import numpy as np

from bmmm.model.budget import (
    ResponseCurve,
    marginal_roas,
    optimize_budget,
    profit_curve,
    profit_maximizing_budget,
    total_response,
)


def _curves() -> dict[str, ResponseCurve]:
    # Same scales; 'good' has higher beta -> should attract more budget.
    return {
        "good": ResponseCurve("good", lam=2.0, beta=1.0, spend_scale=100.0, target_scale=1000.0),
        "weak": ResponseCurve("weak", lam=2.0, beta=0.3, spend_scale=100.0, target_scale=1000.0),
    }


def test_allocation_respects_budget() -> None:
    alloc = optimize_budget(_curves(), total_budget=200.0)
    assert abs(sum(alloc.values()) - 200.0) < 1e-6
    assert all(v >= 0 for v in alloc.values())


def test_prefers_higher_response_channel() -> None:
    alloc = optimize_budget(_curves(), total_budget=200.0)
    assert alloc["good"] > alloc["weak"]


def test_optimum_beats_equal_split() -> None:
    curves = _curves()
    budget = 200.0
    opt = optimize_budget(curves, budget)
    equal = dict.fromkeys(curves, budget / 2)
    assert total_response(opt, curves) >= total_response(equal, curves) - 1e-6


def test_response_curve_monotonic_and_saturating() -> None:
    c = ResponseCurve("c", lam=1.5, beta=1.0, spend_scale=100.0, target_scale=1000.0)
    spends = np.array([0.0, 50.0, 100.0, 500.0])
    resp = c.response(spends)
    assert resp[0] == 0.0
    assert np.all(np.diff(resp) > 0)  # increasing
    # Diminishing returns: marginal gain shrinks.
    assert (resp[2] - resp[1]) < (resp[1] - resp[0])


def test_marginal_roas_decreases_with_spend() -> None:
    c = ResponseCurve("c", lam=2.0, beta=1.0, spend_scale=100.0, target_scale=1000.0)
    assert marginal_roas(c, 10.0) > marginal_roas(c, 200.0)


def test_profit_curve_columns_and_concavity() -> None:
    curves = _curves()
    budgets = np.linspace(10, 600, 25)
    curve = profit_curve(curves, budgets)
    assert {"budget", "ad_sales", "profit", "marginal_roas"}.issubset(curve.columns)
    # Marginal ROAS should fall as the budget grows (diminishing returns).
    assert curve["marginal_roas"].iloc[0] > curve["marginal_roas"].iloc[-1]
    # Profit peaks at an interior budget, not the largest one.
    best = profit_maximizing_budget(curves, budgets)
    assert budgets[0] < best < budgets[-1]
