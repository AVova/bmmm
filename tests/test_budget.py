"""Unit tests for the budget optimiser (deterministic, no model needed)."""

from __future__ import annotations

import numpy as np

from bmmm.model.budget import ResponseCurve, optimize_budget, total_response


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
