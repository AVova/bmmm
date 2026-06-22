"""Unit tests for the deterministic media transforms."""

from __future__ import annotations

import numpy as np
import pytest

from bmmm.data.transforms import geometric_adstock, logistic_saturation


def test_adstock_impulse_response_decays_geometrically() -> None:
    impulse = np.zeros(5)
    impulse[0] = 1.0
    out = geometric_adstock(impulse, alpha=0.5, l_max=5, normalize=False)
    expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(out, expected)


def test_adstock_normalized_preserves_sustained_level() -> None:
    x = np.ones(20)
    out = geometric_adstock(x, alpha=0.7, l_max=12, normalize=True)
    # A sustained unit input should converge to ~1 once the window fills.
    np.testing.assert_allclose(out[-1], 1.0, atol=1e-6)


def test_adstock_zero_alpha_is_identity() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.5])
    np.testing.assert_allclose(geometric_adstock(x, alpha=0.0, l_max=4, normalize=False), x)


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
def test_adstock_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(3), alpha=alpha)


def test_logistic_saturation_matches_closed_form() -> None:
    x = np.array([0.0, 1.0, 2.0, 5.0])
    lam = 0.5
    expected = (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
    np.testing.assert_allclose(logistic_saturation(x, lam), expected)


def test_logistic_saturation_is_monotonic_and_bounded() -> None:
    x = np.linspace(0, 100, 50)
    s = logistic_saturation(x, lam=1.0)
    assert np.all(np.diff(s) >= 0)  # monotonic increasing
    assert s.min() >= 0.0
    assert s.max() <= 1.0  # supremum is 1 (reached numerically for large lam*x)
    assert s[0] == 0.0
    # Strictly below 1 in the meaningful (unsaturated) range.
    assert logistic_saturation(np.array([3.0]), lam=1.0)[0] < 1.0


def test_logistic_saturation_rejects_nonpositive_lam() -> None:
    with pytest.raises(ValueError):
        logistic_saturation(np.ones(3), lam=0.0)
