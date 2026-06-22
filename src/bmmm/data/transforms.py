"""NumPy reference implementations of the media transforms.

These mirror the functional forms used inside PyMC-Marketing's ``MMM``
(geometric adstock + logistic saturation) so that data generated with these
functions has a *known ground truth* the fitted model can be checked against.

Both functions are deterministic and dependency-light, which makes them ideal
unit-test targets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def geometric_adstock(
    x: FloatArray,
    alpha: float,
    l_max: int = 12,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Geometric adstock (carry-over) transform.

    ``out_t = sum_{l=0}^{l_max-1} alpha**l * x_{t-l}`` optionally divided by
    ``sum_{l} alpha**l`` so that a sustained input is preserved in level.

    Parameters
    ----------
    x : array
        1-D spend series.
    alpha : float
        Retention rate in ``[0, 1)``. Larger -> longer carry-over.
    l_max : int
        Number of lags to accumulate.
    normalize : bool
        Divide by the sum of weights (matches PyMC-Marketing default).
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    if l_max <= 0:
        raise ValueError(f"l_max must be > 0, got {l_max}")

    x = np.asarray(x, dtype=np.float64)
    weights = alpha ** np.arange(l_max, dtype=np.float64)
    if normalize:
        weights = weights / weights.sum()

    out = np.zeros_like(x)
    for lag, w in enumerate(weights):
        if lag == 0:
            out += w * x
        else:
            out[lag:] += w * x[:-lag]
    return out


def logistic_saturation(x: FloatArray, lam: float) -> FloatArray:
    """Logistic saturation transform.

    ``f(x) = (1 - exp(-lam * x)) / (1 + exp(-lam * x))`` — a diminishing-returns
    curve mapping non-negative spend into ``[0, 1)``. ``lam`` controls how
    quickly the channel saturates.
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(-lam * x)
    return (1.0 - e) / (1.0 + e)
