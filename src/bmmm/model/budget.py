"""Budget optimisation over the recovered response curves.

PyMC-Marketing ships a budget optimiser, but it operates in the model's
internally-scaled space and behaves poorly through a save/load round-trip with
the classic ``MMM``. Instead we reconstruct each channel's steady-state response
curve *in original sales units* from the posterior and optimise allocation
ourselves. This is transparent, concave (hence a unique optimum) and tells a
clear story: reallocate spend until marginal ROI is equalised across channels.

Steady-state weekly response for a sustained weekly spend ``s`` (a constant
input is unchanged by normalised adstock):

    R_c(s) = max(sales) * beta_c * logistic_saturation(s / max(spend_c); lam_c)

where ``beta_c`` / ``lam_c`` are the posterior-median saturation parameters and
the ``max`` terms undo the model's max-abs scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM
from scipy.optimize import minimize

from bmmm.data.transforms import logistic_saturation
from bmmm.model.mmm import TARGET, split_xy


@dataclass
class ResponseCurve:
    """Per-channel steady-state response curve in original sales units."""

    channel: str
    lam: float
    beta: float
    spend_scale: float  # max observed spend (undoes channel max-abs scaling)
    target_scale: float  # max observed sales (undoes target max-abs scaling)

    def response(self, weekly_spend: float | np.ndarray) -> np.ndarray:
        """Predicted weekly contribution (sales units) at a given weekly spend."""
        x = np.asarray(weekly_spend, dtype=np.float64) / self.spend_scale
        return self.target_scale * self.beta * logistic_saturation(x, self.lam)


def response_curves(mmm: MMM, df: pd.DataFrame, quantile: float = 0.5) -> dict[str, ResponseCurve]:
    """Extract a response curve per channel from the fitted posterior."""
    x, _ = split_xy(df)
    target_scale = float(df[TARGET].max())
    recovered = mmm.format_recovered_transformation_parameters(quantile=quantile)
    curves: dict[str, ResponseCurve] = {}
    for ch in mmm.channel_columns:
        sat = recovered[ch]["saturation_params"]
        curves[ch] = ResponseCurve(
            channel=ch,
            lam=float(sat["lam"]),
            beta=float(sat["beta"]),
            spend_scale=float(x[ch].max()),
            target_scale=target_scale,
        )
    return curves


def current_allocation(df: pd.DataFrame, channels: list[str]) -> dict[str, float]:
    """Mean historical weekly spend per channel (the 'before' baseline)."""
    x, _ = split_xy(df)
    return {ch: float(x[ch].mean()) for ch in channels}


def total_response(allocation: dict[str, float], curves: dict[str, ResponseCurve]) -> float:
    """Sum of per-channel weekly responses for an allocation."""
    return float(sum(curves[ch].response(spend) for ch, spend in allocation.items()))


def optimize_budget(
    curves: dict[str, ResponseCurve],
    total_budget: float,
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Allocate ``total_budget`` across channels to maximise weekly response.

    Concave objective with a linear budget equality constraint, solved by SLSQP.
    Returns the optimal weekly spend per channel.
    """
    channels = list(curves.keys())
    n = len(channels)
    if bounds is None:
        bounds = dict.fromkeys(channels, (0.0, total_budget))
    bound_list = [bounds[ch] for ch in channels]
    x0 = np.full(n, total_budget / n)

    def neg_response(spend: np.ndarray) -> float:
        return -float(sum(curves[ch].response(spend[i]) for i, ch in enumerate(channels)))

    constraints = {"type": "eq", "fun": lambda s: float(np.sum(s) - total_budget)}
    result = minimize(
        neg_response,
        x0,
        method="SLSQP",
        bounds=bound_list,
        constraints=[constraints],
        options={"maxiter": 500, "ftol": 1e-8},
    )
    alloc = {ch: float(max(result.x[i], 0.0)) for i, ch in enumerate(channels)}
    # Renormalise tiny numerical drift back onto the budget.
    s = sum(alloc.values())
    if s > 0:
        alloc = {ch: v * total_budget / s for ch, v in alloc.items()}
    return alloc


def optimization_summary(
    mmm: MMM, df: pd.DataFrame, total_budget: float | None = None
) -> pd.DataFrame:
    """Before/after table: current vs optimised allocation and response."""
    channels = list(mmm.channel_columns)
    curves = response_curves(mmm, df)
    current = current_allocation(df, channels)
    if total_budget is None:
        total_budget = sum(current.values())
    optimal = optimize_budget(curves, total_budget)

    rows = []
    for ch in channels:
        rows.append(
            {
                "channel": ch,
                "current_spend": current[ch],
                "optimal_spend": optimal[ch],
                "current_response": float(curves[ch].response(current[ch])),
                "optimal_response": float(curves[ch].response(optimal[ch])),
            }
        )
    df_out = pd.DataFrame(rows)
    df_out["spend_change_pct"] = 100 * (df_out["optimal_spend"] / df_out["current_spend"] - 1)
    return df_out
