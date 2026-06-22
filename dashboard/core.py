"""Light, self-contained helpers for the dashboard.

Deliberately depends only on numpy (plus the standard library), so the deployed
Streamlit app stays small and starts fast: no PyMC, no 92MB model. It reads the
compact ``dashboard.json`` produced by ``bmmm export-dashboard`` and reconstructs
each channel's response curve from a handful of numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ASSETS = Path(__file__).parent / "assets"


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Same diminishing-returns curve the model uses."""
    e = np.exp(-lam * np.asarray(x, dtype=float))
    return (1.0 - e) / (1.0 + e)


@dataclass
class Channel:
    """A channel's response curve, in original sales units."""

    name: str
    label: str
    lam: float
    beta: float
    spend_scale: float
    target_scale: float
    current_spend: float
    avg_roas: float
    marginal_roas: float
    contribution_share: float
    true_alpha: float
    recovered_alpha: float
    alpha_hdi_low: float
    alpha_hdi_high: float

    def response(self, spend: float | np.ndarray) -> np.ndarray:
        """Predicted weekly contribution to sales at a given weekly spend."""
        x = np.asarray(spend, dtype=float) / self.spend_scale
        return self.target_scale * self.beta * logistic_saturation(x, self.lam)


@dataclass
class DashboardData:
    metrics: dict[str, Any]
    budget: dict[str, float]
    channels: list[Channel]
    profit_curve: dict[str, Any]

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.channels]

    def by_label(self, label: str) -> Channel:
        return next(c for c in self.channels if c.label == label)


def load_data(path: Path | None = None) -> DashboardData:
    """Load the compact dashboard artifact."""
    path = path or (ASSETS / "dashboard.json")
    raw = json.loads(path.read_text())
    channels = [Channel(**c) for c in raw["channels"]]
    return DashboardData(
        metrics=raw["metrics"],
        budget=raw["budget"],
        channels=channels,
        profit_curve=raw["profit_curve"],
    )


def total_response(channels: list[Channel], allocation: dict[str, float]) -> float:
    """Sum of channel responses for a per-channel spend allocation."""
    return float(sum(c.response(allocation[c.name]) for c in channels))


def interp_profit(profit_curve: dict[str, Any], budget: float) -> dict[str, Any]:
    """Read the precomputed profit curve at an arbitrary budget (interpolated).

    Returns the optimally-allocated ad sales, profit, marginal ROAS and the
    per-channel optimal spend at that budget.
    """
    b = np.asarray(profit_curve["budget"], dtype=float)
    return {
        "ad_sales": float(np.interp(budget, b, profit_curve["ad_sales"])),
        "profit": float(np.interp(budget, b, profit_curve["profit"])),
        "marginal_roas": float(np.interp(budget, b, profit_curve["marginal_roas"])),
        "allocation": {
            ch: float(np.interp(budget, b, vals))
            for ch, vals in profit_curve["optimal_allocation"].items()
        },
    }
