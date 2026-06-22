"""Plotting helpers.

Every function returns a matplotlib ``Figure`` so callers can show, save or embed
it. Channel display names come from the per-channel ``label`` in the config
(falling back to the raw name).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from pymc_marketing.mmm import MMM

from bmmm.data.generate import GroundTruth
from bmmm.data.transforms import geometric_adstock, logistic_saturation
from bmmm.model import analysis, budget
from bmmm.model.budget import ResponseCurve
from bmmm.model.mmm import TARGET, idata_group

plt.rcParams.update({"figure.autolayout": True, "axes.grid": True, "grid.alpha": 0.3})

Labels = Mapping[str, str] | None


def _lbl(name: str, labels: Labels) -> str:
    """Display label for a channel name, defaulting to the name itself."""
    return labels.get(name, name) if labels else name


def plot_recovery(mmm: MMM, ground_truth: GroundTruth) -> Figure:
    """Signature plot: true adstock alpha vs recovered posterior (mean + HDI)."""
    table = analysis.recovery_table(mmm, ground_truth)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(table))
    ax.errorbar(
        table["posterior_mean"],
        y,
        xerr=[
            table["posterior_mean"] - table["hdi_low"],
            table["hdi_high"] - table["posterior_mean"],
        ],
        fmt="o",
        color="#1f77b4",
        capsize=4,
        label="posterior (mean + 94% HDI)",
    )
    ax.scatter(table["true"], y, color="#d62728", marker="D", zorder=5, label="true value")
    ax.set_yticks(y)
    ax.set_yticklabels([_lbl(c, ground_truth.labels) for c in table["channel"]])
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.margins(x=0.08)
    ax.set_xlabel("adstock alpha (retention)")
    ax.set_title("Parameter recovery: adstock retention")
    ax.legend(loc="best")
    return fig


def plot_posterior_predictive(mmm: MMM, df: pd.DataFrame) -> Figure:
    """Model fit vs actual sales with a 94% HDI band."""
    import arviz as az

    pp = idata_group(mmm, "posterior_predictive")[mmm.output_var]  # (chain, draw, date)
    stacked = pp.stack(sample=("chain", "draw"))
    mean = stacked.mean("sample").values
    hdi = az.hdi(pp, hdi_prob=0.94)[mmm.output_var].values  # (date, 2)

    dates = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(dates, hdi[:, 0], hdi[:, 1], color="#1f77b4", alpha=0.25, label="94% HDI")
    ax.plot(dates, mean, color="#1f77b4", lw=1.5, label="posterior mean")
    ax.plot(dates, df[TARGET].to_numpy(), color="black", lw=1, alpha=0.7, label="actual sales")
    ax.set_xlabel("date")
    ax.set_ylabel("sales")
    ax.set_title("Posterior predictive vs actual sales")
    ax.legend(loc="upper left")
    return fig


def _decomposition(
    mmm: MMM, df: pd.DataFrame, labels: Labels
) -> tuple[pd.Series, np.ndarray, list[np.ndarray], list[str]]:
    """Shared parts of the sales decomposition.

    Channel contributions come from the posterior (original scale); the baseline
    is the residual ``actual - sum(channels)`` so the stack reconstructs sales.
    """
    contrib = mmm.compute_channel_contribution_original_scale()
    mean = contrib.mean(dim=["chain", "draw"])  # (date, channel)
    channels = [str(c) for c in mean.coords["channel"].values]
    per_channel = [mean.sel(channel=ch).values for ch in channels]

    dates = pd.to_datetime(df["date"])
    sales = df[TARGET].to_numpy()
    baseline = sales - np.sum(per_channel, axis=0)
    stack = [baseline, *per_channel]
    legend = ["baseline", *[_lbl(ch, labels) for ch in channels]]
    return dates, sales, stack, legend


def plot_contribution_breakdown(mmm: MMM, df: pd.DataFrame, labels: Labels = None) -> Figure:
    """Stacked area: baseline plus each channel's contribution, in sales units."""
    dates, sales, stack, legend = _decomposition(mmm, df, labels)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(dates, *stack, labels=legend, alpha=0.85)
    ax.plot(dates, sales, color="black", lw=1, label="actual sales")
    ax.set_xlabel("date")
    ax.set_ylabel("sales")
    ax.set_title("Sales decomposition: absolute contribution")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    return fig


def plot_contribution_share(mmm: MMM, df: pd.DataFrame, labels: Labels = None) -> Figure:
    """Stacked area of relative contributions: each component as a share of sales."""
    dates, sales, stack, legend = _decomposition(mmm, df, labels)
    shares = [100.0 * comp / np.clip(sales, 1e-9, None) for comp in stack]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.stackplot(dates, *shares, labels=legend, alpha=0.85)
    ax.set_xlabel("date")
    ax.set_ylabel("% of sales")
    ax.set_ylim(0, 100)
    ax.set_title("Sales decomposition: share of sales")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    return fig


def plot_roas(mmm: MMM, df: pd.DataFrame, labels: Labels = None) -> Figure:
    """ROAS per channel as a forest-style point + HDI plot."""
    table = analysis.roas_table(mmm, df)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(table))
    ax.errorbar(
        table["roas_mean"],
        y,
        xerr=[table["roas_mean"] - table["hdi_low"], table["hdi_high"] - table["roas_mean"]],
        fmt="o",
        color="#2ca02c",
        capsize=4,
    )
    ax.axvline(1.0, color="grey", ls="--", lw=1, label="break-even (ROAS=1)")
    ax.set_yticks(y)
    ax.set_yticklabels([_lbl(c, labels) for c in table["channel"]])
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.margins(x=0.08)
    ax.set_xlabel("ROAS (sales per unit spend)")
    ax.set_title("Return on ad spend by channel (94% HDI)")
    ax.legend()
    return fig


def plot_marginal_roas(
    curves: dict[str, ResponseCurve], allocation: dict[str, float], labels: Labels = None
) -> Figure:
    """Marginal ROAS per channel at a given allocation, against break-even.

    Bars below 1 (red) are losing money on the next unit of spend; bars above 1
    (green) still pay off. This is the signal for reallocating budget.
    """
    channels = list(curves.keys())
    vals = [budget.marginal_roas(curves[ch], allocation[ch]) for ch in channels]
    colors = ["#d62728" if v < 1 else "#2ca02c" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(channels))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(1.0, color="grey", ls="--", lw=1, label="break-even (=1)")
    ax.set_yticks(y)
    ax.set_yticklabels([_lbl(c, labels) for c in channels])
    ax.set_xlabel("marginal ROAS (return on the next unit of spend)")
    ax.set_title("Marginal ROAS by channel at current spend")
    ax.legend(loc="lower right")
    return fig


def plot_budget_profit(curve: pd.DataFrame, current_budget: float, b_star: float) -> Figure:
    """Profit vs total budget (top) and marginal ROAS vs budget (bottom).

    Shows the profit-maximising budget ``b_star`` (where marginal ROAS hits 1)
    sitting below the current budget.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(curve["budget"], curve["ad_sales"], color="#1f77b4", label="ad-driven sales")
    ax1.plot(curve["budget"], curve["profit"], color="#2ca02c", label="profit (sales - budget)")
    ax1.axvline(
        current_budget, color="grey", ls=":", lw=1.5, label=f"current ({current_budget:.0f})"
    )
    ax1.axvline(b_star, color="#d62728", ls="--", lw=1.5, label=f"profit-max ({b_star:.0f})")
    ax1.set_ylabel("weekly sales / profit")
    ax1.set_title("Profit peaks below the current budget")
    ax1.legend(loc="center right")

    ax2.plot(curve["budget"], curve["marginal_roas"], color="#9467bd", label="marginal ROAS")
    ax2.axhline(1.0, color="grey", ls="--", lw=1, label="break-even (=1)")
    ax2.axvline(b_star, color="#d62728", ls="--", lw=1.5)
    ax2.axvline(current_budget, color="grey", ls=":", lw=1.5)
    ax2.set_xlabel("total weekly budget")
    ax2.set_ylabel("marginal ROAS")
    ax2.legend(loc="upper right")
    return fig


def plot_saturation_curves(ground_truth: GroundTruth, max_x: float = 3.0) -> Figure:
    """True saturation response curves per channel (diminishing returns)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, max_x, 100)
    for ch in ground_truth.channel_names:
        lam = ground_truth.saturation_lam[ch]
        ax.plot(
            x, logistic_saturation(x, lam), label=f"{_lbl(ch, ground_truth.labels)} (lam={lam:.1f})"
        )
    ax.set_xlabel("adstocked spend (scaled)")
    ax.set_ylabel("saturated response")
    ax.set_title("Saturation curves (diminishing returns)")
    ax.legend()
    return fig


def _component_band(da: xr.DataArray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean and 94% band of a (chain, draw, date) component."""
    arr = da.transpose("chain", "draw", "date").values.reshape(-1, n)
    mean = arr.mean(axis=0)
    lo, hi = np.percentile(arr, [3, 97], axis=0)
    return mean, lo, hi


def plot_trend_seasonality(mmm: MMM, df: pd.DataFrame) -> Figure:
    """Recovered trend and yearly seasonality contributions over time.

    Both are baseline components the model estimates: the trend comes from the
    ``time`` control, the seasonality from the Fourier terms. Each is shown with
    its 94% credible band.
    """
    post = idata_group(mmm, "posterior")
    dates = pd.to_datetime(df["date"])
    n = len(df)

    trend = post["control_contribution_original_scale"].sel(control="time")
    seas = post["yearly_seasonality_contribution_original_scale"]
    t_mean, t_lo, t_hi = _component_band(trend, n)
    s_mean, s_lo, s_hi = _component_band(seas, n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.fill_between(dates, t_lo, t_hi, color="#9467bd", alpha=0.25, label="94% HDI")
    ax1.plot(dates, t_mean, color="#9467bd", lw=1.5, label="posterior mean")
    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.set_ylabel("sales effect")
    ax1.set_title("Recovered trend (from the time control)")
    ax1.legend(loc="upper left")

    ax2.fill_between(dates, s_lo, s_hi, color="#8c564b", alpha=0.25, label="94% HDI")
    ax2.plot(dates, s_mean, color="#8c564b", lw=1.5, label="posterior mean")
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.set_ylabel("sales effect")
    ax2.set_xlabel("date")
    ax2.set_title("Recovered yearly seasonality (from the Fourier terms)")
    ax2.legend(loc="upper left")
    return fig


def plot_adstock_decay(ground_truth: GroundTruth, l_max: int = 12) -> Figure:
    """True adstock decay (carry-over impulse response) per channel.

    Uses the *un-normalised* weights ``[1, alpha, alpha**2, ...]`` so the curve
    is the intuitive textbook decay: 100% of the effect in the spend week, then
    a fraction ``alpha`` retained each subsequent week. (The model fits the
    normalised version, which preserves the level of a sustained input.)
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    impulse = np.zeros(l_max)
    impulse[0] = 1.0
    for ch in ground_truth.channel_names:
        alpha = ground_truth.adstock_alpha[ch]
        decay = geometric_adstock(impulse, alpha, l_max, normalize=False)
        ax.plot(
            range(l_max),
            decay,
            marker="o",
            label=f"{_lbl(ch, ground_truth.labels)} (alpha={alpha:.2f})",
        )
    ax.set_xlabel("weeks since spend")
    ax.set_ylabel("retained effect (week 0 = 1.0)")
    ax.set_title("Adstock decay (carry-over) by channel")
    ax.legend()
    return fig


def save_all(
    mmm: MMM, df: pd.DataFrame, ground_truth: GroundTruth, out_dir: str | Path
) -> list[Path]:
    """Render the full figure set to ``out_dir`` (used to populate the docs)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = ground_truth.labels

    curves = budget.response_curves(mmm, df)
    current = budget.current_allocation(df, ground_truth.channel_names)
    b_cur = sum(current.values())
    budgets = np.linspace(0.2 * b_cur, 1.8 * b_cur, 30)
    profit = budget.profit_curve(curves, budgets)
    b_star = float(profit["budget"].to_numpy()[int(profit["profit"].to_numpy().argmax())])

    figs = {
        "recovery": plot_recovery(mmm, ground_truth),
        "posterior_predictive": plot_posterior_predictive(mmm, df),
        "contributions": plot_contribution_breakdown(mmm, df, labels),
        "contributions_share": plot_contribution_share(mmm, df, labels),
        "trend_seasonality": plot_trend_seasonality(mmm, df),
        "roas": plot_roas(mmm, df, labels),
        "marginal_roas": plot_marginal_roas(curves, current, labels),
        "budget_profit": plot_budget_profit(profit, b_cur, b_star),
        "saturation": plot_saturation_curves(ground_truth),
        "adstock": plot_adstock_decay(ground_truth),
    }
    paths = []
    for name, fig in figs.items():
        p = out / f"{name}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
