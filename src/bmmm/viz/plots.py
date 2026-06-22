"""Plotting helpers.

A mix of thin wrappers over PyMC-Marketing's built-in plots and a couple of
custom figures (notably the parameter-recovery chart, which is the project's
signature visual). Every function returns a matplotlib ``Figure`` so callers
can show, save or embed it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pymc_marketing.mmm import MMM

from bmmm.data.generate import GroundTruth
from bmmm.data.transforms import geometric_adstock, logistic_saturation
from bmmm.model import analysis
from bmmm.model.mmm import TARGET, idata_group

plt.rcParams.update({"figure.autolayout": True, "axes.grid": True, "grid.alpha": 0.3})


def plot_recovery(mmm: MMM, ground_truth: GroundTruth) -> Figure:
    """Signature plot: true adstock alpha vs recovered posterior (mean + HDI)."""
    table = analysis.recovery_table(mmm, ground_truth)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(table))
    ax.errorbar(
        table["posterior_mean"],
        y,
        xerr=[table["posterior_mean"] - table["hdi_low"], table["hdi_high"] - table["posterior_mean"]],
        fmt="o",
        color="#1f77b4",
        capsize=4,
        label="posterior (mean + 94% HDI)",
    )
    ax.scatter(table["true"], y, color="#d62728", marker="D", zorder=5, label="true value")
    ax.set_yticks(y)
    ax.set_yticklabels(table["channel"])
    ax.set_xlabel("adstock alpha (retention)")
    ax.set_title("Parameter recovery: adstock retention")
    ax.legend(loc="best")
    return fig


def plot_posterior_predictive(mmm: MMM, df: pd.DataFrame) -> Figure:
    """Model fit vs actual sales with a 94% HDI band.

    Built directly from ``idata`` (in original sales units) rather than the
    library helper, which mis-scales the observed series after a save/load
    round-trip.
    """
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


def plot_contribution_breakdown(mmm: MMM, df: pd.DataFrame) -> Figure:
    """Stacked area: baseline + each channel's contribution over time.

    Channel contributions come from the posterior (original scale); the baseline
    is the residual ``actual - sum(channels)`` so the stack reconstructs sales.
    """
    contrib = mmm.compute_channel_contribution_original_scale()
    mean = contrib.mean(dim=["chain", "draw"])  # (date, channel)
    channels = [str(c) for c in mean.coords["channel"].values]
    per_channel = {ch: mean.sel(channel=ch).values for ch in channels}

    dates = pd.to_datetime(df["date"])
    sales = df[TARGET].to_numpy()
    baseline = sales - np.sum(list(per_channel.values()), axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(
        dates,
        baseline,
        *per_channel.values(),
        labels=["baseline", *channels],
        alpha=0.85,
    )
    ax.plot(dates, sales, color="black", lw=1, label="actual sales")
    ax.set_xlabel("date")
    ax.set_ylabel("sales")
    ax.set_title("Sales decomposition: baseline + channel contributions")
    ax.legend(loc="upper left", ncol=2)
    return fig


def plot_roas(mmm: MMM, df: pd.DataFrame) -> Figure:
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
    ax.set_yticklabels(table["channel"])
    ax.set_xlabel("ROAS (sales per unit spend)")
    ax.set_title("Return on ad spend by channel (94% HDI)")
    ax.legend()
    return fig


def plot_saturation_curves(ground_truth: GroundTruth, max_x: float = 3.0) -> Figure:
    """True saturation response curves per channel (diminishing returns)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, max_x, 100)
    for ch in ground_truth.channel_names:
        lam = ground_truth.saturation_lam[ch]
        ax.plot(x, logistic_saturation(x, lam), label=f"{ch} (lam={lam:.1f})")
    ax.set_xlabel("adstocked spend (scaled)")
    ax.set_ylabel("saturated response")
    ax.set_title("Saturation curves (diminishing returns)")
    ax.legend()
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
        ax.plot(range(l_max), decay, marker="o", label=f"{ch} (alpha={alpha:.2f})")
    ax.set_xlabel("weeks since spend")
    ax.set_ylabel("retained effect (week 0 = 1.0)")
    ax.set_title("Adstock decay (carry-over) by channel")
    ax.legend()
    return fig


def save_all(mmm: MMM, df: pd.DataFrame, ground_truth: GroundTruth, out_dir: str | Path) -> list[Path]:
    """Render the full figure set to ``out_dir`` (used to populate the README)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "recovery": plot_recovery(mmm, ground_truth),
        "posterior_predictive": plot_posterior_predictive(mmm, df),
        "contributions": plot_contribution_breakdown(mmm, df),
        "roas": plot_roas(mmm, df),
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
