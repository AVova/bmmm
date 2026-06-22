"""Post-fit analysis: diagnostics, parameter recovery and business outputs.

All functions take a *fitted* ``MMM`` (and, where relevant, the ground truth)
and return tidy ``pandas`` objects ready for tables, plots or the API.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM

from bmmm.data.generate import GroundTruth
from bmmm.model.mmm import TARGET, split_xy
from bmmm.model.mmm import idata_group as _group


def diagnostics(mmm: MMM) -> dict[str, float]:
    """Sampler health summary: max r-hat, min ESS and divergence count."""
    posterior = _group(mmm, "posterior")
    sample_stats = _group(mmm, "sample_stats")
    summary = az.summary(posterior)
    n_div = int(sample_stats["diverging"].sum()) if "diverging" in sample_stats else 0
    return {
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "num_divergences": n_div,
        "num_draws": int(posterior.sizes["draw"] * posterior.sizes["chain"]),
    }


def _hdi_bounds(samples: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    hdi = az.hdi(samples, hdi_prob=prob)
    return float(hdi[0]), float(hdi[1])


def recovery_table(mmm: MMM, ground_truth: GroundTruth, hdi_prob: float = 0.94) -> pd.DataFrame:
    """Compare true vs recovered adstock ``alpha`` per channel.

    Adstock retention is scale-free, so it is the cleanest recovery target: a
    ``within_hdi`` flag shows whether the true value sits inside the posterior
    credible interval.
    """
    post = _group(mmm, "posterior")
    rows = []
    for i, ch in enumerate(ground_truth.channel_names):
        samples = post["adstock_alpha"].isel(channel=i).values.flatten()
        lo, hi = _hdi_bounds(samples, hdi_prob)
        true = ground_truth.adstock_alpha[ch]
        rows.append(
            {
                "channel": ch,
                "parameter": "adstock_alpha",
                "true": true,
                "posterior_mean": float(samples.mean()),
                "hdi_low": lo,
                "hdi_high": hi,
                "within_hdi": bool(lo <= true <= hi),
            }
        )
    return pd.DataFrame(rows)


def channel_contributions(mmm: MMM, hdi_prob: float = 0.94) -> pd.DataFrame:
    """Total contribution per channel (original sales units) with HDI."""
    contrib = mmm.compute_channel_contribution_original_scale()
    total = contrib.sum(dim="date")  # (chain, draw, channel)
    rows = []
    for ch in total.coords["channel"].values:
        samples = total.sel(channel=ch).values.flatten()
        lo, hi = _hdi_bounds(samples, hdi_prob)
        rows.append(
            {
                "channel": str(ch),
                "contribution_mean": float(samples.mean()),
                "hdi_low": lo,
                "hdi_high": hi,
            }
        )
    df = pd.DataFrame(rows)
    df["contribution_share"] = df["contribution_mean"] / df["contribution_mean"].sum()
    return df


def roas_table(mmm: MMM, df: pd.DataFrame, hdi_prob: float = 0.94) -> pd.DataFrame:
    """Return-on-ad-spend per channel: contribution / total spend, with HDI."""
    contrib = mmm.compute_channel_contribution_original_scale()
    total = contrib.sum(dim="date")
    x, _ = split_xy(df)
    rows = []
    for ch in total.coords["channel"].values:
        spend = float(x[str(ch)].sum())
        samples = total.sel(channel=ch).values.flatten() / spend
        lo, hi = _hdi_bounds(samples, hdi_prob)
        rows.append(
            {
                "channel": str(ch),
                "roas_mean": float(samples.mean()),
                "hdi_low": lo,
                "hdi_high": hi,
            }
        )
    return pd.DataFrame(rows)


def fit_metrics(mmm: MMM, df: pd.DataFrame) -> dict[str, float]:
    """In-sample fit quality: R² and MAPE of the posterior-predictive mean."""
    y_true = df[TARGET].to_numpy()
    pp = _group(mmm, "posterior_predictive")[mmm.output_var]
    y_pred = pp.mean(dim=[d for d in pp.dims if d in ("chain", "draw", "sample")]).values
    y_pred = np.asarray(y_pred).flatten()[: len(y_true)]
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))))
    return {"r2": r2, "mape": mape}
