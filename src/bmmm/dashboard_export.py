"""Export a compact, dependency-light artifact for the dashboard.

The dashboard runs on a small JSON (response-curve parameters, headline metrics,
a precomputed profit curve) plus the pre-rendered figures. It never loads the
92MB model or PyMC. This module does the heavy lifting offline; the dashboard
only reads its output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bmmm.artifacts import load_bundle
from bmmm.config import Config
from bmmm.data.generate import GroundTruth, generate
from bmmm.model import analysis, budget
from bmmm.viz import plots

DASHBOARD_JSON = "dashboard.json"
IMG_DIR = "img"


def build_dashboard_data(mmm: Any, df: pd.DataFrame, gt: GroundTruth) -> dict[str, Any]:
    """Assemble everything the dashboard needs into a plain dict."""
    channels = list(gt.channel_names)
    curves = budget.response_curves(mmm, df)
    current = budget.current_allocation(df, channels)

    diagnostics = analysis.diagnostics(mmm)
    metrics = analysis.fit_metrics(mmm, df)
    recovery = analysis.recovery_table(mmm, gt).set_index("channel")
    rec_mean = recovery["posterior_mean"].to_dict()
    rec_lo = recovery["hdi_low"].to_dict()
    rec_hi = recovery["hdi_high"].to_dict()
    shares = analysis.channel_contributions(mmm).set_index("channel")["contribution_share"].to_dict()
    roas = analysis.roas_table(mmm, df).set_index("channel")["roas_mean"].to_dict()

    # Profit curve plus the optimal split at each budget level, so the dashboard
    # can read allocations directly without re-running the optimiser.
    b_cur = float(sum(current.values()))
    budgets = np.linspace(0.0, 1.8 * b_cur, 40)
    ad_sales: list[float] = []
    alloc_by_channel: dict[str, list[float]] = {ch: [] for ch in channels}
    for b in budgets:
        alloc = budget.optimize_budget(curves, float(b))
        ad_sales.append(budget.total_response(alloc, curves))
        for ch in channels:
            alloc_by_channel[ch].append(round(alloc[ch], 1))
    ad_sales_arr = np.array(ad_sales)
    profit_arr = ad_sales_arr - budgets
    marginal_arr = np.gradient(ad_sales_arr, budgets)
    b_star = float(budgets[int(profit_arr.argmax())])

    channel_rows = []
    for ch in channels:
        c = curves[ch]
        channel_rows.append(
            {
                "name": ch,
                "label": gt.labels[ch],
                "lam": c.lam,
                "beta": c.beta,
                "spend_scale": c.spend_scale,
                "target_scale": c.target_scale,
                "current_spend": current[ch],
                "avg_roas": float(roas[ch]),
                "marginal_roas": budget.marginal_roas(c, current[ch]),
                "contribution_share": float(shares[ch]),
                "true_alpha": float(gt.adstock_alpha[ch]),
                "recovered_alpha": float(rec_mean[ch]),
                "alpha_hdi_low": float(rec_lo[ch]),
                "alpha_hdi_high": float(rec_hi[ch]),
            }
        )

    return {
        "metrics": {
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "max_r_hat": diagnostics["max_r_hat"],
            "num_divergences": diagnostics["num_divergences"],
            "n_weeks": len(df),
            "n_channels": len(channels),
        },
        "budget": {"current": b_cur, "profit_max": b_star},
        "channels": channel_rows,
        "profit_curve": {
            "budget": budgets.round(1).tolist(),
            "ad_sales": ad_sales_arr.round(1).tolist(),
            "profit": profit_arr.round(1).tolist(),
            "marginal_roas": marginal_arr.round(4).tolist(),
            "optimal_allocation": alloc_by_channel,
        },
    }


def export_dashboard(
    artifact_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path,
) -> Path:
    """Write ``dashboard.json`` and copy the figures into ``out_dir``."""
    out = Path(out_dir)
    (out / IMG_DIR).mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(artifact_dir)
    cfg = Config.from_yaml(config_path)
    _, gt = generate(cfg.data)

    data = build_dashboard_data(bundle.mmm, bundle.data, gt)
    (out / DASHBOARD_JSON).write_text(json.dumps(data, indent=2))

    # Render the figure set straight into the dashboard assets.
    plots.save_all(bundle.mmm, bundle.data, gt, out / IMG_DIR)
    return out / DASHBOARD_JSON
