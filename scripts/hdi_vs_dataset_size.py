"""Experiment: how uncertainty shrinks as the dataset grows.

Fits the model on datasets of increasing length and records the width of the 94%
credible interval for each channel's recovered adstock ``alpha``. To separate the
real trend from run-to-run noise, every dataset size is repeated over several
random seeds and the results are averaged. Saves a plot to
``docs/img/hdi_vs_n.png``.

Run with: ``uv run python scripts/hdi_vs_dataset_size.py``
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bmmm.config import Config  # noqa: E402
from bmmm.data.generate import generate  # noqa: E402
from bmmm.model import analysis  # noqa: E402
from bmmm.model.mmm import build_mmm, fit_mmm  # noqa: E402

warnings.filterwarnings("ignore")

# 1, 2, 3, 5 and 8 years of weekly data, each repeated over several seeds.
SIZES = [52, 104, 156, 260, 416]
SEEDS = [0, 1, 2, 3]


def run() -> pd.DataFrame:
    cfg = Config.from_yaml("configs/default.yaml")
    cfg.sampler.draws = 500
    cfg.sampler.tune = 500
    cfg.sampler.chains = 2

    rows: list[dict[str, object]] = []
    for n in SIZES:
        cfg.data.n_weeks = n
        for seed in SEEDS:
            cfg.data.seed = seed
            df, gt = generate(cfg.data)
            mmm = build_mmm(cfg)
            fit_mmm(mmm, df, cfg, progressbar=False)
            rec = analysis.recovery_table(mmm, gt)
            for _, r in rec.iterrows():
                rows.append(
                    {
                        "n_weeks": n,
                        "seed": seed,
                        "label": gt.labels[r["channel"]],
                        "width": float(r["hdi_high"] - r["hdi_low"]),
                    }
                )
        print(f"n_weeks={n}: done ({len(SEEDS)} seeds)")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: str = "docs/img/hdi_vs_n.png") -> None:
    agg = df.groupby(["label", "n_weeks"])["width"].agg(["mean", "min", "max"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, g in agg.groupby("label"):
        g = g.sort_values("n_weeks")
        (line,) = ax.plot(g["n_weeks"], g["mean"], marker="o", label=str(label))
        ax.fill_between(g["n_weeks"], g["min"], g["max"], color=line.get_color(), alpha=0.15)

    # Reference: posterior width usually shrinks like 1 / sqrt(n).
    ns = np.array(sorted(df["n_weeks"].unique()), dtype=float)
    n0 = ns.min()
    w0 = agg.loc[agg["n_weeks"] == n0, "mean"].mean()
    ax.plot(ns, w0 * np.sqrt(n0 / ns), "k--", alpha=0.6, label="~1/sqrt(n) reference")

    ax.set_xlabel("dataset size (weeks)")
    ax.set_ylabel("94% HDI width of adstock alpha")
    ax.set_title("Uncertainty shrinks as the dataset grows (mean over seeds)")
    ax.legend()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    data = run()
    plot(data)
    pivot = data.groupby(["n_weeks", "label"])["width"].mean().unstack().round(3)
    print(pivot.to_string())
