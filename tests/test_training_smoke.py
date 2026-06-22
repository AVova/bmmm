"""Smoke test: the full train -> analyse -> save/load pipeline runs end-to-end.

Marked ``slow`` because it samples (tiny draws). Run with ``pytest -m slow`` or
let CI run it as a dedicated step; the default suite excludes it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from bmmm.config import Config
from bmmm.data.generate import generate
from bmmm.model import analysis
from bmmm.model.mmm import build_mmm, fit_mmm, load_mmm, save_mmm

pytestmark = pytest.mark.slow


def test_train_analyse_roundtrip(tmp_path: Path) -> None:
    warnings.simplefilter("ignore")
    cfg = Config.from_yaml("configs/ci.yaml")
    df, gt = generate(cfg.data)

    mmm = build_mmm(cfg)
    fit_mmm(mmm, df, cfg, progressbar=False)

    diag = analysis.diagnostics(mmm)
    assert diag["num_draws"] == cfg.sampler.draws * cfg.sampler.chains
    assert "max_r_hat" in diag and "num_divergences" in diag

    recovery = analysis.recovery_table(mmm, gt)
    assert set(recovery["channel"]) == set(gt.channel_names)
    assert {"true", "posterior_mean", "within_hdi"}.issubset(recovery.columns)

    contrib = analysis.channel_contributions(mmm)
    assert abs(contrib["contribution_share"].sum() - 1.0) < 1e-6

    metrics = analysis.fit_metrics(mmm, df)
    assert "r2" in metrics and "mape" in metrics

    # Persist and reload without re-sampling.
    path = tmp_path / "mmm.nc"
    save_mmm(mmm, path)
    reloaded = load_mmm(path)
    assert analysis.diagnostics(reloaded)["num_draws"] == diag["num_draws"]
