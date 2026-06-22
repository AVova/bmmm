"""Unit tests for the synthetic data generator."""

from __future__ import annotations

import numpy as np

from bmmm.config import Config
from bmmm.data.generate import generate


def _cfg() -> Config:
    return Config.from_yaml("configs/ci.yaml")


def test_shape_and_columns() -> None:
    cfg = _cfg()
    df, gt = generate(cfg.data)
    assert len(df) == cfg.data.n_weeks
    for col in ["date", "price", "sales", *cfg.data.channel_names]:
        assert col in df.columns
    assert gt.channel_names == cfg.data.channel_names


def test_sales_non_negative() -> None:
    df, _ = generate(_cfg().data)
    assert (df["sales"] >= 0).all()


def test_reproducible_with_seed() -> None:
    cfg = _cfg()
    df1, _ = generate(cfg.data)
    df2, _ = generate(cfg.data)
    np.testing.assert_array_equal(df1["sales"].to_numpy(), df2["sales"].to_numpy())


def test_different_seed_changes_data() -> None:
    cfg = _cfg()
    df1, _ = generate(cfg.data)
    cfg.data.seed = 999
    df2, _ = generate(cfg.data)
    assert not np.array_equal(df1["sales"].to_numpy(), df2["sales"].to_numpy())


def test_ground_truth_populated() -> None:
    _, gt = generate(_cfg().data)
    for ch in gt.channel_names:
        assert gt.adstock_alpha[ch] > 0
        assert gt.contribution[ch] > 0
        assert np.isfinite(gt.roas[ch])
    records = gt.as_records()
    assert len(records) == len(gt.channel_names)
