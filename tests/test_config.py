"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bmmm.config import ChannelSpec, Config, DataConfig


def test_load_default_yaml() -> None:
    cfg = Config.from_yaml("configs/default.yaml")
    assert cfg.data.channel_names == ["tv", "social", "search"]
    assert cfg.sampler.nuts_sampler == "nutpie"
    assert cfg.model.control_columns == ["price", "time"]


def test_duplicate_channel_names_rejected() -> None:
    chan = ChannelSpec(
        name="tv", adstock_alpha=0.5, saturation_lam=1.0, beta=100.0, spend_mean=10.0, spend_sigma=0.1
    )
    dup = chan.model_copy()
    with pytest.raises(ValidationError):
        DataConfig(channels=[chan, dup])


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 2.0])
def test_adstock_alpha_bounds(alpha: float) -> None:
    with pytest.raises(ValidationError):
        ChannelSpec(
            name="x", adstock_alpha=alpha, saturation_lam=1.0, beta=1.0, spend_mean=1.0, spend_sigma=0.1
        )


def test_channel_names_property() -> None:
    cfg = Config.from_yaml("configs/ci.yaml")
    assert cfg.data.channel_names == ["tv", "social"]
