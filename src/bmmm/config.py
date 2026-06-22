"""Typed configuration for the whole project.

A single YAML file (see ``configs/default.yaml``) drives data generation, the
model priors and the sampler. Everything is validated by Pydantic so a bad
config fails fast with a readable error instead of deep inside PyMC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ChannelSpec(BaseModel):
    """Ground-truth specification of a single media channel.

    The ``adstock_alpha`` / ``saturation_lam`` / ``beta`` values are the *true*
    parameters used to simulate sales; after fitting we check the model
    recovers them inside the posterior credible intervals.
    """

    name: str
    adstock_alpha: float = Field(ge=0.0, lt=1.0, description="Geometric adstock retention")
    saturation_lam: float = Field(gt=0.0, description="Logistic saturation steepness")
    beta: float = Field(gt=0.0, description="Contribution amplitude in sales units")
    spend_mean: float = Field(gt=0.0, description="Mean weekly spend level")
    spend_sigma: float = Field(ge=0.0, description="Std-dev of weekly spend (lognormal)")


class DataConfig(BaseModel):
    """Synthetic data-generating process."""

    n_weeks: int = Field(default=156, gt=20, description="Number of weekly observations")
    seed: int = 42
    start_date: str = "2023-01-01"

    baseline: float = Field(default=2000.0, description="Intercept (organic sales)")
    trend_slope: float = Field(default=2.0, description="Linear trend per week")
    seasonality_amplitude: float = Field(default=400.0, ge=0.0)
    n_fourier: int = Field(default=2, ge=1, le=6, description="Yearly seasonality harmonics")

    price_beta: float = Field(default=-300.0, description="Effect of (centered) price control")
    noise_sigma: float = Field(default=150.0, ge=0.0, description="Observation noise std-dev")

    channels: list[ChannelSpec]

    @model_validator(mode="after")
    def _unique_channel_names(self) -> DataConfig:
        names = [c.name for c in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("channel names must be unique")
        return self

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]


class ModelConfig(BaseModel):
    """Model structure and priors handed to PyMC-Marketing."""

    adstock_l_max: int = Field(default=12, gt=0)
    yearly_seasonality: int = Field(default=2, ge=0, description="Fourier modes; 0 disables")
    control_columns: list[str] = Field(default_factory=lambda: ["price"])

    # Prior scales (kept simple & explicit so they show up in the notebook).
    intercept_sigma: float = 2.0
    saturation_beta_sigma: float = 2.0
    adstock_alpha_a: float = Field(default=1.0, description="Beta prior a for adstock alpha")
    adstock_alpha_b: float = Field(default=3.0, description="Beta prior b for adstock alpha")


class SamplerConfig(BaseModel):
    """NUTS sampler settings. CI overrides these with tiny values."""

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = Field(default=0.9, gt=0.0, lt=1.0)
    nuts_sampler: str = Field(default="nutpie", description="nutpie | pymc (numpyro needs AVX/JAX)")
    random_seed: int = 42


class Config(BaseModel):
    """Top-level config aggregating all sections."""

    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
