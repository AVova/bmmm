"""Thin wrapper around PyMC-Marketing's ``MMM``.

Keeps the rest of the codebase decoupled from PyMC-Marketing's exact API and
centralises the choices that matter for the project: transforms, priors,
control/seasonality structure and the sampler. Training is expected to happen
*offline*; the service and dashboard load the persisted artifact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr
from arviz import InferenceData
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

from bmmm.config import Config

TARGET = "sales"


def idata_group(mmm: MMM, name: str) -> xr.Dataset:
    """Fetch an idata group (``posterior``, ``sample_stats``, ...).

    Raises a clear error if the model carries no inference data yet, and gives
    static type-checkers a concrete ``Dataset`` to reason about.
    """
    idata = mmm.idata
    if idata is None:
        raise ValueError("Model has no inference data; fit or load it first.")
    return idata[name]


def build_mmm(config: Config) -> MMM:
    """Construct an (unfitted) ``MMM`` from config.

    Uses geometric adstock + logistic saturation with explicit, weakly-informed
    priors so the modelling choices are visible rather than buried in defaults.
    """
    m = config.model
    adstock = GeometricAdstock(l_max=m.adstock_l_max)
    saturation = LogisticSaturation()

    model_config = {
        "intercept": Prior("Normal", mu=0, sigma=m.intercept_sigma),
        "saturation_beta": Prior("HalfNormal", sigma=m.saturation_beta_sigma),
        "adstock_alpha": Prior("Beta", alpha=m.adstock_alpha_a, beta=m.adstock_alpha_b),
        "saturation_lam": Prior("Gamma", mu=2.0, sigma=1.0),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2.0)),
    }

    return MMM(
        date_column="date",
        channel_columns=config.data.channel_names,
        adstock=adstock,
        saturation=saturation,
        control_columns=m.control_columns or None,
        yearly_seasonality=m.yearly_seasonality or None,
        model_config=model_config,
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the observable DataFrame into features ``X`` and target ``y``."""
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    return x, y


def fit_mmm(
    mmm: MMM,
    df: pd.DataFrame,
    config: Config,
    *,
    progressbar: bool = False,
) -> InferenceData:
    """Fit the model and attach the posterior predictive in-place.

    Sampler settings come from ``config.sampler`` so CI can dial them down to a
    smoke-test (e.g. 50 draws / 1 chain) without touching code.
    """
    s = config.sampler
    x, y = split_xy(df)
    idata = mmm.fit(
        x,
        y,
        draws=s.draws,
        tune=s.tune,
        chains=s.chains,
        target_accept=s.target_accept,
        nuts_sampler=s.nuts_sampler,
        random_seed=s.random_seed,
        progressbar=progressbar,
    )
    mmm.sample_posterior_predictive(x, extend_idata=True, combined=True)
    return idata


def save_mmm(mmm: MMM, path: str | Path) -> None:
    """Persist the fitted model (idata + metadata) as netCDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mmm.save(str(path))


def load_mmm(path: str | Path) -> MMM:
    """Load a previously saved model. No sampling happens here."""
    return MMM.load(str(path))
