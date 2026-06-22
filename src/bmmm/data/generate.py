"""Synthetic Marketing Mix dataset with a known ground truth.

The data-generating process matches the model's assumptions:

    sales_t = baseline + trend_t + seasonality_t + price_effect_t
              + sum_channel  beta_c * sat(adstock(spend_c)) ; lam_c, alpha_c
              + noise_t

Because we *choose* every parameter, we can later check the fitted MMM recovers
them. ``generate()`` returns both the observable DataFrame and a ``GroundTruth``
record carrying the true per-channel parameters and contributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bmmm.config import DataConfig
from bmmm.data.transforms import geometric_adstock, logistic_saturation


@dataclass
class GroundTruth:
    """True parameters and derived quantities behind the synthetic data."""

    channel_names: list[str]
    adstock_alpha: dict[str, float]
    saturation_lam: dict[str, float]
    beta: dict[str, float]
    # Total simulated contribution (sales units) per channel over the period.
    contribution: dict[str, float] = field(default_factory=dict)
    # ROI = contribution / total spend per channel.
    roi: dict[str, float] = field(default_factory=dict)

    def as_records(self) -> list[dict[str, float | str]]:
        """Flatten to per-channel rows for easy tabular comparison."""
        return [
            {
                "channel": name,
                "adstock_alpha": self.adstock_alpha[name],
                "saturation_lam": self.saturation_lam[name],
                "beta": self.beta[name],
                "contribution": self.contribution.get(name, float("nan")),
                "roi": self.roi.get(name, float("nan")),
            }
            for name in self.channel_names
        ]


def _fourier_seasonality(
    n: int, period: float, n_modes: int, amplitude: float, rng: np.random.Generator
) -> np.ndarray:
    """Sum of sine/cosine harmonics with random phases, scaled to ~amplitude."""
    t = np.arange(n)
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_modes + 1):
        a, b = rng.normal(size=2)
        out += a * np.sin(2 * np.pi * k * t / period) + b * np.cos(2 * np.pi * k * t / period)
    # Normalize to the requested amplitude.
    if out.std() > 0:
        out = out / out.std() * amplitude
    return out


def generate(config: DataConfig) -> tuple[pd.DataFrame, GroundTruth]:
    """Simulate a weekly MMM dataset.

    Returns
    -------
    (df, ground_truth)
        ``df`` has columns ``date``, one per channel spend, ``price`` and
        ``sales``. ``ground_truth`` carries the parameters used to build it.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_weeks

    dates = pd.date_range(config.start_date, periods=n, freq="W-MON")
    t = np.arange(n, dtype=np.float64)

    baseline = config.baseline + config.trend_slope * t
    seasonality = _fourier_seasonality(
        n, period=52.0, n_modes=config.n_fourier, amplitude=config.seasonality_amplitude, rng=rng
    )

    # Price control: a centered index with some persistence.
    price = 100.0 + np.cumsum(rng.normal(0, 1.5, size=n))
    price_centered = price - price.mean()
    price_effect = config.price_beta * (price_centered / price_centered.std())

    data: dict[str, np.ndarray] = {}
    contributions: dict[str, float] = {}
    rois: dict[str, float] = {}
    adstock_alpha, saturation_lam, beta = {}, {}, {}

    sales = baseline + seasonality + price_effect

    for ch in config.channels:
        # Spend: positive, bursty, with mild seasonality (flighting).
        flight = 1.0 + 0.3 * np.sin(2 * np.pi * t / 52.0 + rng.uniform(0, 2 * np.pi))
        spend = rng.lognormal(
            mean=np.log(max(config.baseline * 0 + ch.spend_mean, 1e-9)),
            sigma=ch.spend_sigma,
            size=n,
        )
        spend = np.clip(spend * flight, 0, None)

        adstocked = geometric_adstock(spend, ch.adstock_alpha, l_max=12, normalize=True)
        # Saturation expects a roughly unit-scaled input; scale by channel mean
        # so ``lam`` is interpretable and recoverable.
        scale = spend.mean() if spend.mean() > 0 else 1.0
        saturated = logistic_saturation(adstocked / scale, ch.saturation_lam)
        contribution = ch.beta * saturated

        data[ch.name] = spend
        sales = sales + contribution

        contributions[ch.name] = float(contribution.sum())
        rois[ch.name] = float(contribution.sum() / spend.sum()) if spend.sum() > 0 else float("nan")
        adstock_alpha[ch.name] = ch.adstock_alpha
        saturation_lam[ch.name] = ch.saturation_lam
        beta[ch.name] = ch.beta

    noise = rng.normal(0, config.noise_sigma, size=n)
    sales = np.clip(sales + noise, 0, None)

    df = pd.DataFrame({"date": dates, **data, "price": price, "sales": sales})

    gt = GroundTruth(
        channel_names=config.channel_names,
        adstock_alpha=adstock_alpha,
        saturation_lam=saturation_lam,
        beta=beta,
        contribution=contributions,
        roi=rois,
    )
    return df, gt
