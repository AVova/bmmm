# BMMM — Bayesian Marketing Mix Modeling

A compact, **agency-style** Bayesian Marketing Mix Model (MMM) built on
[PyMC-Marketing](https://www.pymc-marketing.io/) and wrapped in a production-ish
ML-engineering shell: typed config, a CLI, a FastAPI service, Docker, CI/CD and
a live interactive dashboard.

!!! tip "The headline story"
    The project's signature is **parameter recovery on synthetic data**. We
    *define* the true adstock, saturation and ROI of each media channel, simulate
    sales, then show the model recovers those parameters inside their credible
    intervals — evidence of understanding the model, not just calling `.fit()`.

## What problem does an MMM solve?

> Given spend across TV / social / search, how much did each channel actually
> drive sales, and how should next quarter's budget be allocated?

The **Bayesian** formulation answers with full uncertainty — credible intervals
on ROAS and contributions — rather than fragile point estimates.

## Results at a glance

The model trained on 3 years of weekly synthetic data (`configs/default.yaml`,
4000 posterior draws, NUTS via `nutpie`):

| Metric | Value |
|---|---|
| In-sample R² | **0.92** |
| MAPE | **2.8 %** |
| Max R̂ | **1.01** |
| Divergences | **0** |
| Adstock α recovery | **3 / 3 channels inside the 94 % HDI** |

### Parameter recovery (the signature plot)

True adstock retention (red) vs the recovered posterior (blue, mean + 94 % HDI):

![Parameter recovery](img/recovery.png){ width="640" }

### Model fit

![Posterior predictive vs actual](img/posterior_predictive.png){ width="760" }

### Sales decomposition

How much of weekly sales each channel and the baseline explain:

![Sales decomposition](img/contributions.png){ width="760" }

### Return on ad spend

![ROAS by channel](img/roas.png){ width="640" }

## How it fits together

```
synthetic data (known ground truth)
        │
        ▼
   PyMC-Marketing MMM ──fit──▶ idata.nc (trained offline, fast sampler)
        │                          │
        │                   ┌──────┴───────┐
        ▼                   ▼              ▼
  parameter recovery   FastAPI service   Streamlit dashboard
   (this site)         /predict          scenario planning
                       /optimize-budget  (reads idata, no sampling)
```

**Key design choice:** MCMC sampling is slow, so the model is trained *offline*
and the posterior is persisted. The service and dashboard load the saved
posterior — they never sample at request time.

## Where to go next

- [What is BMMM →](concepts.md) — adstock, saturation and the model structure
- [Parameter recovery →](parameter-recovery.md) — the validation methodology
- [Budget optimization →](budget-optimization.md) — reallocating spend
- [CLI](usage-cli.md) and [API](usage-api.md) — how to run it
- [API Reference](api/config.md) — auto-generated from docstrings
