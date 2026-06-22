# BMMM Portfolio — Bayesian Marketing Mix Modeling

> An agency-style Bayesian Marketing Mix Model (MMM) built on
> [PyMC-Marketing](https://www.pymc-marketing.io/), wrapped in a production-ish
> ML-engineering shell: typed config, CLI, FastAPI service, Docker, CI/CD and a
> live interactive dashboard.

The headline DS story is **parameter recovery on synthetic data**: we define the
*true* adstock, saturation and ROI per channel, simulate sales, then show the
model recovers those parameters inside their credible intervals — evidence of
understanding the model, not just calling `.fit()`.

## Why this project

Marketing Mix Modeling answers: *given spend across TV / digital / social, how
much did each channel actually drive sales, and how should next quarter's budget
be allocated?* The Bayesian formulation gives full uncertainty (credible
intervals on ROAS), not just point estimates.

## Architecture

```
synthetic data (known ground truth)
        │
        ▼
   PyMC-Marketing MMM  ──fit──▶  idata.nc  (trained offline, fast sampler)
        │                           │
        │                    ┌──────┴───────┐
        ▼                    ▼              ▼
  parameter recovery    FastAPI service   Streamlit dashboard
   (notebook)           /predict          scenario planning
                        /optimize-budget   (reads idata, no sampling)
```

**Key design choice:** MCMC sampling is slow, so the model is trained *offline*
and the posterior (`idata`) is persisted. The service and dashboard load the
saved posterior — they never sample at request time. CI runs a tiny smoke-fit
(≈50 draws) only to prove training doesn't crash.

## Quickstart

```bash
uv sync                       # install
uv run bmmm generate-data     # synthetic dataset with ground truth
uv run bmmm train             # fit MMM, write artifacts/idata.nc
uv run bmmm evaluate          # diagnostics + parameter recovery
uv run bmmm optimize-budget   # budget allocation
uv run uvicorn bmmm.service.app:app --reload   # API
uv run streamlit run dashboard/app.py          # dashboard
```

## Live demo

_Streamlit dashboard on HuggingFace Spaces — link TBD._

## Stack

PyMC-Marketing · NumPyro · ArviZ · Pydantic · Typer · FastAPI · Streamlit ·
Docker · GitHub Actions · uv

## License

MIT
