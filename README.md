# BMMM: Bayesian Marketing Mix Modeling

A Marketing Mix Model (MMM) built on
[PyMC-Marketing](https://www.pymc-marketing.io/), with the engineering pieces
around it: a typed config, a command-line tool, a small API, Docker, CI and an
interactive dashboard.

The main idea is **parameter recovery on synthetic data**. We make up a dataset
where we know the true effect of each channel, fit the model, and check that it
finds those true values back. Documentation: see the `docs/` site (MkDocs).

## What an MMM answers

Given spend on TV, social and search, how much did each channel add to sales, and
how should we split the next budget? The model gives a number for each question
and a range around it, so you can see how certain the answer is.

## Quickstart

```bash
uv sync                       # install
uv run bmmm generate-data     # synthetic dataset with known true values
uv run bmmm train             # fit the model, save artifacts/
uv run bmmm evaluate          # diagnostics and parameter recovery
uv run bmmm optimize-budget   # budget allocation
uv run uvicorn bmmm.service.app:app --reload   # API
uv run streamlit run dashboard/app.py          # dashboard
uv run mkdocs serve                            # docs site
```

## Design note

Fitting the model takes minutes, so we do it once and save the result. The API
and dashboard load that saved result and never refit. CI runs a tiny 50-draw fit
only to check that training still works.

## Stack

PyMC-Marketing, NumPy, ArviZ, Pydantic, Typer, FastAPI, Streamlit, Docker,
GitHub Actions, MkDocs, uv.

## License

MIT
