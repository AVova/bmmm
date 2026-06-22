# BMMM: Bayesian Marketing Mix Modeling

[![CI](https://github.com/AVova/bmmm/actions/workflows/ci.yml/badge.svg)](https://github.com/AVova/bmmm/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://avova.github.io/bmmm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<div align="center">

## 🔗 &nbsp; Live demos &nbsp; 🔗

✨ 🟢 ✨ 🔵 ✨ 🟢 ✨ 🔵 ✨ 🟢 ✨ 🔵 ✨ 🟢 ✨ 🔵 ✨ 🟢 ✨

### 🎛️ &nbsp; [**Live dashboard**](https://avova-ds.streamlit.app/)

Interactive scenario planner, budget optimizer and parameter recovery.

### 📚 &nbsp; [**Project documentation**](https://avova.github.io/bmmm/)

The full write-up: method, results, CLI and API reference.

✨ 🔵 ✨ 🟢 ✨ 🔵 ✨ 🟢 ✨ 🔵 ✨ 🟢 ✨ 🔵 ✨ 🟢 ✨ 🔵 ✨

</div>

A Marketing Mix Model (MMM) built on PyMC-Marketing, with the engineering pieces
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
and dashboard both read that saved result and never refit. CI runs a tiny 50-draw
fit only to check that training still works.

The dashboard and the API reach the model in two different ways, on purpose:

- The **API** (FastAPI, in Docker) loads the full PyMC model and serves live
  predictions over HTTP, so other systems can integrate with it.
- The **dashboard** does not call the API. It reads a small precomputed artifact
  (`dashboard/assets/dashboard.json`, written by `bmmm export-dashboard`) and
  recomputes everything in plain NumPy. This keeps the public Streamlit Cloud
  deploy light: no PyMC, no 92 MB model, and no always-on server to call. The
  textbook alternative (dashboard calls API, API calls model) would need a server
  running 24/7 just to back a demo, so for a free public deploy the self-contained
  dashboard wins.

## Stack

PyMC-Marketing, NumPy, ArviZ, Pydantic, Typer, FastAPI, Streamlit, Docker,
GitHub Actions, MkDocs, uv.

## License

MIT
