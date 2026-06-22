# CLI

The `bmmm` command (Typer) drives the whole pipeline. Every command reads a YAML
config — `configs/default.yaml` for real runs, `configs/ci.yaml` for fast smoke
runs. Full reference: [`bmmm.cli`](api/cli.md).

## Install

```bash
uv sync                 # core + dev
uv sync --extra dashboard   # add Streamlit for the dashboard
```

## Commands

### `generate-data`

Write a synthetic dataset with known ground truth.

```bash
uv run bmmm generate-data --config configs/default.yaml --out artifacts/data.csv
```

### `train`

Fit the MMM and save an **artifact bundle** (`mmm.nc` + `data.csv` +
`metadata.json`) to a directory.

```bash
uv run bmmm train --config configs/default.yaml --out artifacts
```

Sampler settings (draws, chains, sampler backend) come from the config, so the
same command produces a 4000-draw production fit or a 100-draw CI fit without
code changes.

### `evaluate`

Print diagnostics, the parameter-recovery table, channel contributions and ROAS.

```bash
uv run bmmm evaluate --artifact artifacts
```

### `optimize-budget`

Compare the current allocation with the optimised one (optionally at a different
total budget).

```bash
uv run bmmm optimize-budget --artifact artifacts --budget 1600
```

### `plots`

Render the full figure set used across this site and the README.

```bash
uv run bmmm plots --artifact artifacts --out docs/img
```

## Typical workflow

```bash
uv run bmmm train          # once, offline (a couple of minutes)
uv run bmmm evaluate       # inspect recovery + diagnostics
uv run bmmm plots          # refresh figures
uv run bmmm optimize-budget --budget 1600
```
