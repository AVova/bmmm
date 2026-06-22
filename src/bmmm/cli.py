"""Command-line interface for the BMMM pipeline.

    bmmm generate-data   # write a synthetic dataset
    bmmm train           # fit the model, save an artifact bundle
    bmmm evaluate        # diagnostics, parameter recovery, ROAS
    bmmm optimize-budget # current vs optimised allocation
    bmmm plots           # render the figure set

Sampler/data/model settings all come from a YAML config (default:
``configs/default.yaml``); CI uses ``configs/ci.yaml``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from bmmm import artifacts
from bmmm.config import Config
from bmmm.data.generate import generate
from bmmm.model import analysis, budget
from bmmm.model.mmm import build_mmm, fit_mmm

app = typer.Typer(add_completion=False, help="Bayesian Marketing Mix Modeling pipeline.")
console = Console()

ConfigOpt = Annotated[Path, typer.Option(help="Path to YAML config.")]
ArtifactOpt = Annotated[Path, typer.Option(help="Artifact bundle directory.")]


def _df_to_table(df: pd.DataFrame, title: str) -> Table:
    table = Table(title=title)
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    return table


@app.command("generate-data")
def generate_data(
    config: ConfigOpt = Path("configs/default.yaml"),
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path("artifacts/data.csv"),
) -> None:
    """Generate a synthetic MMM dataset with known ground truth."""
    cfg = Config.from_yaml(config)
    df, gt = generate(cfg.data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"[green]Wrote {len(df)} rows to {out}[/green]")
    console.print(_df_to_table(pd.DataFrame(gt.as_records()), "Ground truth"))


@app.command()
def train(
    config: ConfigOpt = Path("configs/default.yaml"),
    out: ArtifactOpt = artifacts.DEFAULT_DIR,
    progress: Annotated[bool, typer.Option(help="Show sampler progress bar.")] = True,
) -> None:
    """Fit the MMM and save an artifact bundle (model + data + metadata)."""
    warnings.simplefilter("ignore")
    cfg = Config.from_yaml(config)
    df, gt = generate(cfg.data)

    console.print(f"[cyan]Fitting MMM ({cfg.sampler.nuts_sampler}, "
                  f"{cfg.sampler.draws} draws x {cfg.sampler.chains} chains)...[/cyan]")
    mmm = build_mmm(cfg)
    fit_mmm(mmm, df, cfg, progressbar=progress)

    diag = analysis.diagnostics(mmm)
    artifacts.save_bundle(out, mmm, df, cfg, gt, diag)
    console.print(f"[green]Saved bundle to {out}/[/green]")
    console.print(f"diagnostics: {diag}")


@app.command()
def evaluate(
    artifact: ArtifactOpt = artifacts.DEFAULT_DIR,
    config: ConfigOpt = Path("configs/default.yaml"),
) -> None:
    """Print diagnostics, parameter recovery, contributions and ROAS."""
    warnings.simplefilter("ignore")
    bundle = artifacts.load_bundle(artifact)
    cfg = Config.from_yaml(config)
    _, gt = generate(cfg.data)

    console.print(f"[bold]Diagnostics:[/bold] {analysis.diagnostics(bundle.mmm)}")
    console.print(f"[bold]Fit metrics:[/bold] {analysis.fit_metrics(bundle.mmm, bundle.data)}")
    console.print(_df_to_table(analysis.recovery_table(bundle.mmm, gt), "Parameter recovery (adstock alpha)"))
    console.print(_df_to_table(analysis.channel_contributions(bundle.mmm), "Channel contributions"))
    console.print(_df_to_table(analysis.roas_table(bundle.mmm, bundle.data), "ROAS"))


@app.command("optimize-budget")
def optimize_budget_cmd(
    artifact: ArtifactOpt = artifacts.DEFAULT_DIR,
    budget_total: Annotated[float, typer.Option("--budget", help="Total weekly budget; default = current.")] = 0.0,
) -> None:
    """Compare current vs optimised budget allocation."""
    warnings.simplefilter("ignore")
    bundle = artifacts.load_bundle(artifact)
    total = budget_total if budget_total > 0 else None
    summary = budget.optimization_summary(bundle.mmm, bundle.data, total_budget=total)
    console.print(_df_to_table(summary, "Budget optimisation (weekly)"))
    uplift = 100 * (summary["optimal_response"].sum() / summary["current_response"].sum() - 1)
    console.print(f"[green]Response uplift at equal budget: {uplift:.2f}%[/green]")


@app.command()
def plots(
    artifact: ArtifactOpt = artifacts.DEFAULT_DIR,
    config: ConfigOpt = Path("configs/default.yaml"),
    out: Annotated[Path, typer.Option(help="Figure output directory.")] = Path("docs/img"),
) -> None:
    """Render the full figure set used in the README."""
    import matplotlib

    matplotlib.use("Agg")
    warnings.simplefilter("ignore")
    from bmmm.viz import plots as viz

    bundle = artifacts.load_bundle(artifact)
    _, gt = generate(Config.from_yaml(config).data)
    paths = viz.save_all(bundle.mmm, bundle.data, gt, out)
    console.print(f"[green]Wrote {len(paths)} figures to {out}/[/green]")


@app.command("export-dashboard")
def export_dashboard_cmd(
    artifact: ArtifactOpt = artifacts.DEFAULT_DIR,
    config: ConfigOpt = Path("configs/default.yaml"),
    out: Annotated[Path, typer.Option(help="Dashboard assets directory.")] = Path("dashboard/assets"),
) -> None:
    """Export the compact dashboard artifact (JSON + figures), no PyMC at runtime."""
    import matplotlib

    matplotlib.use("Agg")
    warnings.simplefilter("ignore")
    from bmmm.dashboard_export import export_dashboard

    path = export_dashboard(artifact, out, config)
    console.print(f"[green]Wrote {path} and figures to {out}/img/[/green]")


if __name__ == "__main__":
    app()
