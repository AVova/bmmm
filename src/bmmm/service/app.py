"""FastAPI service exposing the trained MMM.

The model is loaded once at startup from an artifact bundle; every endpoint
answers from the persisted posterior (response curves) — no MCMC at request
time. Point ``BMMM_ARTIFACTS`` at a bundle directory (default ``artifacts``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from bmmm import artifacts
from bmmm.model import analysis, budget
from bmmm.service import schemas


class _State:
    """Holds the loaded bundle and derived response curves."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle = artifacts.load_bundle(bundle_dir)
        self.curves = budget.response_curves(self.bundle.mmm, self.bundle.data)
        self.channels = list(self.bundle.mmm.channel_columns)
        self.current = budget.current_allocation(self.bundle.data, self.channels)


@lru_cache(maxsize=1)
def get_state() -> _State:
    bundle_dir = Path(os.environ.get("BMMM_ARTIFACTS", str(artifacts.DEFAULT_DIR)))
    return _State(bundle_dir)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Warm the model on startup so the first request isn't slow.
    # If no bundle exists the app still starts; /health reports model_loaded=False.
    with suppress(FileNotFoundError, OSError):
        get_state()
    yield


app = FastAPI(title="BMMM Service", version="0.1.0", lifespan=lifespan)


def _safe_state() -> _State | None:
    try:
        return get_state()
    except (FileNotFoundError, OSError):
        return None


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    state = _safe_state()
    return schemas.HealthResponse(
        status="ok",
        model_loaded=state is not None,
        channels=state.channels if state else [],
    )


@app.get("/info", response_model=schemas.InfoResponse)
def info() -> schemas.InfoResponse:
    state = get_state()
    mmm, data = state.bundle.mmm, state.bundle.data
    share = analysis.channel_contributions(mmm).set_index("channel")["contribution_share"].to_dict()
    roas = analysis.roas_table(mmm, data).set_index("channel")["roas_mean"].to_dict()
    recovered = mmm.format_recovered_transformation_parameters(quantile=0.5)
    stats = [
        schemas.ChannelStat(
            channel=ch,
            contribution_share=float(share[ch]),
            roas_mean=float(roas[ch]),
            adstock_alpha_recovered=float(recovered[ch]["adstock_params"]["alpha"]),
        )
        for ch in state.channels
    ]
    return schemas.InfoResponse(
        channels=state.channels,
        diagnostics=analysis.diagnostics(mmm),
        fit_metrics=analysis.fit_metrics(mmm, data),
        channel_stats=stats,
    )


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(req: schemas.PredictRequest) -> schemas.PredictResponse:
    state = get_state()
    unknown = set(req.allocation) - set(state.channels)
    if unknown:
        raise HTTPException(422, f"Unknown channels: {sorted(unknown)}")
    per_channel = {
        ch: float(state.curves[ch].response(spend)) for ch, spend in req.allocation.items()
    }
    return schemas.PredictResponse(
        per_channel_response=per_channel,
        total_response=float(sum(per_channel.values())),
    )


@app.post("/optimize-budget", response_model=schemas.OptimizeResponse)
def optimize_budget(req: schemas.OptimizeRequest) -> schemas.OptimizeResponse:
    state = get_state()
    if req.bounds:
        unknown = set(req.bounds) - set(state.channels)
        if unknown:
            raise HTTPException(422, f"Unknown channels in bounds: {sorted(unknown)}")
    alloc = budget.optimize_budget(state.curves, req.total_budget, bounds=req.bounds)
    total = budget.total_response(alloc, state.curves)
    current_resp = budget.total_response(state.current, state.curves)
    uplift = 100 * (total / current_resp - 1) if current_resp > 0 else 0.0
    return schemas.OptimizeResponse(
        allocation=alloc,
        total_response=total,
        current_allocation=state.current,
        current_response=current_resp,
        uplift_pct=uplift,
    )
