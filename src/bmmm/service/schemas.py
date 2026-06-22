"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    channels: list[str]


class ChannelStat(BaseModel):
    channel: str
    contribution_share: float
    roas_mean: float
    adstock_alpha_recovered: float


class InfoResponse(BaseModel):
    channels: list[str]
    diagnostics: dict[str, float]
    fit_metrics: dict[str, float]
    channel_stats: list[ChannelStat]


class PredictRequest(BaseModel):
    """Weekly spend per channel."""

    allocation: dict[str, float] = Field(..., description="channel -> weekly spend")


class PredictResponse(BaseModel):
    per_channel_response: dict[str, float]
    total_response: float


class OptimizeRequest(BaseModel):
    total_budget: float = Field(..., gt=0, description="Total weekly budget to allocate")
    bounds: dict[str, tuple[float, float]] | None = Field(
        default=None, description="Optional per-channel (min, max) spend bounds"
    )


class OptimizeResponse(BaseModel):
    allocation: dict[str, float]
    total_response: float
    current_allocation: dict[str, float]
    current_response: float
    uplift_pct: float
