# API service

A thin [FastAPI](https://fastapi.tiangolo.com/) service exposes the trained
model. It loads the artifact bundle **once at startup** and answers every request
from the persisted posterior — there is **no MCMC at request time**. Full
reference: [`bmmm.service.app`](api/service-app.md).

## Run it

```bash
uv run uvicorn bmmm.service.app:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

The bundle directory defaults to `artifacts/`; override with the `BMMM_ARTIFACTS`
environment variable.

## Endpoints

### `GET /health`

Liveness + whether a model is loaded.

```json
{ "status": "ok", "model_loaded": true, "channels": ["tv", "social", "search"] }
```

### `GET /info`

Diagnostics, fit metrics and per-channel stats (contribution share, ROAS,
recovered adstock α).

### `POST /predict`

Predicted weekly contribution for a given spend allocation.

```bash
curl -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"allocation": {"tv": 600, "social": 350, "search": 250}}'
```

```json
{
  "per_channel_response": { "tv": 731.0, "social": 1036.0, "search": 375.0 },
  "total_response": 2142.0
}
```

### `POST /optimize-budget`

Optimal allocation for a total budget, with the uplift versus the current split.

```bash
curl -X POST localhost:8000/optimize-budget \
  -H 'content-type: application/json' \
  -d '{"total_budget": 1600}'
```

Request/response schemas are defined in [`bmmm.service.schemas`](api/service-schemas.md)
and validated by Pydantic, so FastAPI generates the OpenAPI docs automatically.
