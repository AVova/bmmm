"""Artifact bundle: persist and load everything the service/dashboard need.

A trained run produces three files in one directory:

    mmm.nc          fitted PyMC-Marketing model (posterior + metadata)
    data.csv        the (synthetic) dataset the model was trained on
    metadata.json   config, diagnostics, ground truth and a timestamp

Loading the bundle never re-samples; it just reads these files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bmmm.config import Config
from bmmm.data.generate import GroundTruth
from bmmm.model.mmm import MMM, load_mmm, save_mmm

DEFAULT_DIR = Path("artifacts")
MODEL_FILE = "mmm.nc"
DATA_FILE = "data.csv"
META_FILE = "metadata.json"


@dataclass
class Bundle:
    """An in-memory loaded artifact bundle."""

    mmm: MMM
    data: pd.DataFrame
    metadata: dict


def save_bundle(
    out_dir: str | Path,
    mmm: MMM,
    data: pd.DataFrame,
    config: Config,
    ground_truth: GroundTruth,
    diagnostics: dict[str, float],
) -> Path:
    """Write model, data and metadata to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_mmm(mmm, out / MODEL_FILE)
    data.to_csv(out / DATA_FILE, index=False)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.model_dump(),
        "diagnostics": diagnostics,
        "ground_truth": ground_truth.as_records(),
        "channels": config.data.channel_names,
    }
    (out / META_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return out


def load_bundle(in_dir: str | Path = DEFAULT_DIR) -> Bundle:
    """Load a previously saved bundle. No sampling happens."""
    d = Path(in_dir)
    mmm = load_mmm(d / MODEL_FILE)
    data = pd.read_csv(d / DATA_FILE, parse_dates=["date"])
    metadata = json.loads((d / META_FILE).read_text())
    return Bundle(mmm=mmm, data=data, metadata=metadata)
