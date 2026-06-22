"""Smoke test: package imports and exposes a version."""

import bmmm


def test_version() -> None:
    assert bmmm.__version__
