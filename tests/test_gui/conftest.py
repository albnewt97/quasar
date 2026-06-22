"""Shared fixtures for GUI tests: headless Qt platform plugin (§8.5)."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def qapp_env() -> None:
    """Force the offscreen Qt platform plugin so GUI tests run headless."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
