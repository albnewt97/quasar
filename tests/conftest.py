"""Shared fixtures for the Quasar test suite."""
from __future__ import annotations

import pytest


@pytest.fixture
def mock_telemetry_sample() -> dict:
    """Return a TelemetrySample-shaped dict with placeholder environmental data."""
    return {
        "t": 0.0,
        "link_id": "link_test",
        "temperature_C": 20.0,
        "seismic_ms2": 0.0,
        "wind_N": 0.0,
    }


@pytest.fixture
def small_topology() -> dict:
    """Return a 3-node linear graph dict (source → repeater → destination)."""
    return {
        "nodes": [
            {"id": "node_0", "type": "source"},
            {"id": "node_1", "type": "memory_node"},
            {"id": "node_2", "type": "destination"},
        ],
        "links": [
            {"id": "link_01", "source": "node_0", "dest": "node_1"},
            {"id": "link_12", "source": "node_1", "dest": "node_2"},
        ],
    }
