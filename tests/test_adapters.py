"""Tests for telemetry format adapters and sensitivity-matrix fitting (§2)."""
from __future__ import annotations

import pandas as pd
import pytest

from qndt.io.adapters import AdapterRegistry, DataFrameAdapter, RESTPollingAdapter
from qndt.telemetry.calibration import SensitivityFitter, smf28_calibration

# ---------------------------------------------------------------------------
# DataFrameAdapter
# ---------------------------------------------------------------------------


def test_dataframe_adapter() -> None:
    """DataFrameAdapter yields correctly-shaped TelemetrySamples from a DataFrame."""
    df = pd.DataFrame(
        {
            "t": [0.0, 1.0],
            "temp": [20.0, 21.0],
            "seis": [0.0, 0.001],
            "wind": [0.1, 0.2],
        }
    )
    adapter = DataFrameAdapter(df, t_col="t", env_cols=["temp", "seis", "wind"], link_id="l1")

    samples = list(adapter)

    assert len(samples) == 2
    assert samples[0].t == 0.0
    assert samples[0].E.tolist() == [20.0, 0.0, 0.1]
    assert samples[1].link_id == "l1"


def test_dataframe_adapter_missing_column() -> None:
    """DataFrameAdapter raises KeyError with a clear message for missing columns."""
    df = pd.DataFrame({"t": [0.0], "temp": [20.0]})
    adapter = DataFrameAdapter(df, t_col="t", env_cols=["temp", "seis"], link_id="l1")

    with pytest.raises(KeyError, match="seis"):
        list(adapter)


# ---------------------------------------------------------------------------
# AdapterRegistry
# ---------------------------------------------------------------------------


def test_adapter_registry_available() -> None:
    """csv, synthetic, dataframe, and rest are all registered by default."""
    available = AdapterRegistry.available()
    for name in ("csv", "synthetic", "dataframe", "rest"):
        assert name in available


def test_adapter_registry_create_synthetic() -> None:
    """Creating a synthetic adapter via the registry yields the expected sample count."""
    source = AdapterRegistry.create(
        "synthetic", link_id="l1", duration_s=1.0, dt_s=0.5
    )
    samples = list(source)
    assert len(samples) == 2


def test_adapter_registry_unknown() -> None:
    """Creating an unregistered adapter name raises KeyError."""
    with pytest.raises(KeyError):
        AdapterRegistry.create("nonexistent")


# ---------------------------------------------------------------------------
# RESTPollingAdapter
# ---------------------------------------------------------------------------


def test_rest_adapter_bad_url() -> None:
    """An unreachable URL raises RuntimeError once retries are exhausted."""
    adapter = RESTPollingAdapter(
        "http://127.0.0.1:1/nonexistent",
        field_map={"temp": 0},
        link_id="l1",
        max_retries=1,
    )

    with pytest.raises(RuntimeError):
        next(iter(adapter))


# ---------------------------------------------------------------------------
# SensitivityFitter / calibration
# ---------------------------------------------------------------------------


def test_sensitivity_fitter_shape() -> None:
    """fit() returns a (3, 3) sensitivity matrix for the SMF-28 calibration set."""
    S = SensitivityFitter().fit(smf28_calibration())
    assert S.shape == (3, 3)


def test_sensitivity_fitter_r_squared() -> None:
    """The fit recovers most of the variance despite 5% measurement noise."""
    dataset = smf28_calibration()
    S = SensitivityFitter().fit(dataset)
    assert SensitivityFitter().r_squared(dataset, S) > 0.5


def test_calibration_dataset_properties() -> None:
    """smf28_calibration() produces exactly 10 samples."""
    assert smf28_calibration().n_samples == 10
