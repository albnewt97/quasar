# tests/test_viz.py
"""
Visualization tests
===================

Covers:
- plot_physical, plot_protocol, plot_security return matplotlib Figure
- Figures can be saved without error
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd

from sequence_ext.io.metrics import MetricFrame
from sequence_ext.viz.plots import plot_physical, plot_protocol, plot_security


def _dummy_metricframe() -> MetricFrame:
    times = list(range(5))
    mf = MetricFrame()
    mf.physical = pd.DataFrame({"time": times, "loss_dB": [0, 1, 2, 3, 4], "bsm_vis": [0.9, 0.91, 0.92, 0.93, 0.94]})
    mf.network = pd.DataFrame({"time": times, "utilization": [0.1, 0.2, 0.3, 0.4, 0.5]})
    mf.protocol = pd.DataFrame({"time": times, "qber": [0.01, 0.02, 0.015, 0.017, 0.018], "sifted_rate": [100, 200, 300, 400, 500]})
    mf.security = pd.DataFrame({"time": times, "secret_rate": [10, 20, 15, 18, 17]})
    return mf


def test_plot_physical_returns_figure(tmp_path: Path):
    mf = _dummy_metricframe()
    fig = plot_physical(mf)
    assert isinstance(fig, matplotlib.figure.Figure)
    out = tmp_path / "phys.png"
    fig.savefig(out)
    assert out.exists()


def test_plot_protocol_returns_figure(tmp_path: Path):
    mf = _dummy_metricframe()
    fig = plot_protocol(mf)
    assert isinstance(fig, matplotlib.figure.Figure)
    out = tmp_path / "prot.png"
    fig.savefig(out)
    assert out.exists()


def test_plot_security_returns_figure(tmp_path: Path):
    mf = _dummy_metricframe()
    fig = plot_security(mf)
    assert isinstance(fig, matplotlib.figure.Figure)
    out = tmp_path / "sec.png"
    fig.savefig(out)
    assert out.exists()
