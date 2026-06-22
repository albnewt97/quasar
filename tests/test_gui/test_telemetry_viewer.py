"""Tests for TelemetryViewer (§7.5).

Run headless via ``QT_QPA_PLATFORM=offscreen`` (see conftest.py qapp_env).
"""
from __future__ import annotations

from pytestqt.qtbot import QtBot

from qndt.gui.telemetry_viewer import TelemetryViewer
from qndt.telemetry.sources import SyntheticTelemetrySource


def _one_sample() -> object:
    source = SyntheticTelemetrySource(link_id="link_01", duration_s=1.0, dt_s=0.1)
    return next(iter(source))


def test_telemetry_viewer_creates(qtbot: QtBot) -> None:
    """TelemetryViewer() constructs without error."""
    viewer = TelemetryViewer()
    qtbot.addWidget(viewer)
    assert viewer is not None


def test_ingest_sample(qtbot: QtBot) -> None:
    """Ingesting one sample appends exactly one point to the temp buffer."""
    viewer = TelemetryViewer()
    qtbot.addWidget(viewer)

    viewer.ingest_sample(_one_sample())

    assert len(viewer._xs["temp"]) == 1


def test_pause_stops_ingest(qtbot: QtBot) -> None:
    """While paused, ingest_sample() is a no-op."""
    viewer = TelemetryViewer()
    qtbot.addWidget(viewer)

    viewer._toggle_pause()
    viewer.ingest_sample(_one_sample())

    assert len(viewer._xs["temp"]) == 0


def test_stale_label_hidden_initially(qtbot: QtBot) -> None:
    """The stale-data warning label is hidden before any samples arrive."""
    viewer = TelemetryViewer()
    qtbot.addWidget(viewer)

    assert viewer._stale_label.isVisible() is False


def test_clear_resets(qtbot: QtBot) -> None:
    """clear() empties all channel buffers after several ingests."""
    viewer = TelemetryViewer()
    qtbot.addWidget(viewer)

    source = SyntheticTelemetrySource(link_id="link_01", duration_s=1.0, dt_s=0.1)
    samples = iter(source)
    for _ in range(3):
        viewer.ingest_sample(next(samples))

    viewer.clear()

    for key in ("temp", "seis", "wind"):
        assert viewer._xs[key] == []
        assert viewer._ys[key] == []
