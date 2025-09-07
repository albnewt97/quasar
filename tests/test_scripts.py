# tests/test_scripts.py
"""
Script entrypoint tests
=======================

Covers:
- run_scenario.py runs successfully with a config
- sweep.py generates summary.csv for a small sweep
- report.py generates HTML report with plots

These are smoke tests: they check CLI wiring, not deep correctness.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest

# Path to scripts
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
PYTHON = sys.executable


def _run_script(name: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    script = SCRIPTS_DIR / name
    proc = subprocess.run(
        [PYTHON, str(script)] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return proc


def test_run_scenario_equ(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
        scenario:
          name: scenario1_equidistant
          pulse_rate_hz: 1000000
          duration_s: 0.01
          output_dir: run
        devices:
          detector_eta: 0.8
          detector_dark_per_gate: 1e-6
          detector_dead_time_ns: 60
          detector_afterpulse: 0.02
          bsm_visibility: 0.98
          coincidence_window_ps: 500
        weather: null
        """,
        encoding="utf-8",
    )

    proc = _run_script("run_scenario.py", [str(cfg)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "Scenario finished" in proc.stdout + proc.stderr


def test_sweep_generates_summary(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
        scenario:
          name: scenario1_equidistant
          pulse_rate_hz: 1000000
          duration_s: 0.01
          output_dir: run
        devices:
          detector_eta: 0.8
          detector_dark_per_gate: 1e-6
          detector_dead_time_ns: 60
          detector_afterpulse: 0.02
          bsm_visibility: 0.98
          coincidence_window_ps: 500
        weather: null
        """,
        encoding="utf-8",
    )
    out_dir = tmp_path / "sweep"
    args = [str(cfg), "--out", str(out_dir), "--pulse-rate", "1000000", "--duration", "0.01", "--workers", "1"]
    proc = _run_script("sweep.py", args, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "summary.csv").exists()


def test_report_generates_html(tmp_path: Path):
    # Create fake run_dir with minimal parquet/csv
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    import pandas as pd

    times = [0, 1, 2]
    pd.DataFrame({"time": times, "loss_dB": [0, 1, 2], "bsm_vis": [0.9, 0.91, 0.92]}).to_csv(run_dir / "physical.csv", index=False)
    pd.DataFrame({"time": times, "utilization": [0.1, 0.2, 0.3]}).to_csv(run_dir / "network.csv", index=False)
    pd.DataFrame({"time": times, "qber": [0.01, 0.02, 0.03], "sifted_rate": [100, 200, 300]}).to_csv(run_dir / "protocol.csv", index=False)
    pd.DataFrame({"time": times, "secret_rate": [10, 20, 30]}).to_csv(run_dir / "security.csv", index=False)

    out_html = tmp_path / "report.html"
    proc = _run_script("report.py", ["--run-dir", str(run_dir), "--out", str(out_html)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert out_html.exists()
    txt = out_html.read_text(encoding="utf-8")
    assert "<html" in txt.lower()
