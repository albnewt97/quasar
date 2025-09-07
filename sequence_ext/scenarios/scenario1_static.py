from dataclasses import dataclass
from typing import Dict, Any

# Placeholders for SeQUeNCe imports (to be replaced with real ones)
# from sequence.kernel.timeline import Timeline
# from sequence.components.detector import Detector
# from sequence.components.bsm import TimeBinBSM
# from sequence.components.optical_channel import QuantumChannel, ClassicalChannel
# from sequence.topology.node import Node

from sequence_ext.orchestrator.mdi_controller import MDIController
from sequence_ext.io.recorder import Recorder


@dataclass
class ScenarioResults:
    meta: Dict[str, Any]
    kpis: Dict[str, Any]
    events_path: str


class Scenario1Static:
    """Scenario 1: Static nodes (1a equidistant, 1b uneven distances).

    This is a minimal placeholder that wires the QUASAR orchestrator hooks and
    emits a fake result structure so the CLI and dashboard can be exercised.
    Replace the TODOs with real SeQUeNCe setup.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.recorder = Recorder()
        self.controller = MDIController(self.recorder)

    def run(self) -> ScenarioResults:
        # TODO: Build SeQUeNCe timeline, nodes, channels from config.
        # TODO: Register callbacks from detectors/BSM into self.recorder.
        # TODO: Start timeline, wait until stop_time.
        # For now, emit minimal synthetic KPIs to prove the pipeline.
        kpis = {
            "physical": {"coincidence_rate_hz": 0.0},
            "protocol": {"raw_key_rate_bps": 0.0, "qber": 0.0},
            "security": {"privacy_throughput_bps": 0.0},
        }
        meta = {"scenario": self.cfg.get("scenario", {}), "nodes": self.cfg.get("nodes", [])}
        events_path = self.recorder.flush("data/runs/events_placeholder.parquet")
        return ScenarioResults(meta=meta, kpis=kpis, events_path=events_path)

    def export(self, results: ScenarioResults, out_dir: str) -> None:
        # TODO: write KPIs to parquet/csv
        import json, pathlib

        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "kpis.json").write_text(json.dumps(results.kpis, indent=2))
        (out / "meta.json").write_text(json.dumps(results.meta, indent=2))
        # Optionally copy events file
        # Real pipeline will write directly into out_dir via Recorder.
