"""Coexistence Sweep example: QBER vs. classical WDM channel count.

Sweeps the number of co-propagating 1310 nm classical channels and plots
the resulting mean QBER on a single quantum link, with BB84 security
(0.11) and useless (0.25) threshold lines for reference. Uses matplotlib
(not pyqtgraph) since this is a headless, one-shot static plot.
"""
from __future__ import annotations

import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qndt.core.orchestrator import LinkConfig, NodeConfig, TwinOrchestrator
from qndt.physics.raman import ClassicalChannelSpec

_CHANNEL_COUNTS = (0, 1, 2, 4, 8, 16)
_N_STEPS = 20
_LAMBDA_C_NM = 1310.0
_BB84_BOUND = 0.11
_USELESS_THRESHOLD = 0.25
_OUTPUT_PATH = "examples/coexistence_sweep.png"


def build_orchestrator() -> TwinOrchestrator:
    """Build a single-link, two-node topology for the sweep."""
    node_configs = [
        NodeConfig(node_id="Alice", qubit_index=0),
        NodeConfig(node_id="Bob", qubit_index=1),
    ]
    link_configs = [
        LinkConfig(
            link_id="link_01",
            source_node="Alice",
            dest_node="Bob",
            lambda_q_nm=1550.0,
            gate_width_s=1e-9,
            qubit_index=0,
        ),
    ]
    return TwinOrchestrator.build_simple(
        n_qubits=len(node_configs),
        link_configs=link_configs,
        node_configs=node_configs,
        duration_s=2.0,
        dt_s=0.1,
    )


def mean_qber_for_channel_count(n_channels: int) -> float:
    """Run a fresh orchestrator with ``n_channels`` active WDM channels."""
    orchestrator = build_orchestrator()
    for i in range(n_channels):
        orchestrator._coexistence_engine.register_channel(
            ClassicalChannelSpec(f"ch_{i}", _LAMBDA_C_NM, 1.0)
        )

    for _ in range(_N_STEPS):
        orchestrator.step()

    qbers = [r.qber for r in orchestrator.results_for_link("link_01")]
    return statistics.mean(qbers)


def main() -> None:
    """Sweep channel counts, plot mean QBER, and save the figure."""
    mean_qbers = [mean_qber_for_channel_count(n) for n in _CHANNEL_COUNTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_CHANNEL_COUNTS, mean_qbers, marker="o", label="Mean QBER")
    ax.axhline(_BB84_BOUND, color="orange", linestyle="--", label="BB84 security bound (0.11)")
    ax.axhline(_USELESS_THRESHOLD, color="red", linestyle="--", label="Useless threshold (0.25)")
    ax.set_xlabel("Classical WDM channel count (1310 nm)")
    ax.set_ylabel("Mean QBER")
    ax.set_title("QBER vs. Classical Channel Coexistence Load")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_OUTPUT_PATH)

    print("Saved coexistence_sweep.png")


if __name__ == "__main__":
    main()
