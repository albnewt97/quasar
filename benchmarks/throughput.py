"""Simulation Throughput Benchmark.

Measures simulation steps per second as a function of network size
(number of links). Establishes performance baseline for the dense
fallback backend.

Run: python benchmarks/throughput.py
Outputs: benchmarks/throughput.png, prints results table.
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qndt.core.orchestrator import LinkConfig, NodeConfig, TwinOrchestrator

_LINK_COUNTS = (1, 2, 4, 8)
_STEPS_PER_RUN = 20
_WARMUP_STEPS = 5
_MIN_STEPS_PER_SEC = 1.0
_OUTPUT_PATH = "benchmarks/throughput.png"


def build_orchestrator(n_links: int) -> TwinOrchestrator:
    """Build a chain topology of ``n_links`` links and ``n_links + 1`` nodes."""
    node_configs = [
        NodeConfig(node_id=f"node_{i}", qubit_index=i) for i in range(n_links + 1)
    ]
    link_configs = [
        LinkConfig(
            link_id=f"link_{i}",
            source_node=f"node_{i}",
            dest_node=f"node_{i + 1}",
            lambda_q_nm=1550.0,
            gate_width_s=1e-9,
            qubit_index=i,
        )
        for i in range(n_links)
    ]
    return TwinOrchestrator.build_simple(
        n_qubits=n_links + 1,
        link_configs=link_configs,
        node_configs=node_configs,
    )


def measure_throughput(n_links: int) -> float:
    """Build, warm up, and time ``n_links``-link network throughput in steps/sec."""
    orchestrator = build_orchestrator(n_links)

    for _ in range(_WARMUP_STEPS):
        orchestrator.step()

    start = time.perf_counter()
    for _ in range(_STEPS_PER_RUN):
        orchestrator.step()
    elapsed = time.perf_counter() - start

    return _STEPS_PER_RUN * n_links / elapsed


def run_sweep() -> dict[int, float]:
    """Measure throughput for every configured network size."""
    return {n_links: measure_throughput(n_links) for n_links in _LINK_COUNTS}


def print_table(results: dict[int, float]) -> None:
    """Print the Links | Steps/sec | ms/step results table."""
    print(f"{'Links':>10} {'Steps/sec':>15} {'ms/step':>10}")
    for n_links in _LINK_COUNTS:
        steps_per_sec = results[n_links]
        ms_per_step = 1000.0 / steps_per_sec
        print(f"{n_links:>10} {steps_per_sec:>15.2f} {ms_per_step:>10.3f}")
        if steps_per_sec < _MIN_STEPS_PER_SEC:
            print(
                f"WARNING: {n_links}-link network below 1 step/sec threshold. "
                f"Consider reducing n_qubits or chi_max."
            )


def plot_results(results: dict[int, float]) -> None:
    """Plot throughput vs network size (log-scale y-axis) and save to PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_LINK_COUNTS, [results[n] for n in _LINK_COUNTS], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Number of links")
    ax.set_ylabel("Steps/sec (log scale)")
    ax.set_title("Simulation Throughput vs Network Size")
    fig.tight_layout()
    fig.savefig(_OUTPUT_PATH)


def main() -> None:
    """Run the sweep, print the results table, and save the plot."""
    results = run_sweep()
    print_table(results)
    plot_results(results)
    print(f"Saved {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
