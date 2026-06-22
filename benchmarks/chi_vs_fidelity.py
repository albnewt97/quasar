"""Bond Dimension vs Fidelity Scaling Benchmark.

Measures how TensorStateTracker fidelity degrades as chi_max is reduced,
for increasing system sizes. Demonstrates that the dense fallback is exact
(chi_max has no effect in dense mode) and establishes the scaling baseline
for future tensor-network backend integration.

Run: python benchmarks/chi_vs_fidelity.py
Outputs: benchmarks/chi_vs_fidelity.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qndt.physics.channels import depolarising_ptm
from qndt.quantum.tracker import TensorStateTracker

_N_QUBITS_RANGE = (2, 3, 4)
_CHI_RANGE = (1, 2, 4, 8, 16, 32)
_N_NOISE_STEPS = 10
_NOISE_P = 0.05
_OUTPUT_PATH = "benchmarks/chi_vs_fidelity.png"


def run_sweep() -> dict[tuple[int, int], float]:
    """Run the (n_qubits, chi_max) sweep and return final Bell-pair fidelities."""
    ptm = depolarising_ptm(_NOISE_P)
    results: dict[tuple[int, int], float] = {}

    for n_qubits in _N_QUBITS_RANGE:
        for chi in _CHI_RANGE:
            tracker = TensorStateTracker(n_qubits, chi_max=chi)
            tracker.entangle(0, 1)
            for _ in range(_N_NOISE_STEPS):
                tracker.apply_channel(0, ptm)
            results[(n_qubits, chi)] = tracker.fidelity(0, 1)

    return results


def print_table(results: dict[tuple[int, int], float]) -> None:
    """Print the (n_qubits, chi_max) -> fidelity results table."""
    print(f"{'n_qubits':>10} {'chi_max':>10} {'fidelity':>10}")
    for n_qubits in _N_QUBITS_RANGE:
        for chi in _CHI_RANGE:
            print(f"{n_qubits:>10} {chi:>10} {results[(n_qubits, chi)]:>10.6f}")


def plot_results(results: dict[tuple[int, int], float]) -> None:
    """Plot fidelity vs chi_max, one line per n_qubits, and save to PNG."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_qubits in _N_QUBITS_RANGE:
        fidelities = [results[(n_qubits, chi)] for chi in _CHI_RANGE]
        ax.plot(_CHI_RANGE, fidelities, marker="o", label=f"n_qubits={n_qubits}")

    ax.set_xlabel("chi_max")
    ax.set_ylabel("Fidelity")
    ax.set_title("Bond Dimension vs Fidelity (dense fallback: chi has no effect)")
    ax.legend()
    ax.annotate(
        "Note: dense fallback active — chi_max is a config\n"
        "parameter only; all values give identical results in this backend.",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        style="italic",
    )
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
