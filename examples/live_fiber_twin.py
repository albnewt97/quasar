"""Live Fiber Twin example (headless full-stack demonstration).

4-node linear network (Alice -> R1 -> R2 -> Bob) driven by synthetic
telemetry, Raman co-existence from two classical WDM channels, device
aging, and classical packet routing -- all wired through TwinOrchestrator
exactly as the GUI's SimulationController does, but without Qt.
"""
from __future__ import annotations

import statistics

from qndt.core.orchestrator import LinkConfig, NodeConfig, TwinOrchestrator
from qndt.physics.key_rate import BB84KeyRateCalculator, KeyRateParams
from qndt.physics.raman import ClassicalChannelSpec

_NODE_IDS = ("Alice", "R1", "R2", "Bob")
_N_STEPS = 50


def build_orchestrator() -> TwinOrchestrator:
    """Build a 4-node, 3-link linear quantum repeater chain."""
    node_configs = [
        NodeConfig(node_id=node_id, qubit_index=i) for i, node_id in enumerate(_NODE_IDS)
    ]
    link_configs = [
        LinkConfig(
            link_id="link_01",
            source_node="Alice",
            dest_node="R1",
            lambda_q_nm=1550.0,
            gate_width_s=1e-9,
            qubit_index=0,
        ),
        LinkConfig(
            link_id="link_02",
            source_node="R1",
            dest_node="R2",
            lambda_q_nm=1550.0,
            gate_width_s=1e-9,
            qubit_index=1,
        ),
        LinkConfig(
            link_id="link_03",
            source_node="R2",
            dest_node="Bob",
            lambda_q_nm=1550.0,
            gate_width_s=1e-9,
            qubit_index=2,
        ),
    ]
    return TwinOrchestrator.build_simple(
        n_qubits=len(node_configs),
        link_configs=link_configs,
        node_configs=node_configs,
        duration_s=5.0,
        dt_s=0.1,
    )


def main() -> None:
    """Run the demo: register WDM channels, route packets, simulate, report."""
    orchestrator = build_orchestrator()

    orchestrator._coexistence_engine.register_channel(
        ClassicalChannelSpec("ch_1310", 1310.0, 1.0)
    )
    orchestrator._coexistence_engine.register_channel(
        ClassicalChannelSpec("ch_1550_cw", 1530.0, 0.5)
    )

    control_plane = orchestrator._control_plane
    control_plane.route_packet("pkt_001", "Alice", "Bob", 0.0)
    control_plane.route_packet("pkt_002", "Alice", "R1", 0.0)
    control_plane.route_packet("pkt_003", "R2", "Bob", 0.0)

    kr_calc = BB84KeyRateCalculator(
        KeyRateParams(
            mu=0.1,
            f_ec=1.16,
            detector_efficiency=0.8,
            dark_count_rate=1e-5,
            repetition_rate_hz=1e9,
        )
    )

    print(
        f"{'t':>6} {'Link':>12} {'QBER':>8} {'Fidelity':>10} "
        f"{'Raman Hz':>12} {'SKR bps':>12} {'Secure':>8}"
    )
    link_ids = [link.link_id for link in orchestrator._config.links]
    for _ in range(_N_STEPS):
        for result in orchestrator.step():
            kr = kr_calc.calculate(result.qber)
            print(
                f"{result.t:6.2f} {result.link_id:>12} {result.qber:8.4f} "
                f"{result.fidelity:10.4f} {result.raman_rate_hz:12.2e} "
                f"{kr.secret_key_rate_bps:12.3e} {str(kr.is_positive):>8}"
            )

    print()
    print("Summary statistics")
    print("-------------------")
    any_backflow = False
    for link_id in link_ids:
        series = orchestrator.results_for_link(link_id)
        qbers = [r.qber for r in series]
        fidelities = [r.fidelity for r in series]
        raman_rates = [r.raman_rate_hz for r in series]
        rhp_values = [r.rhp_witness for r in series]
        skr_series = [kr_calc.calculate(q).secret_key_rate_bps for q in qbers]
        n_secure = sum(1 for q in qbers if kr_calc.calculate(q).is_positive)
        print(
            f"{link_id}: mean QBER={statistics.mean(qbers):.4f}, "
            f"std QBER={statistics.pstdev(qbers):.4f}, "
            f"min fidelity={min(fidelities):.4f}, "
            f"max Raman={max(raman_rates):.2e} Hz, "
            f"mean SKR={statistics.mean(skr_series):.3e} bps, "
            f"secure steps={n_secure}/{len(qbers)}"
        )
        if any(v > 0.0 for v in rhp_values):
            any_backflow = True

    print(f"Non-Markovian behaviour detected (N_RHP > 0): {any_backflow}")

    dist = kr_calc.distance_budget(fiber_loss_db_per_km=0.2)
    print(f"\nDistance budget (0.2 dB/km fiber): {dist:.1f} km")


if __name__ == "__main__":
    main()
