"""Device aging model for T2 coherence time decay and gate overrotation drift.

Implements §5.5 and §6: T2(N) = T2_0·exp(-N/Nc), idle dephasing pz, cumulative
calibration overrotation ε(t) = ε_0 + κ·elapsed, and Pauli-twirled longitudinal
relaxation λz = exp(-t/T1).  Implements NoiseContributor.
"""
from __future__ import annotations

import math

import numpy as np

from qndt.core.context import OpContext

_T2_FLOOR: float = 1e-9   # minimum T2 to prevent division by zero [s]
_PZ_MAX: float = 0.499     # maximum idle dephasing pz (NoiseContributor constraint)


class DeviceAgingModel:
    """Models exponential T2 decay and gate overrotation drift from op-count wear.

    Implements the ``NoiseContributor`` protocol via ``ptm(ctx)``.  Each quantum
    node is tracked independently by ``node_id``.

    **Phenomenological model**: ``T2(N) = T2_0·exp(−N/Nc)`` and
    ``ε(t) = ε_0 + κ·t`` are empirical fitting forms, not derived from
    first principles.  There is no single canonical reference for this specific
    wear law; it is calibrated from field/lab measurements for each device
    type.  The physically exact relation *is* ``pz = (1 − exp(−t_idle/T2))/2``
    (standard T2 dephasing relation, Nielsen & Chuang Ch. 8), which the model
    uses once T2(N) is known.

    The T1–T2–Tφ identity ``1/T2 = 1/(2T1) + 1/Tφ`` (§6) is respected by
    choosing ``t1_nominal ≥ t2_nominal / 2``.  Wear acts on the pure-dephasing
    time Tφ, which in the Bucket-B wear convention is approximated by T2(N).
    Nominal default ``t1_nominal = 200.0 s`` (200×T2_0 = 1 s) ensures
    transverse dephasing dominates (1/Tφ >> 1/(2T1)) and that the Z axis
    decays smoothly across a representative 300 s simulation window
    (λz(300) = exp(−300/200) ≈ 0.22, rather than collapsing to < 1 % by
    t ≈ 30 s as occurs with shorter T1 values).  At T1 = 200 s the
    1/(2T1) = 0.0025 s⁻¹ term is negligible vs 1/Tφ ≈ 1 s⁻¹.

    Args:
        t2_nominal: Initial T2 coherence time at zero operations [s].
        wear_const_nc: Characteristic operation count for 1/e T2 decay (Nc).
        calib_drift_rate: Gate overrotation drift rate κ [rad/s].
        gate_overrotation_0: Initial systematic overrotation ε₀ [rad].
        t1_nominal: Longitudinal relaxation time T1 [s]; not subject to wear.
            Must be > 0.  Choose ≥ t2_nominal/2 to satisfy the T1–T2–Tφ
            identity.  Default 200.0 s.

    Raises:
        ValueError: If ``t2_nominal``, ``wear_const_nc``, or ``t1_nominal``
            is not positive.
    """

    def __init__(
        self,
        t2_nominal: float,
        wear_const_nc: float,
        calib_drift_rate: float,
        gate_overrotation_0: float = 0.0,
        t1_nominal: float = 200.0,
    ) -> None:
        if t2_nominal <= 0.0:
            raise ValueError(f"t2_nominal must be > 0; got {t2_nominal}")
        if wear_const_nc <= 0.0:
            raise ValueError(f"wear_const_nc must be > 0; got {wear_const_nc}")
        if t1_nominal <= 0.0:
            raise ValueError(f"t1_nominal must be > 0; got {t1_nominal}")
        self._t2_nominal = t2_nominal
        self._wear_const_nc = wear_const_nc
        self._calib_drift_rate = calib_drift_rate
        self._gate_overrotation_0 = gate_overrotation_0
        self._t1_nominal = t1_nominal
        self._op_counts: dict[str, int] = {}
        self._first_op_time: dict[str, float] = {}
        self._node_params: dict[str, dict[str, float]] = {}

    def set_node_params(
        self,
        node_id: str,
        *,
        t2_nominal: float | None = None,
        wear_const_nc: float | None = None,
        calib_drift_rate: float | None = None,
        gate_overrotation_0: float | None = None,
        t1_nominal: float | None = None,
    ) -> None:
        """Override aging parameters for a specific node.

        Any parameter left as ``None`` falls back to the global default for
        that node.  Existing overrides for other parameters are preserved.

        Args:
            node_id: Node to configure.
            t2_nominal: Per-node T2_0 coherence time [s].  Must be > 0 if given.
            wear_const_nc: Per-node wear constant Nc.  Must be > 0 if given.
            calib_drift_rate: Per-node calibration drift rate κ.  Must be ≥ 0 if given.
            gate_overrotation_0: Per-node initial overrotation ε₀ [rad].
            t1_nominal: Per-node T1 longitudinal relaxation time [s].
                Must be > 0 if given.  Not subject to wear.

        Raises:
            ValueError: If any supplied value violates its positivity constraint.
        """
        if t2_nominal is not None and t2_nominal <= 0.0:
            raise ValueError(f"t2_nominal must be > 0; got {t2_nominal}")
        if wear_const_nc is not None and wear_const_nc <= 0.0:
            raise ValueError(f"wear_const_nc must be > 0; got {wear_const_nc}")
        if calib_drift_rate is not None and calib_drift_rate < 0.0:
            raise ValueError(f"calib_drift_rate must be >= 0; got {calib_drift_rate}")
        if t1_nominal is not None and t1_nominal <= 0.0:
            raise ValueError(f"t1_nominal must be > 0; got {t1_nominal}")

        existing = dict(self._node_params.get(node_id, {}))
        if t2_nominal is not None:
            existing["t2_nominal"] = t2_nominal
        if wear_const_nc is not None:
            existing["wear_const_nc"] = wear_const_nc
        if calib_drift_rate is not None:
            existing["calib_drift_rate"] = calib_drift_rate
        if gate_overrotation_0 is not None:
            existing["gate_overrotation_0"] = gate_overrotation_0
        if t1_nominal is not None:
            existing["t1_nominal"] = t1_nominal
        self._node_params[node_id] = existing

    def node_params(self, node_id: str) -> dict[str, float]:
        """Return the effective aging parameters for a node.

        Merges any per-node overrides (set via ``set_node_params``) over the
        global defaults so that callers always receive a complete five-key dict.

        Args:
            node_id: Node identifier.

        Returns:
            Dict with keys ``t2_nominal``, ``wear_const_nc``,
            ``calib_drift_rate``, ``gate_overrotation_0``, and ``t1_nominal``.
        """
        override = self._node_params.get(node_id, {})
        return {
            "t2_nominal": override.get("t2_nominal", self._t2_nominal),
            "wear_const_nc": override.get("wear_const_nc", self._wear_const_nc),
            "calib_drift_rate": override.get("calib_drift_rate", self._calib_drift_rate),
            "gate_overrotation_0": override.get(
                "gate_overrotation_0", self._gate_overrotation_0
            ),
            "t1_nominal": override.get("t1_nominal", self._t1_nominal),
        }

    def register_op(self, node_id: str, kind: str, t: float) -> None:
        """Record one quantum operation on a node.

        Args:
            node_id: Node identifier.
            kind: Operation type label (e.g. ``"gate"``, ``"measure"``).
            t: Simulation time of the operation [s].
        """
        self._op_counts[node_id] = self._op_counts.get(node_id, 0) + 1
        if node_id not in self._first_op_time:
            self._first_op_time[node_id] = t

    def op_count(self, node_id: str) -> int:
        """Return cumulative operation count for a node.

        Args:
            node_id: Node identifier.

        Returns:
            Number of operations registered; 0 if the node is unknown.
        """
        return self._op_counts.get(node_id, 0)

    def coherence_time(self, node_id: str, t: float) -> float:
        """Return current T2(N) coherence time for a node.

        Computes ``T2(N) = T2_0 · exp(-N / Nc)`` where N is the cumulative
        op count.  Floored at ``1e-9`` s to prevent division-by-zero in
        downstream dephasing calculations.

        Args:
            node_id: Node identifier.
            t: Current simulation time [s].

        Returns:
            T2 coherence time in seconds, at least ``1e-9``.
        """
        params = self.node_params(node_id)
        n = self._op_counts.get(node_id, 0)
        t2 = params["t2_nominal"] * math.exp(-n / params["wear_const_nc"])
        return max(t2, _T2_FLOOR)

    def gate_overrotation(self, node_id: str, t: float) -> float:
        """Return cumulative gate overrotation ε(t) for a node.

        Computes ``ε = ε_0 + κ · elapsed`` where ``elapsed = t - t_first_op``.
        Returns ``gate_overrotation_0`` for a node that has never been seen.

        Args:
            node_id: Node identifier.
            t: Current simulation time [s].

        Returns:
            Gate overrotation angle in radians.
        """
        params = self.node_params(node_id)
        elapsed = t - self._first_op_time.get(node_id, t)
        return params["gate_overrotation_0"] + params["calib_drift_rate"] * elapsed

    def idle_dephasing_pz(self, node_id: str, idle_time: float, t: float) -> float:
        """Return the Z-dephasing error probability for an idle qubit.

        Computes ``pz = 0.5 · (1 - exp(-t_idle / T2(N)))`` clamped to
        ``[0.0, 0.499]`` per the NoiseContributor rate constraint.

        Args:
            node_id: Node identifier.
            idle_time: Duration the qubit has been idle [s].
            t: Current simulation time [s].

        Returns:
            Z-error probability in ``[0.0, 0.499]``.
        """
        t2 = self.coherence_time(node_id, t)
        pz = 0.5 * (1.0 - math.exp(-idle_time / t2))
        return float(min(max(pz, 0.0), _PZ_MAX))

    def ptm(self, ctx: OpContext) -> np.ndarray:
        """Return the diagonal PTM from T2 transverse dephasing and T1 longitudinal decay.

        When ``ctx.node_id`` is ``None`` (channel-only operation with no memory
        node), returns the identity PTM ``[1, 1, 1, 1]``.

        Otherwise:
          λx = λy = 1 − 2·pz   (transverse; T2 dephasing, §5.5)
          λz = exp(−t_idle / T1)  (longitudinal; Pauli-twirled T1 decay, §6)

        Non-unital amplitude damping Φ_AD is NOT implemented here: true Φ_AD
        produces off-diagonal PTM elements and would violate the diagonal-PTM
        invariant that ChannelComposer and TCLSolver both rely on (§2).

        Args:
            ctx: Current operation context.  ``idle_time`` and ``node_id`` are used.

        Returns:
            Length-4 diagonal PTM ``[1, λx, λy, λz]``.
        """
        if ctx.node_id is None:
            return np.ones(4, dtype=np.float64)
        pz = self.idle_dephasing_pz(ctx.node_id, ctx.idle_time, ctx.t)
        lx_ly = 1.0 - 2.0 * pz
        t1 = self.node_params(ctx.node_id)["t1_nominal"]
        lz = math.exp(-ctx.idle_time / t1)
        return np.array([1.0, lx_ly, lx_ly, lz], dtype=np.float64)
