"""Environmental Telemetry Engine: kernel convolution → PauliRateVector.

Implements §3.5 and §5.2: discretised convolution of buffered environmental
samples with the memory kernel, squashing to valid Pauli rates, and online
RHP witness tracking.  Implements NoiseContributor via ptm(ctx).
"""
from __future__ import annotations

import numpy as np

from qndt.core.context import OpContext, PauliRateVector
from qndt.physics.kernels import MemoryKernel
from qndt.physics.master_equation import CanonicalRates, RHPWitness, TCLSolver
from qndt.telemetry.resampler import TelemetryResampler
from qndt.telemetry.sources import TelemetrySample


class EnvironmentalTelemetryEngine:
    """Non-Markovian noise engine driven by live environmental telemetry.

    Implements the discretised TCL convolution from docs/architecture.md §5.2:

    ``acc += K(t−t'_i) @ (S @ (E'_i − E_ref)) · Δt_i``

    The convolution drives on the deviation from a per-channel reference vector
    ``E_ref``, not on the absolute environmental value.  For temperature, this
    removes the Celsius-origin pedestal so the model responds to thermal drift
    around the operating point, not to the arbitrary 0 °C datum.

    The raw accumulation is squashed via ``0.5·tanh(u)`` to guarantee valid
    Pauli rates, then optionally renormalised if the total exceeds 0.499.

    Implements ``NoiseContributor``: ``ptm(ctx)`` delegates to ``pauli_rates``.

    Args:
        sensitivity: S matrix mapping env dims to Pauli dims, shape ``(3, M)``.
        kernel: Memory kernel instance (``ExponentialKernel`` etc.).
        resampler: Pre-constructed resampler; a default is created if ``None``.
        squash_scale: Scales the raw convolution before squashing (sensitivity
            tuning knob).
        env_ref: Environmental reference vector, shape ``(M,)``.  The
            convolution drives on ``E − env_ref`` so that a channel at its
            operating point contributes zero noise.  Defaults to
            ``[20.0, 0.0, …]`` for the standard three-channel layout (index 0 =
            temperature in °C at the 20 °C operating-point baseline; indices 1+
            are fluctuation-centred channels whose reference is 0).

    Raises:
        ValueError: If ``sensitivity`` is not a 2-D array with 3 rows, or if
            ``env_ref`` does not have shape ``(M,)``.
    """

    def __init__(
        self,
        sensitivity: np.ndarray,
        kernel: MemoryKernel,
        resampler: TelemetryResampler | None = None,
        squash_scale: float = 1.0,
        env_ref: np.ndarray | None = None,
    ) -> None:
        if sensitivity.ndim != 2 or sensitivity.shape[0] != 3:
            raise ValueError(
                f"sensitivity must be shape (3, M); got {sensitivity.shape}"
            )
        M = sensitivity.shape[1]
        if env_ref is None:
            # Default: temperature (index 0) referenced to the 20 °C operating-
            # point baseline; seismic and wind are fluctuation-centred so
            # their reference is 0.
            _ref = np.zeros(M, dtype=np.float64)
            if M >= 1:
                _ref[0] = 20.0
            self._env_ref: np.ndarray = _ref
        else:
            _ref_arr = np.asarray(env_ref, dtype=np.float64)
            if _ref_arr.shape != (M,):
                raise ValueError(
                    f"env_ref must have shape ({M},) matching sensitivity columns; "
                    f"got {_ref_arr.shape}"
                )
            self._env_ref = _ref_arr.copy()
        self._sensitivity = sensitivity
        self._kernel = kernel
        self.resampler: TelemetryResampler = (
            resampler if resampler is not None else TelemetryResampler()
        )
        self._squash_scale = squash_scale
        self._cache: dict[tuple[str, float], PauliRateVector] = {}
        self._rhp: dict[str, RHPWitness] = {}
        self._tcl: TCLSolver = TCLSolver()
        self._last_rates: dict[str, tuple[PauliRateVector, float]] = {}
        self._last_canonical_rates: dict[str, CanonicalRates] = {}

    def ingest(self, sample: TelemetrySample) -> None:
        """Push a new environmental sample into the resampler buffer.

        Invalidates any cached ``pauli_rates`` results for the same link so
        the next query re-computes with the updated data.

        Args:
            sample: Incoming environmental sample.
        """
        self.resampler.push(sample)
        stale_keys = [k for k in self._cache if k[0] == sample.link_id]
        for k in stale_keys:
            del self._cache[k]

    def pauli_rates(self, link_id: str, t: float) -> PauliRateVector:
        """Compute the Pauli error rate vector for a link at time ``t``.

        Results are cached by ``(link_id, t)`` until the next ``ingest`` for
        that link.

        If no telemetry has been pushed for ``link_id``, returns
        ``PauliRateVector(0, 0, 0)`` (no noise from this engine).

        Algorithm (§5.2 discretised convolution):
          ``acc += K(t−t'_i) @ (S @ E'_i) · Δt_i``  for i = 1 … N-1

        Squashing map:
          ``u = clip(acc·scale, 0, ∞)``
          ``p = 0.5·tanh(u)``
          ``if sum(p) > 0.499: p *= 0.499 / sum(p)``

        Args:
            link_id: Fiber link identifier.
            t: Query time [s].

        Returns:
            ``PauliRateVector`` with valid (non-negative, sum ≤ 1) rates.
        """
        cache_key = (link_id, t)
        if cache_key in self._cache:
            return self._cache[cache_key]

        samples = self.resampler.window(link_id, t)
        if not samples:
            return PauliRateVector(0.0, 0.0, 0.0)

        acc = np.zeros(3, dtype=np.float64)
        for i in range(1, len(samples)):
            tau = t - samples[i].t
            dt = samples[i].t - samples[i - 1].t
            K = self._kernel.eval(tau)
            SE = self._sensitivity @ (samples[i].E - self._env_ref)
            acc += K @ SE * dt

        acc *= self._squash_scale
        u = np.clip(acc, 0.0, None)
        p = 0.5 * np.tanh(u)
        total = float(p.sum())
        if total > 0.499:
            p = p * (0.499 / total)

        result = PauliRateVector(float(p[0]), float(p[1]), float(p[2]))
        self._cache[cache_key] = result
        self._update_rhp(link_id, result, t)
        return result

    def rhp_witness(self, link_id: str) -> RHPWitness:
        """Return the RHP witness accumulator for a link (creates one if absent).

        Args:
            link_id: Fiber link identifier.

        Returns:
            ``RHPWitness`` for ``link_id``.
        """
        return self._rhp.setdefault(link_id, RHPWitness())

    def rhp_value(self, link_id: str) -> float:
        """Return current accumulated N_RHP for a link, or 0.0 if no data.

        Args:
            link_id: Fiber link identifier.

        Returns:
            Accumulated N_RHP witness value.
        """
        if link_id not in self._rhp:
            return 0.0
        return self._rhp[link_id].current_value()

    def ptm(self, ctx: OpContext) -> np.ndarray:
        """Return the diagonal PTM from the convolved Pauli rates (§3.1, §5.2).

        Args:
            ctx: Current operation context; uses ``ctx.link_id`` and ``ctx.t``.

        Returns:
            Length-4 diagonal PTM ``[1, λx, λy, λz]``.
        """
        return self.pauli_rates(ctx.link_id, ctx.t).ptm()

    def _update_rhp(self, link_id: str, rates: PauliRateVector, t: float) -> None:
        """Update the per-link RHP witness with newly computed rates.

        On the first call for a link, stores the rates as the baseline.  On
        subsequent calls, derives ``CanonicalRates`` via the TCL solver and
        feeds them to the witness.

        Args:
            link_id: Fiber link identifier.
            rates: Freshly computed Pauli rate vector.
            t: Simulation time of this computation [s].
        """
        witness = self._rhp.setdefault(link_id, RHPWitness())
        if link_id in self._last_rates:
            last_rates, _ = self._last_rates[link_id]
            canonical: CanonicalRates = self._tcl.canonical_rates(
                rates_t=rates,
                rates_t_minus_dt=last_rates,
                t=t,
            )
            witness.update(canonical)
            self._last_canonical_rates[link_id] = canonical
        self._last_rates[link_id] = (rates, t)

    def latest_canonical_rates(self, link_id: str) -> CanonicalRates | None:
        """Return the most recently computed canonical rates for ``link_id``.

        Returns ``None`` until at least two ``pauli_rates`` calls have been
        made for this link (the TCL solver needs a previous state to
        differentiate against).

        Args:
            link_id: Fiber link identifier.
        """
        return self._last_canonical_rates.get(link_id)
