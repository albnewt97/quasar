"""BB84 secret key rate estimator (GLLP/Shor-Preskill asymptotic formula).

Standalone physics module: no Qt, no orchestrator dependency, no I/O.
Any simulation layer can import and call ``BB84KeyRateCalculator`` directly.

References:
    Gottesman, Lo, Lütkenhaus, Preskill, Quant. Inf. Comput. 4, 325–360
    (2004). [GLLP]
    Ma et al., Phys. Rev. A 72, 012326 (2005). [decoy states]
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_SUPPORTED_PROTOCOLS: frozenset[str] = frozenset({"bb84", "bb84_decoy"})


def binary_entropy(p: float) -> float:
    """Binary Shannon entropy H₂(p) = −p·log₂(p) − (1−p)·log₂(1−p).

    Args:
        p: Probability value in ``[0, 1]``.

    Returns:
        H₂(p) in ``[0, 1]``; exactly 0.0 at the boundaries (p=0 or p=1),
        and 1.0 at the maximum (p=0.5).

    Raises:
        ValueError: If ``p < 0`` or ``p > 1``.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if p < 1e-15 or p > 1.0 - 1e-15:
        return 0.0
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


@dataclass(frozen=True, slots=True)
class KeyRateParams:
    """Physical and protocol parameters for the BB84 key rate estimator.

    Args:
        mu: Mean photon number per pulse (weak coherent source).  Typical
            range 0.01–0.5; default 0.1 is standard WCP BB84.
        f_ec: Error correction inefficiency.  f=1.0 is the Shannon limit;
            f=1.16 is realistic for CASCADE/LDPC.  Must be >= 1.0.
        detector_efficiency: Single-photon detector efficiency η_d in (0, 1].
        dark_count_rate: Per-gate dark count probability p_dc in (0, 1).
        repetition_rate_hz: Clock rate of the QKD system [Hz].  Used to
            convert per-pulse rate to bits/second.
        protocol: Security analysis model.  One of ``"bb84"`` (standard
            GLLP) or ``"bb84_decoy"`` (with decoy-state yield estimation).

    Raises:
        ValueError: If any field violates its domain constraint.
    """

    mu: float = 0.1
    f_ec: float = 1.16
    detector_efficiency: float = 0.8
    dark_count_rate: float = 1e-5
    repetition_rate_hz: float = 1e9
    protocol: str = "bb84"

    def __post_init__(self) -> None:
        if self.mu <= 0.0:
            raise ValueError(f"mu must be > 0; got {self.mu}")
        if self.f_ec < 1.0:
            raise ValueError(f"f_ec must be >= 1.0 (Shannon limit); got {self.f_ec}")
        if not (0.0 < self.detector_efficiency <= 1.0):
            raise ValueError(
                f"detector_efficiency must be in (0, 1]; got {self.detector_efficiency}"
            )
        if not (0.0 < self.dark_count_rate < 1.0):
            raise ValueError(
                f"dark_count_rate must be in (0, 1); got {self.dark_count_rate}"
            )
        if self.repetition_rate_hz <= 0.0:
            raise ValueError(
                f"repetition_rate_hz must be > 0; got {self.repetition_rate_hz}"
            )
        if self.protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Unsupported protocol {self.protocol!r}; "
                f"choose from {sorted(_SUPPORTED_PROTOCOLS)}"
            )


@dataclass(frozen=True, slots=True)
class KeyRateResult:
    """Full secret key rate calculation result for one observed QBER.

    Args:
        qber: Input QBER value.
        raw_rate_per_pulse: Secret key bits per pulse (pre-repetition-rate
            scaling); zero if below threshold.
        secret_key_rate_bps: ``raw_rate_per_pulse × repetition_rate_hz``
            [bits/second].
        is_positive: ``True`` if ``raw_rate_per_pulse > 0`` (link is secure).
        security_margin: ``qber_threshold − qber``.  Positive means safe.
        qber_threshold: QBER at which the secret key rate crosses zero.
        h2_qber: H₂(QBER) — binary entropy of the observed error rate.
        h2_e11: H₂(e11) — binary entropy of the single-photon error rate.
        info_leakage_fraction: Fraction of the sifted key consumed by error
            correction: ``f_ec · H₂(QBER) / (1 − H₂(e11))``.
    """

    qber: float
    raw_rate_per_pulse: float
    secret_key_rate_bps: float
    is_positive: bool
    security_margin: float
    qber_threshold: float
    h2_qber: float
    h2_e11: float
    info_leakage_fraction: float


class BB84KeyRateCalculator:
    """Asymptotic BB84 secret key rate estimator (GLLP / decoy-state).

    Implements the GLLP security proof result for weak coherent pulse (WCP)
    sources.  When ``params.protocol == "bb84_decoy"``, the single-photon
    yield is estimated from decoy-state analysis.

    Args:
        params: Physical and protocol parameters.

    References:
        Gottesman, Lo, Lütkenhaus, Preskill, Quant. Inf. Comput. 4, 325–360
        (2004). [GLLP]
        Ma et al., Phys. Rev. A 72, 012326 (2005). [decoy states]
    """

    def __init__(self, params: KeyRateParams) -> None:
        self._params = params

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def qber_threshold(self) -> float:
        """QBER value at which the secret key rate reaches zero.

        Solves ``1 − H₂(e11) − f_ec · H₂(Q) = 0`` for Q via binary search.
        For standard BB84 with f_ec=1.0 the threshold is ≈ 0.110; with the
        realistic f_ec=1.16 it is ≈ 0.098.

        Returns:
            QBER threshold Q* where ``_raw_rate(Q*)`` ≈ 0, in ``[0, 0.5]``.
        """
        if self._raw_rate(0.0) <= 0.0:
            return 0.0
        lo, hi = 0.0, 0.5
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if self._raw_rate(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def e11(self, qber: float) -> float:
        """Estimate the single-photon QBER e11 from the observed channel QBER.

        For ``protocol="bb84"`` (no decoy): treats all pulses as single-photon
        → e11 = qber (conservative upper bound).

        For ``protocol="bb84_decoy"``: estimates the single-photon yield Y1
        from the WCP gain model and divides:
        ``e11 = min(qber / Y1, 0.5)``
        where ``Y1 ≈ 1 − exp(−μ · η_d)`` (lower-bound yield estimate).

        Args:
            qber: Observed quantum bit error rate.

        Returns:
            Single-photon error rate in ``[0, 0.5]``.
        """
        p = self._params
        if p.protocol == "bb84_decoy":
            y1 = 1.0 - math.exp(-p.mu * p.detector_efficiency)
            if y1 < 1e-15:
                return 0.5
            return float(min(qber / y1, 0.5))
        return float(min(max(qber, 0.0), 0.5))

    def calculate(self, qber: float) -> KeyRateResult:
        """Full key rate calculation for a given observed QBER.

        Args:
            qber: Observed quantum bit error rate in ``[0, 0.5]``.

        Returns:
            ``KeyRateResult`` with all diagnostic fields populated.

        Raises:
            ValueError: If ``qber`` is outside ``[0, 0.5]``.
        """
        if not (0.0 <= qber <= 0.5):
            raise ValueError(f"QBER must be in [0, 0.5]; got {qber}")

        raw = self._raw_rate(qber)
        skr_bps = raw * self._params.repetition_rate_hz
        threshold = self.qber_threshold()
        e = self.e11(qber)
        h2q = binary_entropy(qber)
        h2e = binary_entropy(e)

        denom = max(1.0 - h2e, 1e-10)
        leakage = (self._params.f_ec * h2q) / denom

        return KeyRateResult(
            qber=qber,
            raw_rate_per_pulse=raw,
            secret_key_rate_bps=skr_bps,
            is_positive=raw > 0.0,
            security_margin=threshold - qber,
            qber_threshold=threshold,
            h2_qber=h2q,
            h2_e11=h2e,
            info_leakage_fraction=float(min(leakage, 1.0)),
        )

    def rate_vs_qber(self, n_points: int = 200) -> tuple[list[float], list[float]]:
        """Generate the (qber, rate_bps) curve from QBER 0 to 0.5.

        Useful for plotting the rate curve and marking the current operating
        point.

        Args:
            n_points: Number of points on the curve.

        Returns:
            ``(qber_list, rate_bps_list)`` each of length ``n_points``.
        """
        qbers = [i / (n_points - 1) * 0.5 for i in range(n_points)]
        rates = [self._raw_rate(q) * self._params.repetition_rate_hz for q in qbers]
        return qbers, rates

    def distance_budget(
        self,
        fiber_loss_db_per_km: float = 0.2,
        connector_loss_db: float = 1.0,
    ) -> float:
        """Maximum fiber distance at which the secret key rate remains positive.

        Models channel transmission as:
        ``η_channel(L) = η_d · 10^{−(α·L + connector_loss)/10}``

        The QBER at distance L is estimated from dark counts dominating the
        gain: ``Q(L) = 0.5 · p_dc / (p_dc + η_signal · (1 − p_dc))``.

        Args:
            fiber_loss_db_per_km: Fiber attenuation coefficient [dB/km].
            connector_loss_db: Fixed connector/splicing insertion loss [dB].

        Returns:
            Maximum reach in km; 0.0 if the rate is already zero at L=0.
        """

        def rate_at_distance(dist: float) -> float:
            total_loss_db = fiber_loss_db_per_km * dist + connector_loss_db
            eta = self._params.detector_efficiency * 10.0 ** (-total_loss_db / 10.0)
            p = self._params
            eta_signal = p.mu * eta
            if eta_signal < 1e-15:
                return 0.0
            q_loss = 0.5 * p.dark_count_rate / (
                p.dark_count_rate + eta_signal * (1.0 - p.dark_count_rate)
            )
            q_loss = min(q_loss, 0.5)
            return self._raw_rate(q_loss)

        if rate_at_distance(0.0) <= 0.0:
            return 0.0
        lo, hi = 0.0, 1000.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if rate_at_distance(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_rate(self, qber: float) -> float:
        """Secret key rate per pulse (pre-repetition-rate, clamped to 0).

        GLLP asymptotic formula:
        ``R = q · Q_μ · [(1 − H₂(e11)) − f_ec · H₂(Q)]``

        where q=0.5 is the BB84 sifting factor and Q_μ is the detection
        gain.  Returns ``max(0.0, R)`` — rate is non-negative.

        Args:
            qber: Observed quantum bit error rate.

        Returns:
            Secret key rate per pulse, clamped to zero from below.
        """
        p = self._params
        q_sift = 0.5
        if p.protocol == "bb84_decoy":
            q_mu = p.mu * p.detector_efficiency * math.exp(-p.mu)
        else:
            q_mu = 1.0 - math.exp(-p.mu * p.detector_efficiency)

        e = self.e11(qber)
        rate = q_sift * q_mu * (
            (1.0 - binary_entropy(e)) - p.f_ec * binary_entropy(qber)
        )
        return max(0.0, float(rate))
