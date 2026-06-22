"""Sensitivity-matrix fitting and illustrative SMF-28 defaults (§2, §5.3).

Provides two things:
  1. ``SensitivityFitter`` — fits S from real field-calibration data (paired
     environmental state vectors and measured Pauli error rates).
  2. ``S_SMF28_DEFAULT`` / ``smf28_calibration()`` — *illustrative, uncalibrated*
     demonstration values; see the comments on ``S_SMF28_DEFAULT`` for the
     caveats.  These are NOT derived from measured fibre data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from qndt.core.context import PauliRateVector
from qndt.physics.kernels import MemoryKernel

# Default SMF-28 sensitivity matrix (§5.3), rows [px, py, pz] x cols [T, seismic, wind].
# Matches the GUI's TelemetryPanel default (qndt.gui.panels.telemetry_panel) — this
# module is the canonical, non-GUI source of truth for the illustrative default values.
#
# IMPORTANT — ILLUSTRATIVE / UNCALIBRATED DEFAULTS.
# No field measurement, literature citation, or formal derivation underpins these
# specific magnitudes.  They are physically-motivated guesses.
# The temperature channel now couples to ΔT from the 20 °C operating-point reference
# (set in EnvironmentalTelemetryEngine.env_ref), not to the absolute Celsius value.
# With the ΔT coupling the pedestal S[pz,T]×20 is removed; QBER is driven only by
# thermal drift around the setpoint.  The S magnitudes here were chosen for the
# pre-fix absolute-T regime and should be re-tuned for the ΔT regime in a future
# calibration pass (deferred per architecture decision).
# For a specific real fibre, replace S_SMF28_DEFAULT with values derived from
# measured data or use SensitivityFitter with a real CalibrationDataset.
S_SMF28_DEFAULT: np.ndarray = np.array(
    [
        [0.0, 0.001, 0.0005],
        [0.0, 0.001, 0.0],
        [0.002, 0.0, 0.0005],
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class CalibrationDataset:
    """Paired environmental and Pauli-rate measurements from field calibration.

    Args:
        env_samples: Environmental state vectors ``E`` at each calibration time.
        pauli_samples: Measured Pauli rates at the same calibration times.
        link_id: Fiber link this dataset was measured on.
        description: Free-text description of the measurement campaign.
    """

    env_samples: list[np.ndarray]
    pauli_samples: list[PauliRateVector]
    link_id: str
    description: str = field(default="")

    @property
    def n_samples(self) -> int:
        """Number of calibration samples."""
        return len(self.env_samples)

    @property
    def n_env_dims(self) -> int:
        """Dimensionality ``M`` of the environmental state vector."""
        return len(self.env_samples[0])


class SensitivityFitter:
    """Fits the sensitivity matrix S from a ``CalibrationDataset`` via least squares."""

    def _design_matrices(self, dataset: CalibrationDataset) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack(dataset.env_samples).astype(np.float64)
        Y = np.array(
            [[p.px, p.py, p.pz] for p in dataset.pauli_samples], dtype=np.float64
        )
        return X, Y

    def fit(
        self, dataset: CalibrationDataset, kernel: MemoryKernel | None = None
    ) -> np.ndarray:
        """Fit the sensitivity matrix S, shape ``(3, M)``, from ``dataset``.

        With no kernel, performs an instantaneous (unweighted) least-squares
        fit.  With a kernel, weights every sample by the kernel's peak
        coupling ``K(0)[0, 0]`` before fitting — a simplified stand-in for a
        full convolution-aware fit.

        Args:
            dataset: Calibration measurements.
            kernel: Optional memory kernel whose peak coupling weights the fit.

        Returns:
            Best-fit S matrix, values clamped to ``[-10, 10]``.
        """
        X, Y = self._design_matrices(dataset)

        if kernel is not None:
            weight = float(kernel.eval(0.0)[0, 0])
            sqrt_w = math.sqrt(max(weight, 0.0))
            X = X * sqrt_w
            Y = Y * sqrt_w

        s_t, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return np.asarray(np.clip(s_t.T, -10.0, 10.0), dtype=np.float64)

    def residuals(self, dataset: CalibrationDataset, S: np.ndarray) -> np.ndarray:
        """Return per-sample residual Pauli rate error, shape ``(n_samples, 3)``.

        Args:
            dataset: Calibration measurements.
            S: Sensitivity matrix to evaluate, shape ``(3, M)``.
        """
        X, Y = self._design_matrices(dataset)
        y_pred = X @ S.T
        return np.asarray(Y - y_pred, dtype=np.float64)

    def r_squared(self, dataset: CalibrationDataset, S: np.ndarray) -> float:
        """Return the coefficient of determination R² for ``S`` on ``dataset``.

        Args:
            dataset: Calibration measurements.
            S: Sensitivity matrix to evaluate, shape ``(3, M)``.

        Returns:
            R² in ``[0, 1]``; ``1.0`` if the dataset has zero variance.
        """
        _, Y = self._design_matrices(dataset)
        resid = self.residuals(dataset, S)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
        if ss_tot == 0.0:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)


def smf28_calibration() -> CalibrationDataset:
    """Return a synthetic ``CalibrationDataset`` for a typical SMF-28 fiber.

    .. warning::
        **Self-consistency fixture only — not a field calibration.**
        Pauli rates are computed directly from ``S_SMF28_DEFAULT`` (plus 5%
        synthetic noise), so fitting ``SensitivityFitter`` on this dataset
        merely recovers ``S_SMF28_DEFAULT`` approximately.  It does not
        constitute an independent validation of those values against real
        fibre measurements.

    Generates 10 samples spanning temperature 10-30°C, seismic acceleration
    0-0.01 m/s², and wind force 0-1.0 N, with Pauli rates derived from
    ``S_SMF28_DEFAULT`` and 5% multiplicative Gaussian noise.
    """
    rng = np.random.default_rng(42)
    temps = np.linspace(10.0, 30.0, 10)
    seismic = np.linspace(0.0, 0.01, 10)
    wind = np.linspace(0.0, 1.0, 10)

    env_samples: list[np.ndarray] = []
    pauli_samples: list[PauliRateVector] = []
    for i in range(10):
        E = np.array([temps[i], seismic[i], wind[i]], dtype=np.float64)
        rates = S_SMF28_DEFAULT @ E
        noisy = rates * (1.0 + rng.normal(0.0, 0.05, size=3))
        noisy = np.clip(noisy, 0.0, 0.49)
        env_samples.append(E)
        pauli_samples.append(
            PauliRateVector(px=float(noisy[0]), py=float(noisy[1]), pz=float(noisy[2]))
        )

    return CalibrationDataset(
        env_samples=env_samples,
        pauli_samples=pauli_samples,
        link_id="smf28_reference",
        description=(
            "Self-consistency fixture (§5.3 illustrative defaults, "
            "5% synthetic noise — not field data)."
        ),
    )
