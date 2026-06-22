"""Memory kernel implementations for non-Markovian noise convolution (§5.2).

Each kernel returns a 3×3 matrix K(τ) evaluated at time lag τ ≥ 0.
The matrix is used as: ``acc += K.eval(t - t_k) @ (S @ E_k) * dt_k``
where S is the sensitivity matrix and E is the environmental state vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_I3: np.ndarray = np.eye(3, dtype=np.float64)


def _check_tau(tau: float) -> None:
    if tau < 0.0:
        raise ValueError(f"tau must be >= 0; got {tau}")


@runtime_checkable
class MemoryKernel(Protocol):
    """Protocol for non-Markovian memory kernels K(τ).

    All implementations adopt the unit-area convention from Appendix A §3:
    ∫₀^∞ K(τ) dτ = I₃.  ``K(τ) → 0`` as ``τ → ∞``.
    ``tau`` is always a non-negative time lag in seconds.

    Returns:
        A 3×3 numpy array.
    """

    def eval(self, tau: float) -> np.ndarray:
        """Evaluate the kernel at time lag ``tau`` seconds.

        Args:
            tau: Non-negative time lag in seconds.

        Returns:
            3×3 numpy array ``K(τ)``.

        Raises:
            ValueError: If ``tau < 0``.
        """
        ...


@dataclass(frozen=True, slots=True)
class ExponentialKernel:
    """Diagonal exponential memory kernel with per-axis decay time constants.

    ``K_α(τ) = (1/τ_α) · exp(-τ/τ_α)``  for α ∈ {x, y, z}.

    Each axis integrates to 1 over [0, ∞) — unit-area convention (Appendix A §3).
    Appropriate for Markovian-limit environments where memory decays
    monotonically with a characteristic timescale.

    Args:
        tau_x: X-axis decay time constant in seconds.
        tau_y: Y-axis decay time constant in seconds.
        tau_z: Z-axis decay time constant in seconds.

    Raises:
        ValueError: If any time constant is <= 0.
    """

    tau_x: float
    tau_y: float
    tau_z: float

    def __post_init__(self) -> None:
        if self.tau_x <= 0.0 or self.tau_y <= 0.0 or self.tau_z <= 0.0:
            raise ValueError(
                f"All time constants must be > 0; "
                f"got tau_x={self.tau_x}, tau_y={self.tau_y}, tau_z={self.tau_z}"
            )

    def eval(self, tau: float) -> np.ndarray:
        """Evaluate the exponential kernel at time lag ``tau``.

        Returns ``diag((1/τx)·exp(-τ/τx), (1/τy)·exp(-τ/τy), (1/τz)·exp(-τ/τz))``.
        Each diagonal entry integrates to 1 over [0, ∞).

        Args:
            tau: Non-negative time lag in seconds.

        Returns:
            3×3 diagonal array.

        Raises:
            ValueError: If ``tau < 0``.
        """
        _check_tau(tau)
        return np.diag(
            np.array(
                [
                    np.exp(-tau / self.tau_x) / self.tau_x,
                    np.exp(-tau / self.tau_y) / self.tau_y,
                    np.exp(-tau / self.tau_z) / self.tau_z,
                ],
                dtype=np.float64,
            )
        )


@dataclass(frozen=True, slots=True)
class LorentzianKernel:
    """Oscillatory non-Markovian kernel modelling a damped mechanical resonance.

    ``K(τ) = N_L · exp(-γ·τ) · cos(ω₀·τ) · I₃``

    where ``N_L = (γ²+ω₀²)/γ`` is the unit-area normalisation constant
    derived in Appendix A §3.  The cosine factor produces sign changes that
    correspond to information backflow — a hallmark of genuine non-Markovian
    dynamics detected by the RHP witness (§5.6).

    Args:
        gamma: Damping half-width in Hz (must be > 0).
        omega_0: Centre (resonance) frequency in Hz.

    Raises:
        ValueError: If ``gamma <= 0``.
    """

    gamma: float
    omega_0: float

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0; got {self.gamma}")

    def eval(self, tau: float) -> np.ndarray:
        """Evaluate the Lorentzian kernel at time lag ``tau``.

        Returns ``N_L · exp(-γ·τ) · cos(ω₀·τ) · I₃`` where
        ``N_L = (γ²+ω₀²)/γ`` so that ∫₀^∞ K(τ) dτ = I₃.

        Args:
            tau: Non-negative time lag in seconds.

        Returns:
            3×3 array.

        Raises:
            ValueError: If ``tau < 0``.
        """
        _check_tau(tau)
        n_l = (self.gamma**2 + self.omega_0**2) / self.gamma
        scalar = n_l * float(np.exp(-self.gamma * tau) * np.cos(self.omega_0 * tau))
        return scalar * _I3


@dataclass(frozen=True, slots=True)
class GaussianKernel:
    """Gaussian memory kernel modelling correlated bath fluctuations.

    ``K(τ) = amplitude · N_G · exp(-τ²/(2σ²)) · I₃``

    where ``N_G = √(2/(π·σ²))`` is the unit-area normalisation constant
    derived in Appendix A §3.  With the default ``amplitude=1.0``,
    ∫₀^∞ K(τ) dτ = I₃.  Always non-negative — no information backflow.

    Args:
        sigma: Gaussian width in seconds (must be > 0).
        amplitude: Multiplies N_G (default 1.0 → unit-area kernel).

    Raises:
        ValueError: If ``sigma <= 0``.
    """

    sigma: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0; got {self.sigma}")

    def eval(self, tau: float) -> np.ndarray:
        """Evaluate the Gaussian kernel at time lag ``tau``.

        Returns ``amplitude · N_G · exp(-τ²/(2σ²)) · I₃`` where
        ``N_G = √(2/(π·σ²))`` so that ∫₀^∞ K(τ) dτ = amplitude · I₃.

        Args:
            tau: Non-negative time lag in seconds.

        Returns:
            3×3 array.

        Raises:
            ValueError: If ``tau < 0``.
        """
        _check_tau(tau)
        n_g = float(np.sqrt(2.0 / (np.pi * self.sigma**2)))
        scalar = self.amplitude * n_g * float(np.exp(-(tau**2) / (2.0 * self.sigma**2)))
        return scalar * _I3
