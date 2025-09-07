# sequence_ext/physics/free_space.py
"""
Free-space optical channel model
================================

Models atmospheric free-space optical links used in QUASAR scenarios
(satellite downlinks, UAVs, ground terminals).

Effects modeled
---------------
- Geometric loss (diffraction-limited divergence).
- Atmospheric attenuation (weather presets).
- Turbulence (Fried parameter r0, scintillation index).
- Pointing error (Gaussian jitter).
- Background noise (sky radiance).

References
----------
- Andrews, L. C., & Phillips, R. L. "Laser Beam Propagation through Random Media."
- Kaushal & Kaddoum, "Optical Communication in Space: Challenges and Mitigation Techniques."
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FreeSpaceChannel:
    """Free-space optical link."""

    distance_km: float
    wavelength_nm: float = 1550.0
    tx_aperture_m: float = 0.1
    rx_aperture_m: float = 0.2
    attenuation_db_per_km: float = 0.05
    cn2: float = 1e-15  # turbulence structure parameter [m^-2/3]
    pointing_sigma_urad: float = 5.0
    sky_background_photons: float = 1e-6

    def geometric_loss_eta(self) -> float:
        """
        Geometric coupling efficiency (0–1).
        Based on Gaussian beam divergence and aperture capture.
        """
        L = self.distance_km * 1e3
        lam = self.wavelength_nm * 1e-9
        theta = 1.22 * lam / self.tx_aperture_m
        spot_radius = L * theta
        capture = np.exp(-(self.rx_aperture_m / (2 * spot_radius)) ** 2)
        return capture

    def atmospheric_loss_eta(self) -> float:
        """Exponential loss due to scattering/absorption."""
        total_db = self.attenuation_db_per_km * self.distance_km
        return 10 ** (-total_db / 10)

    def turbulence_factor(self) -> float:
        """
        Turbulence penalty via Rytov variance.
        sigma_R^2 = 1.23 * Cn^2 * k^(7/6) * L^(11/6)
        """
        L = self.distance_km * 1e3
        lam = self.wavelength_nm * 1e-9
        k = 2 * np.pi / lam
        sigma_R2 = 1.23 * self.cn2 * (k ** (7 / 6)) * (L ** (11 / 6))
        return np.exp(-sigma_R2)

    def pointing_loss_eta(self) -> float:
        """Efficiency reduction due to jitter (Gaussian)."""
        sigma_rad = self.pointing_sigma_urad * 1e-6
        return np.exp(-(sigma_rad ** 2) / 2)

    def transmission_eta(self) -> float:
        """Overall linear transmission efficiency (0–1)."""
        return (
            self.geometric_loss_eta()
            * self.atmospheric_loss_eta()
            * self.turbulence_factor()
            * self.pointing_loss_eta()
        )

    def background_noise_photons(self, filter_bw_nm: float) -> float:
        """Background photons admitted by optical filter."""
        return self.sky_background_photons * filter_bw_nm * self.distance_km


__all__ = ["FreeSpaceChannel"]
