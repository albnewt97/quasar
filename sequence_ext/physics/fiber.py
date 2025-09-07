# sequence_ext/physics/fiber.py
"""
Fiber channel model
===================

Models optical fiber links with attenuation, dispersion, PMD, and Raman noise.

References
----------
- Agrawal, G. P. "Fiber-Optic Communication Systems." Wiley, 2012.
- Eraerds et al., "Quantum key distribution and 1 Gbit/s data encryption over a single fibre,"
  New Journal of Physics, 2010.

Conventions
-----------
- Length in kilometers (km).
- Attenuation in dB/km.
- Dispersion parameter D in ps/(nm·km).
- PMD coefficient in ps/sqrt(km).
- Raman noise approximated as broadband background photons per nm.

This model is simplified and intended for system-level simulation. Detailed
nonlinear effects are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FiberChannel:
    """Optical fiber channel model."""

    length_km: float
    attenuation_db_per_km: float = 0.2
    dispersion_ps_nm_km: float = 17.0
    pmd_ps_sqrt_km: float = 0.1
    raman_photons_per_nm: float = 1e-9

    def transmission_loss_db(self) -> float:
        """Total link loss (dB)."""
        return self.attenuation_db_per_km * self.length_km

    def transmission_eta(self) -> float:
        """Linear transmission factor (0–1)."""
        return 10 ** (-self.transmission_loss_db() / 10)

    def dispersion_broadening_ps(self, wavelength_nm: float, bandwidth_nm: float) -> float:
        """
        Chromatic dispersion broadening [ps].

        Parameters
        ----------
        wavelength_nm : float
            Center wavelength.
        bandwidth_nm : float
            Source spectral width.
        """
        return self.dispersion_ps_nm_km * self.length_km * bandwidth_nm

    def pmd_broadening_ps(self) -> float:
        """Polarization-mode dispersion RMS broadening [ps]."""
        return self.pmd_ps_sqrt_km * np.sqrt(self.length_km)

    def raman_noise_photons(self, filter_bw_nm: float) -> float:
        """Estimate number of Raman noise photons passed by filter."""
        return self.raman_photons_per_nm * filter_bw_nm * self.length_km


__all__ = ["FiberChannel"]
