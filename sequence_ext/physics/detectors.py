# sequence_ext/physics/detectors.py
"""
Detector models
===============

Models single-photon detectors (SPDs) used in MDI-QKD.

Effects modeled
---------------
- Quantum efficiency (detection probability).
- Dark counts per gate.
- Dead time (after a click).
- Afterpulsing probability.

References
----------
- Hadfield, R. H. "Single-photon detectors for optical quantum information applications."
  Nature Photonics, 2009.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Detector:
    """Single-photon detector model."""

    eta: float = 0.8
    dark_per_gate: float = 1e-6
    dead_time_ns: float = 60.0
    afterpulse_prob: float = 0.02

    _last_click_time: float = -np.inf

    def detect(self, photons: int, time_ns: float | None = None) -> bool:
        """
        Simulate a detection attempt.

        Parameters
        ----------
        photons : int
            Number of incoming photons.
        time_ns : float | None
            Absolute arrival time in ns (to apply dead time).
        """
        # Dead time check
        if time_ns is not None and (time_ns - self._last_click_time) < self.dead_time_ns:
            return False

        # Photon detection
        prob_click = 1 - (1 - self.eta) ** photons

        # Dark count
        if np.random.rand() < self.dark_per_gate:
            prob_click = 1.0

        # Draw event
        click = np.random.rand() < prob_click
        if click and time_ns is not None:
            self._last_click_time = time_ns

        # Afterpulse
        if click and np.random.rand() < self.afterpulse_prob:
            return True

        return click


__all__ = ["Detector"]
