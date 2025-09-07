# sequence_ext/physics/__init__.py
"""
Physics Models
==============

Contains physical-layer models used in QUASAR simulations.

Modules
-------
- fiber.py       : Optical fiber channel model (attenuation, dispersion, PMD, Raman noise).
- free_space.py  : Free-space optical channel (turbulence, weather, pointing errors).
- detectors.py   : Single-photon detector models (efficiency, dark counts, afterpulsing).
- profiles.py    : Convenience profiles (presets for weather, devices, etc.).

Design goals
------------
- Deterministic, vectorized computations (NumPy / SciPy).
- Side-effect free: functions/classes do not modify global state.
- Composable: scenarios can mix fiber + free-space + device models easily.
- Config-driven: parameters passed in from pydantic config models or YAML.

Example
-------
from sequence_ext.physics.fiber import FiberChannel
fiber = FiberChannel(length_km=50, attenuation_db_per_km=0.2)
loss_db = fiber.transmission_loss_db()

from sequence_ext.physics.detectors import Detector
d = Detector(eta=0.8, dark_per_gate=1e-6)
clicks = d.detect(photons=1000)
"""

__all__ = []
