
# sequence_ext/scenarios/__init__.py
"""
Scenarios
=========

Simulation scenarios defined for QUASAR.

Each scenario encapsulates:
- Configuration (topology, channels, devices, traffic patterns).
- Orchestration logic (how to wire SeQUeNCe components).
- Output location.

Scenarios implemented
---------------------
- scenario1_static.py : Static equidistant and uneven node placement (fiber).
- scenario2_moving.py : Free-space moving terminals (weather + pointing).
- scenario3_city.py   : Urban network overlays (routing, resilience).
- scenario4_uk_opt.py : UK-wide optimization study (hybrid channels).

Usage
-----
from sequence_ext.scenarios.scenario1_static import Scenario1Equidistant
sc = Scenario1Equidistant(pulse_rate_hz=50e6, duration_s=1.0, output_dir="data/run")
sc.run()
"""

__all__ = []
