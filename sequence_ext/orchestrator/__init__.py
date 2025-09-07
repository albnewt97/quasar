# sequence_ext/orchestrator/__init__.py
"""
Orchestrator package
====================

Coordinates QUASAR simulation execution.

Modules
-------
- events.py       : Immutable dataclasses for detector and BSM events.
- orchestrator.py : Main Orchestrator class (run simulations, collect metrics).

Public API
----------
The orchestrator layer provides:
- `BSMEvent`, `DetectorEvent`: canonical event types.
- `Orchestrator`: central runner that integrates scenarios, physics, and I/O.
"""

from .events import BSMEvent, DetectorEvent
from .orchestrator import Orchestrator

__all__ = ["BSMEvent", "DetectorEvent", "Orchestrator"]

