# sequence_ext/orchestrator/events.py
"""
Event definitions
=================

Lightweight, immutable dataclasses representing discrete events within the
simulation. These are intended as the canonical types passed from SeQUeNCe
callbacks (or mock generators) into the Orchestrator and downstream metrics
pipeline.

Design notes
------------
- Frozen dataclasses: safe to hash, compare, and log.
- Use explicit types (ints for timestamps, floats for analog values, bools
  for clicks, etc.).
- Timestamps are expressed in integer nanoseconds (ns) for precision.
- These event types should remain minimal: only fields that are universally
  understood. Scenario-specific metadata can be attached via extra dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True, slots=True)
class BSMEvent:
    """Bell-state measurement event."""

    time_ns: int
    coincident: bool
    visibility: float
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class DetectorEvent:
    """Detector click event."""

    time_ns: int
    click: bool
    dark: bool
    detector_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


__all__ = ["BSMEvent", "DetectorEvent"]
