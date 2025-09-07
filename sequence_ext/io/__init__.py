# sequence_ext/io/__init__.py
"""
I/O utilities
=============

Provides core input/output infrastructure for QUASAR simulations.

Modules
-------
- logging.py : Centralized Loguru-based logging with contextvars and stdlib bridge.
- metrics.py : MetricFrame container (physical, network, protocol, security).
- writers.py : ResultWriter for persisting metrics to disk (Parquet/CSV).

Public API
----------
- logger, reconfigure_logging, set_run_id, bind_component
- MetricFrame
- ResultWriter
"""

from .logging import logger, reconfigure_logging, set_run_id, bind_component
from .metrics import MetricFrame
from .writers import ResultWriter

__all__ = [
    "logger",
    "reconfigure_logging",
    "set_run_id",
    "bind_component",
    "MetricFrame",
    "ResultWriter",
]

