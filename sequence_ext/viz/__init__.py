# sequence_ext/viz/__init__.py
"""
Visualization utilities
=======================

Static plotting and interactive dashboards for QUASAR simulation results.

Modules
-------
- plots.py     : Matplotlib plots for physical, protocol, and security metrics.
- dashboard.py : Streamlit dashboard for interactive exploration.

Public API
----------
- plot_physical
- plot_protocol
- plot_security
- (dashboard launched via `streamlit run sequence_ext/viz/dashboard.py`)
"""

from .plots import plot_physical, plot_protocol, plot_security

__all__ = [
    "plot_physical",
    "plot_protocol",
    "plot_security",
]

