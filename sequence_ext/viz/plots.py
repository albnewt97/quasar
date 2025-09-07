# sequence_ext/viz/plots.py
"""
Visualization utilities
=======================

Provides common plotting functions for QUASAR metrics.

Backends
--------
- Matplotlib for static figures (PNG, PDF).
- Functions take a MetricFrame and plot physical, protocol, or security metrics.

Design goals
------------
- Non-interactive by default; saves to file or returns figure.
- Minimal styling for publication-quality output.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from ..io.metrics import MetricFrame


def plot_physical(mf: MetricFrame, out_path: Path | None = None):
    fig, ax = plt.subplots()
    ax.plot(mf.physical["time"], mf.physical["loss_dB"], label="Loss [dB]")
    ax2 = ax.twinx()
    ax2.plot(mf.physical["time"], mf.physical["bsm_vis"], color="tab:red", label="BSM vis.")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loss [dB]")
    ax2.set_ylabel("BSM visibility")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def plot_protocol(mf: MetricFrame, out_path: Path | None = None):
    fig, ax = plt.subplots()
    ax.plot(mf.protocol["time"], mf.protocol["qber"], label="QBER")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("QBER")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def plot_security(mf: MetricFrame, out_path: Path | None = None):
    fig, ax = plt.subplots()
    ax.plot(mf.security["time"], mf.security["secret_rate"], label="Secret key rate")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Secret key rate [bits/s]")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    return fig


__all__ = ["plot_physical", "plot_protocol", "plot_security"]
