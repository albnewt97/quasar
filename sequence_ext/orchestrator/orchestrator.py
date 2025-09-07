# sequence_ext/orchestrator/orchestrator.py
"""
Orchestrator (streaming & memory-safe)
=====================================

This version processes the simulation in CHUNKS and aggregates into BINNED
metrics on-the-fly, so memory usage is ~O(num_bins) instead of O(rate*duration).

Key params
----------
- pulse_rate_hz : input pulse rate
- duration_s    : total duration
- output_dir    : where results will be written
- bin_sec       : width of aggregation bin in seconds (e.g., 1e-3 = 1 ms)
- chunk_sec     : how many seconds to process per streaming chunk (e.g., 0.1 s)

Behavior
--------
For each chunk:
  1) Generate mock physical/protocol series for that time slice.
  2) Aggregate within that slice into fixed bin_sec buckets (mean/sum).
  3) Append aggregated rows to MetricFrame tables.
Finally returns a MetricFrame that’s already small.

Tip
---
Set bin_sec so that (duration / bin_sec) is modest (e.g., 2 s / 1 ms = 2,000 rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..io.logging import logger
from ..io.metrics import MetricFrame
from ..io.writers import ResultWriter


def _aggregate_timeslice(
    t0: float,
    rate: int,
    chunk_sec: float,
    bin_sec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a mock timeslice [t0, t0+chunk_sec) and aggregate into bins.

    Returns four *aggregated* DataFrames for physical, network, protocol, security.
    """
    n = int(round(rate * chunk_sec))
    if n <= 0:
        # empty slice
        return (
            pd.DataFrame(columns=["time", "loss_dB", "bsm_vis", "dark_rate"]),
            pd.DataFrame(columns=["time", "path", "util", "latency_ns"]),
            pd.DataFrame(columns=["time", "qber", "sifted_rate", "ec_leak"]),
            pd.DataFrame(columns=["time", "secret_rate", "epsilon"]),
        )

    # Local times for this chunk
    # NOTE: we keep it float32 to cut transient memory; precision is fine for binning
    t = t0 + (np.arange(n, dtype=np.float32) / float(rate))

    # ---- Mock models (replace with SeQUeNCe hooks as needed) -----------------
    loss_dB = 15.0 + 0.2 * np.sin(2 * np.pi * t)
    bsm_vis = 0.96 + 0.02 * np.cos(2 * np.pi * t)
    dark_rate = np.full_like(t, 1e-6, dtype=np.float32)

    qber = 0.02 + 0.01 * (1.0 - bsm_vis)
    sifted = np.maximum(0.0, 5e6 * (1.0 - qber))
    ec_leak = 0.1 * sifted
    secret = np.maximum(0.0, sifted * (1.0 - 1.2 * qber) - ec_leak)
    epsilon = np.full_like(t, 1e-10, dtype=np.float32)

    # ---- Build raw per-sample frames for this slice --------------------------
    # (kept local; dropped immediately after grouping)
    phys = pd.DataFrame(
        {"time": t.astype(np.float64), "loss_dB": loss_dB, "bsm_vis": bsm_vis, "dark_rate": dark_rate}
    )
    net = pd.DataFrame(
        {"time": t.astype(np.float64), "path": "A->C<-B", "util": 0.5, "latency_ns": 1000}
    )
    prot = pd.DataFrame(
        {"time": t.astype(np.float64), "qber": qber, "sifted_rate": sifted, "ec_leak": ec_leak}
    )
    sec = pd.DataFrame(
        {"time": t.astype(np.float64), "secret_rate": secret, "epsilon": epsilon}
    )

    # ---- Aggregate into bin_sec buckets --------------------------------------
    # Compute bin edges by flooring to nearest bin_sec
    for df in (phys, net, prot, sec):
        df["bin"] = (np.floor((df["time"] - t0) / bin_sec) * bin_sec + t0).astype(np.float64)

    # Aggregations: mean for continuous metrics; first for path; mean for util/latency
    phys_agg = (
        phys.groupby("bin", as_index=False)[["loss_dB", "bsm_vis", "dark_rate"]]
        .mean()
        .rename(columns={"bin": "time"})
    )
    net_agg = (
        net.groupby("bin", as_index=False)[["util", "latency_ns"]]
        .mean()
        .assign(path="A->C<-B")
        .rename(columns={"bin": "time"})
    )
    prot_agg = (
        prot.groupby("bin", as_index=False)[["qber", "sifted_rate", "ec_leak"]]
        .mean()
        .rename(columns={"bin": "time"})
    )
    sec_agg = (
        sec.groupby("bin", as_index=False)[["secret_rate", "epsilon"]]
        .mean()
        .rename(columns={"bin": "time"})
    )

    return phys_agg, net_agg, prot_agg, sec_agg


@dataclass
class Orchestrator:
    """Streaming, memory-safe orchestration engine."""

    pulse_rate_hz: int
    duration_s: float
    output_dir: Path
    bin_sec: float = 1e-3     # aggregate to 1 ms bins by default
    chunk_sec: float = 0.1    # generate 0.1 s of raw samples per iteration

    def run(self) -> MetricFrame:
        """
        Run a simulation job in streaming chunks and return aggregated metrics.
        """
        logger.info(
            "Starting orchestration (streaming): rate={} Hz, duration={} s, bin={} s, chunk={} s",
            self.pulse_rate_hz, self.duration_s, self.bin_sec, self.chunk_sec,
        )

        mf = MetricFrame()
        t0 = 0.0
        remaining = float(self.duration_s)

        # Pre-allocate nothing; append aggregated slices per chunk
        while remaining > 0.0:
            cur = min(self.chunk_sec, remaining)
            phys, net, prot, sec = _aggregate_timeslice(
                t0=t0, rate=self.pulse_rate_hz, chunk_sec=cur, bin_sec=self.bin_sec
            )
            mf.append(physical=phys, network=net, protocol=prot, security=sec)

            t0 += cur
            remaining -= cur

        # Ensure time is monotonic across concatenated chunks
        for df in (mf.physical, mf.network, mf.protocol, mf.security):
            df.sort_values("time", inplace=True, kind="mergesort", ignore_index=True)

        n_rows = len(mf.physical)
        logger.info("Finished orchestration; produced {} aggregated rows (bin={} s)", n_rows, self.bin_sec)
        return mf

    def write(self, mf: MetricFrame, fmt: str = "parquet") -> Path:
        """Persist metrics to disk."""
        writer = ResultWriter(self.output_dir, fmt=fmt)
        writer.write(mf)
        logger.info("Wrote metrics to {} (fmt={})", self.output_dir, fmt)
        return self.output_dir
