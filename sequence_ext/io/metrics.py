# sequence_ext/io/metrics.py
"""
MetricFrame
===========

Unified in-memory container for QUASAR simulation metrics. Provides:

- Structured DataFrames for different metric categories:
  * physical   : channel and device physics (loss, visibility, dark counts, etc.)
  * network    : routing and topology usage
  * protocol   : protocol-level performance (QBER, sifted key rate, EC leakage)
  * security   : final secure key metrics (secret key rate, security parameter epsilon)

- Convenient serialization to Parquet and CSV.
- Append/merge helpers to support long runs or multiple segments.
- Validation on schema to avoid silent errors.

All tables share a `time` column in SI units (seconds) as float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: empty frames with explicit dtypes (prevents concat warnings)
# ---------------------------------------------------------------------------
def _empty_physical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="float64"),
            "loss_dB": pd.Series(dtype="float64"),
            "bsm_vis": pd.Series(dtype="float64"),
            "dark_rate": pd.Series(dtype="float64"),
        }
    )


def _empty_network() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="float64"),
            "path": pd.Series(dtype="string"),
            "util": pd.Series(dtype="float64"),
            "latency_ns": pd.Series(dtype="float64"),
        }
    )


def _empty_protocol() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="float64"),
            "qber": pd.Series(dtype="float64"),
            "sifted_rate": pd.Series(dtype="float64"),
            "ec_leak": pd.Series(dtype="float64"),
        }
    )


def _empty_security() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="float64"),
            "secret_rate": pd.Series(dtype="float64"),
            "epsilon": pd.Series(dtype="float64"),
        }
    )


def _coerce_like(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has same columns (order) and dtypes as ref."""
    # Reindex to ref columns to enforce order; missing cols become NaN then cast.
    out = df.reindex(columns=ref.columns)
    # Align dtypes (skip if incompatible string—pandas will upcast)
    for c, dt in ref.dtypes.items():
        try:
            out[c] = out[c].astype(dt, copy=False)
        except Exception:
            # If strict cast fails (e.g., None into string), let pandas handle NA-friendly type.
            pass
    return out


@dataclass
class MetricFrame:
    """Container for QUASAR metrics across all abstraction layers."""

    physical: pd.DataFrame = field(default_factory=_empty_physical)
    network: pd.DataFrame = field(default_factory=_empty_network)
    protocol: pd.DataFrame = field(default_factory=_empty_protocol)
    security: pd.DataFrame = field(default_factory=_empty_security)

    # -------------------------------------------------------------------------
    # Append / merge helpers (warning-free and dtype-stable)
    # -------------------------------------------------------------------------
    def append(
        self,
        *,
        physical: pd.DataFrame | None = None,
        network: pd.DataFrame | None = None,
        protocol: pd.DataFrame | None = None,
        security: pd.DataFrame | None = None,
    ) -> None:
        """Append new rows to existing frames, preserving schema and dtypes."""
        if physical is not None and not physical.empty:
            src = _coerce_like(physical, self.physical)
            if self.physical.empty:
                self.physical = src.reset_index(drop=True)
            else:
                self.physical = pd.concat([self.physical, src], ignore_index=True, copy=False)

        if network is not None and not network.empty:
            src = _coerce_like(network, self.network)
            if self.network.empty:
                self.network = src.reset_index(drop=True)
            else:
                self.network = pd.concat([self.network, src], ignore_index=True, copy=False)

        if protocol is not None and not protocol.empty:
            src = _coerce_like(protocol, self.protocol)
            if self.protocol.empty:
                self.protocol = src.reset_index(drop=True)
            else:
                self.protocol = pd.concat([self.protocol, src], ignore_index=True, copy=False)

        if security is not None and not security.empty:
            src = _coerce_like(security, self.security)
            if self.security.empty:
                self.security = src.reset_index(drop=True)
            else:
                self.security = pd.concat([self.security, src], ignore_index=True, copy=False)

    def merge(self, others: Iterable["MetricFrame"]) -> "MetricFrame":
        """Return a new MetricFrame merged with a list of others."""
        out = MetricFrame()
        frames = [self, *others]
        # Use the safe append semantics by concatenating after coercion
        for f in frames:
            out.append(
                physical=f.physical,
                network=f.network,
                protocol=f.protocol,
                security=f.security,
            )
        return out

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    def to_parquet(self, out_dir: Path, *, overwrite: bool = True) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write(self.physical, out_dir / "physical.parquet", "parquet", overwrite)
        self._write(self.network, out_dir / "network.parquet", "parquet", overwrite)
        self._write(self.protocol, out_dir / "protocol.parquet", "parquet", overwrite)
        self._write(self.security, out_dir / "security.parquet", "parquet", overwrite)

    def to_csv(self, out_dir: Path, *, overwrite: bool = True) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write(self.physical, out_dir / "physical.csv", "csv", overwrite)
        self._write(self.network, out_dir / "network.csv", "csv", overwrite)
        self._write(self.protocol, out_dir / "protocol.csv", "csv", overwrite)
        self._write(self.security, out_dir / "security.csv", "csv", overwrite)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _write(df: pd.DataFrame, path: Path, fmt: Literal["csv", "parquet"], overwrite: bool) -> None:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists and overwrite=False")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unknown format {fmt}")
