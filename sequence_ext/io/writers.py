# sequence_ext/io/writers.py
"""
ResultWriter
============

Responsible for persisting MetricFrame results to disk in different formats
(CSV, Parquet). Provides a clean abstraction so scenarios/orchestrator do not
have to deal with file formats directly.

Features
--------
- Supports parquet (preferred for performance) and csv (for portability).
- Creates directories automatically.
- Validates format option strictly.
- Allows extension: users may subclass to implement HDF5, Arrow Flight, DB sinks.

Usage
-----
from pathlib import Path
from sequence_ext.io.metrics import MetricFrame
from sequence_ext.io.writers import ResultWriter

mf = MetricFrame()
# populate mf...
writer = ResultWriter(Path("data/run1"), fmt="parquet")
writer.write(mf)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .metrics import MetricFrame


class ResultWriter:
    """Persist simulation results to disk."""

    def __init__(self, out_dir: Path, fmt: Literal["parquet", "csv"] = "parquet") -> None:
        self.out_dir = out_dir
        if fmt not in ("parquet", "csv"):
            raise ValueError(f"Unsupported format: {fmt}")
        self.fmt = fmt

    def write(self, mf: MetricFrame, *, overwrite: bool = True) -> None:
        """Write MetricFrame to self.out_dir in chosen format."""
        if self.fmt == "parquet":
            mf.to_parquet(self.out_dir, overwrite=overwrite)
        elif self.fmt == "csv":
            mf.to_csv(self.out_dir, overwrite=overwrite)
        else:
            raise RuntimeError(f"Unexpected format {self.fmt!r}")

    def __repr__(self) -> str:
        return f"<ResultWriter dir={self.out_dir} fmt={self.fmt}>"
