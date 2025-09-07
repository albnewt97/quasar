# dashboard/components.py
"""
Reusable Streamlit components for the QUASAR dashboard
======================================================

This module provides small, composable UI building blocks used by
`dashboard/app.py`. They are intentionally dependency-light and UI-agnostic
(where possible) so they can be reused in notebooks or other apps.

Components
----------
- file_selector         : Sidebar selector for a run directory.
- load_tables           : Load Parquet/CSV artifacts into DataFrames.
- kpi_cards             : Display core KPIs in a neat row.
- line_panel            : Convenience wrapper for time-series line charts.
- download_section      : CSV/Parquet download buttons for a table.
- summary_table         : Render small summary DataFrames with consistent style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _read_table(run_dir: Path, stem: str) -> pd.DataFrame:
    pq = run_dir / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = run_dir / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet/.csv in {run_dir}")


@st.cache_data(show_spinner=False)
def load_tables(run_dir: str) -> Dict[str, pd.DataFrame]:
    """Load standard QUASAR artifacts from a run directory."""
    p = Path(run_dir)
    return {
        "physical": _read_table(p, "physical"),
        "protocol": _read_table(p, "protocol"),
        "security": _read_table(p, "security"),
        "network": _read_table(p, "network"),
    }


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------
def file_selector(
    *,
    label: str = "Run directory",
    default: str = "data/runs/sc1_equ",
    help_text: str = "Directory with physical.*, protocol.*, security.*, network.*",
) -> Tuple[str, bool]:
    """Render a run directory text input and a Load/Refresh button."""
    run_dir = st.text_input(label, value=default, help=help_text)
    refresh = st.button("Load / Refresh", type="primary")
    return run_dir, refresh


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _fmt_float(x: float, digits: int = 4) -> str:
    try:
        return f"{x:.{digits}g}"
    except Exception:
        return "-"


def kpi_cards(*, phys: pd.DataFrame, prot: pd.DataFrame, sec: pd.DataFrame) -> None:
    """Display core KPIs in a row of cards."""
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        span = (phys["time"].max() - phys["time"].min()) if len(phys) else 0.0
        st.subheader("Time Span")
        st.write(f"{_fmt_float(float(span))} s")

    with c2:
        val = float(prot["qber"].mean()) if "qber" in prot.columns and len(prot) else float("nan")
        st.subheader("Mean QBER")
        st.write(_fmt_float(val))

    with c3:
        val = float(sec["secret_rate"].mean()) if "secret_rate" in sec.columns and len(sec) else float("nan")
        st.subheader("Mean Secret Rate")
        st.write(_fmt_float(val))

    with c4:
        st.subheader("Samples")
        st.write(len(phys))


def line_panel(
    *,
    df: pd.DataFrame,
    time_col: str = "time",
    value_cols: Sequence[str],
    title: Optional[str] = None,
    caption: Optional[str] = None,
) -> None:
    """Render a simple line chart for the given columns vs time."""
    if title:
        st.subheader(title)
    cols = [c for c in value_cols if c in df.columns and c != time_col]
    if not cols:
        st.info("No matching columns to plot.")
        return
    st.line_chart(df.set_index(time_col)[cols])
    if caption:
        st.caption(caption)


def download_section(df: pd.DataFrame, *, stem: str) -> None:
    """Offer CSV/Parquet downloads for a DataFrame."""
    st.caption("Download")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label=f"{stem} (CSV)",
            data=df.to_csv(index=False),
            file_name=f"{stem}.csv",
            mime="text/csv",
        )
    with c2:
        try:
            data = df.to_parquet(index=False)
            st.download_button(
                label=f"{stem} (Parquet)",
                data=data,
                file_name=f"{stem}.parquet",
                mime="application/octet-stream",
            )
        except Exception:
            st.caption("Parquet download unavailable in this environment.")


def summary_table(df: pd.DataFrame, *, title: str, max_rows: int = 100) -> None:
    """Render a compact summary table with a title."""
    st.subheader(title)
    st.dataframe(df.head(max_rows))
