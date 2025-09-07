# dashboard/app.py
"""
QUASAR Dashboard (Streamlit)
============================

Launch:
    streamlit run dashboard/app.py -- --run-dir data/runs/sc1_equ

Features
--------
- Sidebar: choose a run directory; auto-detect Parquet/CSV artifacts.
- KPIs: mean QBER, mean secret rate, time span.
- Tabs: Physical, Protocol, Security, Network sample.
- Export: download CSV/Parquet tables directly from the UI.

Notes
-----
- This is a thin UI over the on-disk artifacts written by scenarios/orchestrator.
- The app is intentionally dependency-light (pandas, streamlit, matplotlib optional).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# CLI passthrough (supports `--run-dir` when launching Streamlit)
# -----------------------------------------------------------------------------
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", type=str, default="data/runs/sc1_equ")
    # Streamlit injects its own args; ignore unknowns
    args, _ = parser.parse_known_args()
    return args


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def _read_table(run_dir: Path, stem: str) -> pd.DataFrame:
    pq = run_dir / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = run_dir / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {stem}.parquet/.csv in {run_dir}")

@st.cache_data(show_spinner=False)
def load_metrics(run_dir: str) -> Dict[str, pd.DataFrame]:
    p = Path(run_dir)
    return {
        "physical": _read_table(p, "physical"),
        "protocol": _read_table(p, "protocol"),
        "security": _read_table(p, "security"),
        "network": _read_table(p, "network"),
    }


def _kpi(label: str, value, help_text: Optional[str] = None):
    st.metric(label, value=None if value is None else (f"{value:,}" if isinstance(value, (int, float)) else value), help=help_text)
    # Streamlit's metric expects numerical deltas; we only show the main value via st.write next to it.

def _fmt_float(x: float, digits: int = 4) -> str:
    try:
        return f"{x:.{digits}g}"
    except Exception:
        return "-"


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main():
    args = _parse_cli_args()

    st.set_page_config(page_title="QUASAR Dashboard", layout="wide")
    st.title("QUASAR – Simulation Dashboard")

    with st.sidebar:
        st.header("Data Source")
        run_dir = st.text_input("Run directory", value=args.run_dir)
        refresh = st.button("Load / Refresh", type="primary")
        st.caption("Provide a directory containing QUASAR artifacts: "
                   "`physical.*`, `protocol.*`, `security.*`, `network.*`")

    if refresh or "metrics" not in st.session_state or st.session_state.get("run_dir") != run_dir:
        try:
            metrics = load_metrics(run_dir)
            st.session_state["metrics"] = metrics
            st.session_state["run_dir"] = run_dir
            st.success(f"Loaded metrics from: {run_dir}")
        except Exception as e:
            st.error(f"Failed to load metrics from '{run_dir}': {e}")
            return

    metrics = st.session_state.get("metrics")
    if not metrics:
        st.info("No data loaded yet. Choose a valid run directory and click **Load / Refresh**.")
        return

    phys = metrics["physical"]
    prot = metrics["protocol"]
    sec = metrics["security"]
    net = metrics["network"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        span = (phys["time"].max() - phys["time"].min()) if len(phys) else 0.0
        st.subheader("Time Span")
        st.write(f"{_fmt_float(span)} s")
    with c2:
        st.subheader("Mean QBER")
        st.write(_fmt_float(float(prot["qber"].mean()) if "qber" in prot.columns else float("nan")))
    with c3:
        st.subheader("Mean Secret Rate")
        st.write(_fmt_float(float(sec["secret_rate"].mean()) if "secret_rate" in sec.columns else float("nan")))
    with c4:
        st.subheader("Samples")
        st.write(len(phys))

    # Tabs
    t1, t2, t3, t4 = st.tabs(["Physical", "Protocol", "Security", "Network"])

    with t1:
        st.subheader("Physical Metrics")
        sel_cols = [c for c in ["loss_dB", "bsm_vis", "dark_rate"] if c in phys.columns]
        if sel_cols:
            st.line_chart(phys.set_index("time")[sel_cols])
        st.caption("Download")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("Physical (CSV)", data=phys.to_csv(index=False), file_name="physical.csv", mime="text/csv")
        with c_dl2:
            try:
                data = phys.to_parquet(index=False)
                st.download_button("Physical (Parquet)", data=data, file_name="physical.parquet", mime="application/octet-stream")
            except Exception:
                st.caption("Parquet download unavailable in this environment.")

    with t2:
        st.subheader("Protocol Metrics")
        sel_cols = [c for c in ["qber", "sifted_rate", "ec_leak"] if c in prot.columns]
        if sel_cols:
            st.line_chart(prot.set_index("time")[sel_cols])
        st.caption("Download")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("Protocol (CSV)", data=prot.to_csv(index=False), file_name="protocol.csv", mime="text/csv")
        with c_dl2:
            try:
                data = prot.to_parquet(index=False)
                st.download_button("Protocol (Parquet)", data=data, file_name="protocol.parquet", mime="application/octet-stream")
            except Exception:
                st.caption("Parquet download unavailable in this environment.")

    with t3:
        st.subheader("Security Metrics")
        sel_cols = [c for c in ["secret_rate", "epsilon"] if c in sec.columns]
        if sel_cols:
            st.line_chart(sec.set_index("time")[sel_cols])
        st.caption("Download")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("Security (CSV)", data=sec.to_csv(index=False), file_name="security.csv", mime="text/csv")
        with c_dl2:
            try:
                data = sec.to_parquet(index=False)
                st.download_button("Security (Parquet)", data=data, file_name="security.parquet", mime="application/octet-stream")
            except Exception:
                st.caption("Parquet download unavailable in this environment.")

    with t4:
        st.subheader("Network (Sample)")
        show_cols = [c for c in net.columns if c != "time"]
        if not show_cols:
            st.write("No additional network columns to display.")
        else:
            st.dataframe(net[["time"] + show_cols].head(100))

    st.divider()
    st.caption(f"Run directory: `{run_dir}`")


if __name__ == "__main__":
    main()
