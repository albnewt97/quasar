# sequence_ext/viz/dashboard.py
"""
Dashboard
=========

Interactive Streamlit dashboard for exploring QUASAR simulation results.

Features
--------
- File selector to load Parquet/CSV results.
- Tabs for physical, protocol, and security metrics.
- Interactive plots with matplotlib or built-in Streamlit line_chart.
- Basic summary stats (mean QBER, mean secret rate, etc.).

Usage
-----
Run from repo root:
    streamlit run sequence_ext/viz/dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from ..io.metrics import MetricFrame


def load_metricframe(run_dir: Path) -> MetricFrame:
    """Load MetricFrame from a run directory containing parquet files."""
    mf = MetricFrame()
    try:
        mf.physical = pd.read_parquet(run_dir / "physical.parquet")
        mf.network = pd.read_parquet(run_dir / "network.parquet")
        mf.protocol = pd.read_parquet(run_dir / "protocol.parquet")
        mf.security = pd.read_parquet(run_dir / "security.parquet")
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
    return mf


def main():
    st.set_page_config(page_title="QUASAR Dashboard", layout="wide")
    st.title("QUASAR Simulation Dashboard")

    run_dir = st.text_input("Run directory", "data/run")
    run_path = Path(run_dir)

    if st.button("Load metrics"):
        if not run_path.exists():
            st.error(f"Directory {run_path} does not exist")
        else:
            mf = load_metricframe(run_path)

            tab1, tab2, tab3 = st.tabs(["Physical", "Protocol", "Security"])

            with tab1:
                st.subheader("Physical metrics")
                st.line_chart(mf.physical.set_index("time")[["loss_dB", "bsm_vis"]])

            with tab2:
                st.subheader("Protocol metrics")
                st.line_chart(mf.protocol.set_index("time")[["qber", "sifted_rate"]])

            with tab3:
                st.subheader("Security metrics")
                st.line_chart(mf.security.set_index("time")[["secret_rate"]])

            st.subheader("Summary stats")
            st.json(
                {
                    "mean_qber": float(mf.protocol["qber"].mean()),
                    "mean_secret_rate": float(mf.security["secret_rate"].mean()),
                }
            )


if __name__ == "__main__":
    main()
