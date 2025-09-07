import pathlib
import sys
import json
import streamlit as st

# Allow importing local packages when running `streamlit run dashboard/app.py`
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sequence_ext.viz.plots import plot_basic_kpis

st.set_page_config(page_title="QUASAR Dashboard", layout="wide")
st.title("QUASAR – MDI-QKD Simulation Dashboard")

st.sidebar.header("Scenario 1a: Static Equidistant Fiber (MVP)")
distance_km = st.sidebar.number_input("Distance (A–C = B–C, km)", min_value=1.0, max_value=200.0, value=10.0)
click_rate = st.sidebar.number_input("Synthetic raw key rate (bps)", min_value=0.0, max_value=1e6, value=0.0)

run = st.sidebar.button("Run Simulation")

placeholder = st.empty()

if run:
# Placeholder: pretend to run and show a figure from synthetic KPIs
kpis = {"protocol": {"raw_key_rate_bps": click_rate}, "security": {"privacy_throughput_bps": click_rate * 0.8}}
fig = plot_basic_kpis(kpis)
meta = {"scenario": {"id": "1a"}, "distance_km": distance_km}

with placeholder.container():
st.subheader("Topology")
st.write(f"Equidistant fiber: A–C = B–C = {distance_km} km")
st.subheader("KPIs")
st.plotly_chart(fig, use_container_width=True)
st.subheader("Metadata")
st.code(json.dumps(meta, indent=2))
else:
st.info("Configure parameters on the left and click 'Run Simulation'.")
