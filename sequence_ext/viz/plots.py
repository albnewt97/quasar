from typing import Dict, Any
import plotly.graph_objects as go


def plot_basic_kpis(kpis: Dict[str, Any]) -> go.Figure:
y = [kpis.get("protocol", {}).get("raw_key_rate_bps", 0), kpis.get("security", {}).get("privacy_throughput_bps", 0)]
fig = go.Figure(data=[go.Bar(x=["raw_key_rate", "privacy_throughput"], y=y)])
fig.update_layout(title="Key Rates", xaxis_title="Metric", yaxis_title="bits/s")
return fig
