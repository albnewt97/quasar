#!/usr/bin/env python3
# scripts/report.py
"""
Generate an HTML report for a QUASAR run
=======================================

Creates a self-contained HTML report (with embedded PNG plots) summarizing a
simulation run located at a directory containing QUASAR metric artifacts:

- physical.parquet (or .csv)
- network.parquet  (or .csv)
- protocol.parquet (or .csv)
- security.parquet (or .csv)

Usage
-----
python scripts/report.py --run-dir data/runs/sc1_equ --out report.html

Options
-------
--title            Report title (default: derived from run directory)
--embed-images     Embed PNGs as base64 in the HTML (default: true)
--plots-dir        Where to save plot PNGs (default: <run-dir>/plots)
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sequence_ext.io.logging import logger, reconfigure_logging


@dataclass
class RunMetrics:
    physical: pd.DataFrame
    network: pd.DataFrame
    protocol: pd.DataFrame
    security: pd.DataFrame


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------
def _read_table(run_dir: Path, stem: str) -> pd.DataFrame:
    """
    Read a metrics table from <run_dir>/<stem>.parquet (preferred) or .csv.
    """
    pq = run_dir / f"{stem}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = run_dir / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing metrics file: {pq.name} or {csv.name} in {run_dir}")


def load_metrics(run_dir: Path) -> RunMetrics:
    physical = _read_table(run_dir, "physical")
    network = _read_table(run_dir, "network")
    protocol = _read_table(run_dir, "protocol")
    security = _read_table(run_dir, "security")
    # Basic sanity
    for name, df in [
        ("physical", physical),
        ("network", network),
        ("protocol", protocol),
        ("security", security),
    ]:
        if "time" not in df.columns:
            raise ValueError(f"{name} table lacks required 'time' column")
    return RunMetrics(physical, network, protocol, security)


# -----------------------------------------------------------------------------
# Plotting (matplotlib; no seaborn; single-axes figures)
# -----------------------------------------------------------------------------
def _plot_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_physical(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["loss_dB"], label="Loss [dB]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loss [dB]")
    ax2 = ax.twinx()
    ax2.plot(df["time"], df["bsm_vis"], label="BSM visibility")
    ax2.set_ylabel("BSM visibility")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Physical Metrics")
    return _plot_png(fig)


def plot_protocol(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["qber"], label="QBER")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("QBER")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Protocol Metrics")
    return _plot_png(fig)


def plot_security(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["secret_rate"], label="Secret key rate")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Secret key rate [bits/s]")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Security Metrics")
    return _plot_png(fig)


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def compute_summary(m: RunMetrics) -> pd.DataFrame:
    """
    Compute a compact summary (single-row) with key statistics.
    """
    def _safe_mean(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if col in df.columns and not df.empty else float("nan")

    def _safe_min(df: pd.DataFrame, col: str) -> float:
        return float(df[col].min()) if col in df.columns and not df.empty else float("nan")

    def _safe_max(df: pd.DataFrame, col: str) -> float:
        return float(df[col].max()) if col in df.columns and not df.empty else float("nan")

    row = {
        "time_span_s": float((m.physical["time"].max() - m.physical["time"].min()) if not m.physical.empty else 0.0),
        "loss_dB_mean": _safe_mean(m.physical, "loss_dB"),
        "bsm_vis_mean": _safe_mean(m.physical, "bsm_vis"),
        "qber_mean": _safe_mean(m.protocol, "qber"),
        "sifted_rate_mean": _safe_mean(m.protocol, "sifted_rate"),
        "secret_rate_mean": _safe_mean(m.security, "secret_rate"),
        "secret_rate_min": _safe_min(m.security, "secret_rate"),
        "secret_rate_max": _safe_max(m.security, "secret_rate"),
    }
    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# HTML generation
# -----------------------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _table_html(df: pd.DataFrame, title: str) -> str:
    # Use pandas to_html with sensible defaults
    html = df.to_html(index=False, border=0, classes="tbl")
    return f"""
<section>
  <h2>{title}</h2>
  {html}
</section>
"""


def _img_html(png_bytes: bytes, caption: str, embed: bool, save_path: Optional[Path]) -> str:
    if not embed and save_path:
        save_path.write_bytes(png_bytes)
        src = save_path.name
    else:
        src = f"data:image/png;base64,{_b64(png_bytes)}"
    return f"""
<figure>
  <img src="{src}" alt="{caption}" />
  <figcaption>{caption}</figcaption>
</figure>
"""


def build_report_html(
    run_dir: Path,
    metrics: RunMetrics,
    title: str,
    embed_images: bool,
    plots_dir: Path,
) -> str:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    phys_png = plot_physical(metrics.physical)
    prot_png = plot_protocol(metrics.protocol)
    sec_png  = plot_security(metrics.security)

    # Summary
    summary_df = compute_summary(metrics)

    # Small network sample (first 10 rows)
    net_sample = metrics.network.head(10).copy()

    # HTML
    head = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
  h1 {{ margin-top: 0; }}
  section {{ margin: 24px 0; }}
  figure {{ margin: 16px 0; }}
  figcaption {{ color: #666; font-size: 0.9em; }}
  .tbl {{ border-collapse: collapse; width: 100%; }}
  .tbl th, .tbl td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  .meta {{ color: #666; font-size: 0.95em; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
  @media (min-width: 1000px) {{
    .grid {{ grid-template-columns: 1fr 1fr; }}
  }}
  code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">Run directory: <code>{run_dir}</code></div>
</header>
"""

    body = ""

    # Summary table
    body += _table_html(summary_df, "Run Summary")

    # Plots
    body += '<section class="grid">'
    body += _img_html(phys_png, "Physical metrics: Loss [dB] & BSM visibility", embed_images, plots_dir / "physical.png")
    body += _img_html(prot_png, "Protocol metrics: QBER, Sifted rate", embed_images, plots_dir / "protocol.png")
    body += _img_html(sec_png,  "Security metrics: Secret key rate", embed_images, plots_dir / "security.png")
    body += "</section>"

    # Network sample
    body += _table_html(net_sample, "Network (sample)")

    tail = """
<footer>
  <p class="meta">Generated by <strong>QUASAR report.py</strong></p>
</footer>
</body>
</html>
"""
    return head + body + tail


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an HTML report for a QUASAR run directory.")
    p.add_argument("--run-dir", type=str, required=True, help="Path to run directory with metrics.")
    p.add_argument("--out", type=str, default=None, help="Output HTML file (default: <run-dir>/report.html).")
    p.add_argument("--title", type=str, default=None, help="Report title (default: derived from run dir).")
    p.add_argument("--embed-images", action="store_true", default=True, help="Embed PNGs as base64 (default true).")
    p.add_argument("--no-embed-images", dest="embed_images", action="store_false", help="Save PNGs next to HTML and link.")
    p.add_argument("--plots-dir", type=str, default=None, help="Directory for plot images (default: <run-dir>/plots).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    reconfigure_logging()
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run directory does not exist: {}", run_dir)
        return 2

    out_html = Path(args.out) if args.out else (run_dir / "report.html")
    title = args.title or f"QUASAR Report – {run_dir.name}"
    plots_dir = Path(args.plots_dir) if args.plots_dir else (run_dir / "plots")

    try:
        metrics = load_metrics(run_dir)
    except Exception as e:
        logger.exception("Failed to load metrics from {}: {}", run_dir, e)
        return 1

    html = build_report_html(
        run_dir=run_dir,
        metrics=metrics,
        title=title,
        embed_images=bool(args.embed_images),
        plots_dir=plots_dir,
    )
    out_html.write_text(html, encoding="utf-8")
    logger.info("Wrote report → {}", out_html)
    print(out_html)  # also print path to stdout for scripting convenience
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
