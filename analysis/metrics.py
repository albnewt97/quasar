# =============================================
# File: analysis/metrics.py
# Description: Load results JSON from scenarios and compute KPI metrics:
#   - BSM counts & rate, sifted key size & rate, QBER(Z),
#   - secret fraction & secret bits, secret bit rate (bits/s).
# Emits CSV/Parquet summaries and optional plots.
# =============================================

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional heavy deps guarded:
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore

try:
    # We only import if plotting is requested
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


APP_NAME = "quasar-metrics"
APP_VERSION = "1.0.0"


# ---------------------------
# Helpers
# ---------------------------

def _setup_logging(level: str) -> None:
    level_no = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_no,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported JSON at {path} (top-level is not an object)")
    return data


def _safe_get(d: Dict[str, Any], dotted: str, default=None):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _bin_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    import math
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


@dataclass(slots=True)
class RunMetrics:
    run_path: str
    scenario: str
    backend: str
    duration_ns: int
    duration_s: float

    # Raw counts from scenario metrics:
    bsm_results: int
    bsm_successes: int
    sift_accepts: int
    sift_drops: int
    sifted_key_len_z: int

    # Derived:
    bsm_success_rate: float
    qber_z: float
    secret_fraction_est: float
    secret_bits_est: int

    # Normalized (rates):
    bsm_results_per_s: float
    bsm_successes_per_s: float
    sifted_rate_bps: float        # sifted bits per second (Z)
    secret_rate_bps: float        # secret bits per second

    # Meta passthrough:
    finished_utc: Optional[str]
    seed: Optional[int]
    notes: Optional[str] = None


def _coerce_run_metrics(run_json: Dict[str, Any], path: Path) -> RunMetrics:
    # Simulator duration
    duration_ns = (
        _safe_get(run_json, "simulator.duration_ns")
        or _safe_get(run_json, "meta.simulator.duration_ns")
        or 0
    )
    if not isinstance(duration_ns, int) or duration_ns <= 0:
        # Last resort: if events exist, infer from meta.elapsed_sec
        elapsed_s = _safe_get(run_json, "meta.elapsed_sec") or 0.0
        duration_ns = int(float(elapsed_s) * 1e9) if elapsed_s else 0
    duration_s = duration_ns / 1e9 if duration_ns else 0.0

    metrics = run_json.get("metrics", {}) or {}
    scenario = run_json.get("scenario", "unknown")
    backend = run_json.get("backend", "unknown")

    bsm_results = int(metrics.get("bsm_results", 0))
    bsm_successes = int(metrics.get("bsm_successes", 0))
    sift_accepts = int(metrics.get("sift_accepts", 0))
    sift_drops = int(metrics.get("sift_drops", 0))
    sifted_key_len_z = int(metrics.get("sifted_key_len_z", 0))
    qber_z = float(metrics.get("qber_z", 0.0))
    secret_fraction_est = float(metrics.get("secret_fraction_est", 0.0))
    secret_bits_est = int(metrics.get("secret_bits_est", 0))

    # Derived
    bsm_success_rate = (bsm_successes / bsm_results) if bsm_results else 0.0
    if duration_s > 0:
        bsm_results_per_s = bsm_results / duration_s
        bsm_successes_per_s = bsm_successes / duration_s
        sifted_rate_bps = sifted_key_len_z / duration_s
        secret_rate_bps = secret_bits_est / duration_s
    else:
        bsm_results_per_s = 0.0
        bsm_successes_per_s = 0.0
        sifted_rate_bps = 0.0
        secret_rate_bps = 0.0

    finished_utc = _safe_get(run_json, "meta.finished_utc")
    seed = _safe_get(run_json, "simulator.random_seed")

    return RunMetrics(
        run_path=str(path),
        scenario=scenario,
        backend=backend,
        duration_ns=duration_ns,
        duration_s=duration_s,
        bsm_results=bsm_results,
        bsm_successes=bsm_successes,
        sift_accepts=sift_accepts,
        sift_drops=sift_drops,
        sifted_key_len_z=sifted_key_len_z,
        bsm_success_rate=bsm_success_rate,
        qber_z=qber_z,
        secret_fraction_est=secret_fraction_est,
        secret_bits_est=secret_bits_est,
        bsm_results_per_s=bsm_results_per_s,
        bsm_successes_per_s=bsm_successes_per_s,
        sifted_rate_bps=sifted_rate_bps,
        secret_rate_bps=secret_rate_bps,
        finished_utc=finished_utc,
        seed=seed,
    )


def _validate_metrics_consistency(m: RunMetrics) -> List[str]:
    errs: List[str] = []
    if not (0.0 <= m.qber_z <= 0.5 + 1e-9):
        errs.append(f"QBER out of range [0,0.5]: {m.qber_z}")
    if m.bsm_successes > m.bsm_results:
        errs.append("bsm_successes > bsm_results (impossible)")
    if m.secret_fraction_est < 0.0 or m.secret_fraction_est > 1.0:
        errs.append(f"secret_fraction_est out of [0,1]: {m.secret_fraction_est}")
    if m.duration_ns <= 0:
        errs.append("duration_ns <= 0 (cannot compute rates)")
    return errs


def _to_dataframe(metrics_list: List[RunMetrics]):
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for table export. `pip install pandas`")
    rows = [asdict(m) for m in metrics_list]
    return pd.DataFrame(rows)


def _write_csv(df, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _write_parquet(df, out_path: Path) -> None:
    if pq is None or pa is None:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Parquet. `pip install pyarrow`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path)


def _plot_basic(df, out_dir: Path, title_suffix: str = "") -> None:
    if plt is None:  # pragma: no cover
        logging.warning("matplotlib not available; skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Secret rate bar chart
    fig1 = plt.figure()
    df.plot(kind="bar", x="run_path", y="secret_rate_bps", legend=False)
    plt.ylabel("Secret bits / s")
    plt.title(f"Secret Key Rate {title_suffix}".strip())
    plt.tight_layout()
    fig1_path = out_dir / "secret_rate_bps.png"
    plt.savefig(fig1_path)
    plt.close(fig1)

    # QBER scatter
    fig2 = plt.figure()
    df.plot(kind="scatter", x="run_path", y="qber_z")
    plt.ylabel("QBER (Z)")
    plt.title(f"QBER (Z) {title_suffix}".strip())
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig2_path = out_dir / "qber_z.png"
    plt.savefig(fig2_path)
    plt.close(fig2)

    logging.info("Saved plots: %s, %s", fig1_path, fig2_path)


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute MDI-QKD KPI metrics from run results JSON and export summaries."
    )
    p.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Path(s) to results JSON file(s) produced by scripts/run_scenario.py",
    )
    p.add_argument(
        "--out-dir",
        required=False,
        default="data/runs",
        help="Directory to write summaries (CSV/Parquet/plots).",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Write CSV summary to <out-dir>/metrics.csv",
    )
    p.add_argument(
        "--parquet",
        action="store_true",
        help="Write Parquet summary to <out-dir>/metrics.parquet (requires pyarrow).",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Generate basic plots (requires matplotlib).",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Also print JSON summary to stdout.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    p.add_argument(
        "--title",
        default="",
        help="Optional title suffix for plots.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    log = logging.getLogger("metrics")

    results_paths: List[Path] = []
    for pth in args.results:
        p = Path(pth).expanduser().resolve()
        if not p.exists():
            log.error("Results file not found: %s", p)
            return 2
        results_paths.append(p)

    metrics_list: List[RunMetrics] = []
    for rp in results_paths:
        try:
            run_json = _read_json(rp)
            m = _coerce_run_metrics(run_json, rp)
            errs = _validate_metrics_consistency(m)
            if errs:
                for e in errs:
                    log.warning("Consistency warning for %s: %s", rp, e)
            metrics_list.append(m)
        except Exception as e:
            log.exception("Failed to process %s: %s", rp, e)
            return 3

    # Build a JSON-serializable summary
    summary_dict: Dict[str, Any] = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "count": len(metrics_list),
        "metrics": [asdict(m) for m in metrics_list],
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write tabular outputs
    try:
        if args.csv or args.parquet or args.plots:
            df = _to_dataframe(metrics_list)
        else:
            df = None  # type: ignore

        if args.csv and df is not None:
            _write_csv(df, out_dir / "metrics.csv")
            log.info("Wrote CSV -> %s", out_dir / "metrics.csv")

        if args.parquet and df is not None:
            _write_parquet(df, out_dir / "metrics.parquet")
            log.info("Wrote Parquet -> %s", out_dir / "metrics.parquet")

        if args.plots and df is not None:
            _plot_basic(df, out_dir, title_suffix=args.title)
    except Exception as e:
        log.exception("Table/plot export failed: %s", e)
        return 4

    # Also write a compact JSON summary for programmatic use
    json_out = out_dir / "metrics_summary.json"
    try:
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        log.info("Wrote JSON summary -> %s", json_out)
    except Exception as e:
        log.exception("Failed to write JSON summary: %s", e)
        return 5

    if args.stdout:
        print(json.dumps(summary_dict, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
