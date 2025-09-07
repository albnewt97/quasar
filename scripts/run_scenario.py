#!/usr/bin/env python3
"""
Run a QUASAR scenario from a YAML config.
"""
import argparse
import pathlib
import sys
import yaml

# Ensure local packages are importable when invoked from repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sequence_ext.scenarios.scenario1_static import Scenario1Static # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run a QUASAR scenario")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scenario_id = cfg.get("scenario", {}).get("id", "1a")
    if scenario_id not in ("1a", "1b"):
        raise ValueError("This runner currently supports scenario 1a/1b only.")

    out_dir = cfg.get("run", {}).get("output_dir", "data/runs")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    scenario = Scenario1Static(cfg)
    results = scenario.run()
    scenario.export(results, out_dir)
    print("Run complete. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
