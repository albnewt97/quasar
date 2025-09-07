# tests/test_configs.py
"""
Config schema & YAML loading tests
==================================

Covers:
- Pydantic validation for core fields
- Weather union (preset string vs inline object)
- YAML loader error handling
- Example factory roundtrip
"""

from __future__ import annotations

from pathlib import Path
import textwrap
import pytest

from configs.schema import (
    Config,
    Scenario,
    Devices,
    WeatherInline,
    load_config,
    example_scenario1_equidistant,
)


def write(tmp: Path, rel: str, content: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return p


def test_pydantic_minimal_validates():
    cfg = Config(
        scenario=Scenario(
            name="scenario1_equidistant",
            pulse_rate_hz=50_000_000,
            duration_s=2.0,
            output_dir="data/runs/test",
        ),
        devices=Devices(),
        weather=None,
    )
    assert cfg.scenario.name == "scenario1_equidistant"
    assert cfg.devices.detector_eta == pytest.approx(0.8)


def test_weather_inline_validation():
    w = WeatherInline(
        attenuation_db_per_km=0.05,
        cn2=1e-15,
        pointing_sigma_urad=3.0,
    )
    cfg = Config(
        scenario=Scenario(
            name="scenario2_moving",
            pulse_rate_hz=40_000_000,
            duration_s=1.0,
            output_dir="out",
        ),
        devices=Devices(),
        weather=w.model_dump(),  # allow dict
    )
    # Union allows dict; model_validate should keep structure
    assert isinstance(cfg.weather, dict)
    assert cfg.weather["attenuation_db_per_km"] == pytest.approx(0.05)


def test_yaml_load_valid(tmp_path: Path):
    path = write(
        tmp_path,
        "cfg.yaml",
        """
        scenario:
          name: scenario1_uneven
          pulse_rate_hz: 30000000
          duration_s: 1.5
          output_dir: data/runs/yaml_ok
        devices:
          detector_eta: 0.82
          detector_dark_per_gate: 1.0e-6
          detector_dead_time_ns: 60
          detector_afterpulse: 0.02
          bsm_visibility: 0.98
          coincidence_window_ps: 500
        weather: null
        """,
    )
    cfg = load_config(path)
    assert cfg.scenario.name == "scenario1_uneven"
    assert cfg.scenario.pulse_rate_hz == 30_000_000
    assert cfg.weather is None


def test_yaml_load_invalid_top_level(tmp_path: Path):
    bad = write(tmp_path, "bad.yaml", "- not_a_mapping\n- still_not_a_mapping\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_yaml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_example_factory_roundtrip(tmp_path: Path):
    cfg = example_scenario1_equidistant(output_dir=str(tmp_path / "run"))
    # Ensure it validates and fields are sane
    cfg2 = Config.model_validate(cfg.model_dump())
    assert cfg2.scenario.output_dir.endswith("run")
    assert cfg2.devices.coincidence_window_ps == 500
