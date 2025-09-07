# configs/__init__.py
"""
Configuration package
=====================

Provides schema definitions and example YAML configuration files
for QUASAR simulations.

Modules
-------
- schema.py : Pydantic models for validating simulation configs.
- (YAML files) : Example scenario configurations.

Usage
-----
from configs.schema import ScenarioConfig
cfg = ScenarioConfig.model_validate_yaml("configs/scenario1_equidistant.yaml")
"""

__all__ = []
