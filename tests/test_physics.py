# tests/test_physics.py
"""
Physics layer tests
===================

Covers:
- FiberChannel attenuation
- FreeSpaceChannel transmission efficiency
- Detector response (dark counts, dead time)
"""

from __future__ import annotations

import math
import pytest

from sequence_ext.physics.fiber import FiberChannel
from sequence_ext.physics.free_space import FreeSpaceChannel
from sequence_ext.physics.detectors import Detector


# -----------------------------------------------------------------------------
# FiberChannel
# -----------------------------------------------------------------------------
def test_fiberchannel_loss_and_transmission():
    fc = FiberChannel(length_km=50, attenuation_db_per_km=0.2)
    loss = fc.loss_dB()
    assert pytest.approx(loss, rel=1e-6) == 50 * 0.2
    eta = fc.transmission_eta()
    # 10^(-loss/10)
    assert math.isclose(eta, 10 ** (-loss / 10), rel=1e-12)


# -----------------------------------------------------------------------------
# FreeSpaceChannel
# -----------------------------------------------------------------------------
def test_free_space_clear_conditions():
    fso = FreeSpaceChannel(
        distance_km=100,
        attenuation_db_per_km=0.05,
        cn2=1e-15,
        pointing_sigma_urad=3.0,
    )
    eta = fso.transmission_eta()
    assert 0 < eta <= 1
    # Longer distance should reduce eta
    fso2 = FreeSpaceChannel(
        distance_km=200,
        attenuation_db_per_km=0.05,
        cn2=1e-15,
        pointing_sigma_urad=3.0,
    )
    assert fso2.transmission_eta() < eta


# -----------------------------------------------------------------------------
# Detector
# -----------------------------------------------------------------------------
def test_detector_efficiency_and_dark_counts():
    d = Detector(
        eta=0.8,
        dark_count_prob=1e-6,
        dead_time_ns=50,
        afterpulse_prob=0.01,
    )
    # Check effective detection probability for 1000 photons
    p = d.detect_prob(n_photons=1000)
    assert 0 <= p <= 1

    # Ensure dark counts contribute when no photons
    p_dark = d.detect_prob(n_photons=0)
    assert p_dark >= d.dark_count_prob
