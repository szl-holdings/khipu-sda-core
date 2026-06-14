"""
test_orbit_screen.py — REAL assertions on the orbital conjunction screen.

These tests run actual sgp4 propagation + the pairwise closest-approach screen
(no mocks). They assert the screen finds the real, sub-threshold ISS X derived-
companion conjunction with a sane TCA/miss/rel-speed, and that the honest
SCREENING-GRADE posture (no Pc, conformal ESTIMATE band) is present.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sgp4.api import jday

from szl_sda_orbit import (screen_catalog, demo_catalog, derive_companion_tle,
                           tle_checksum, closest_approach, propagate_tle,
                           SAMPLE_TLE_ISS, HONEST_LIMIT)


def test_derived_companion_tle_is_valid():
    """The derived companion must be a syntactically valid TLE (good checksum)
    and parse + propagate under sgp4 without error."""
    l1, l2 = derive_companion_tle(*SAMPLE_TLE_ISS, delta_mean_anomaly_deg=0.02)
    assert len(l2) == 69
    assert int(l2[-1]) == tle_checksum(l2)            # recomputed checksum valid
    jd0, fr0 = jday(2019, 12, 9, 12, 0, 0)
    states = propagate_tle(l1, l2, jd0, fr0, n_steps=10, step_minutes=1.0)
    assert all(s.error_code == 0 for s in states)     # propagates cleanly


def test_screen_finds_real_iss_companion_conjunction():
    """REAL sgp4 screen must flag exactly the ISS X companion pair < 5 km, with a
    physically sane miss distance, sub-step TCA, and small relative speed (a
    co-orbiting trailing companion has near-zero encounter velocity)."""
    cat = demo_catalog()
    jd0, fr0 = jday(2019, 12, 9, 12, 0, 0)
    events = screen_catalog(cat, jd0, fr0, n_steps=180, step_minutes=0.5,
                            threshold_km=5.0)
    assert len(events) == 1, f"expected 1 conjunction, got {len(events)}"
    ev = events[0]
    # the only sub-5km pair is ISS X its derived companion
    assert "ISS" in ev.object_a and "ISS" in ev.object_b
    # real miss distance: strictly between 0 and the 5 km threshold
    assert 0.0 < ev.miss_distance_km < 5.0
    # co-orbiting companion: tiny relative speed at closest approach
    assert ev.relative_speed_km_s < 0.1
    # TCA falls inside the screening window (180 steps * 0.5 min = 90 min)
    assert 0.0 <= ev.tca_offset_s <= 180 * 0.5 * 60.0
    # honest posture present on the event
    assert "SCREENING-GRADE" in ev.note and "Pc" in ev.note


def test_no_false_conjunction_at_tight_threshold():
    """A far-apart catalog pair must NOT be flagged: with only the real public
    objects (no companion) and a tight threshold, the screen finds nothing."""
    cat = {
        "VANGUARD": demo_catalog()["VANGUARD 00005 [public verif TLE]"],
        "GOES": demo_catalog()["GOES 06251 [public verif TLE]"],
    }
    jd0, fr0 = jday(2006, 6, 25, 0, 0, 0)
    events = screen_catalog(cat, jd0, fr0, n_steps=120, step_minutes=1.0,
                            threshold_km=5.0)
    assert events == [], "two unrelated LEO objects should not conjunct < 5 km"


def test_closest_approach_geometry_is_real():
    """closest_approach must return a finite, refined minimum strictly <= the
    coarse discrete minimum (parabolic refinement never increases the miss)."""
    cat = demo_catalog()
    jd0, fr0 = jday(2019, 12, 9, 12, 0, 0)
    a = propagate_tle(*cat["ISS (ZARYA) 25544 [public sample]"], jd0, fr0,
                      n_steps=180, step_minutes=0.5)
    b = propagate_tle(*cat["ISS-COMPANION [DERIVED sample]"], jd0, fr0,
                      n_steps=180, step_minutes=0.5)
    ca = closest_approach(a, b, step_minutes=0.5)
    assert ca["ok"]
    # discrete minimum at the reported index
    coarse = float(np.linalg.norm(a[ca["t_index"]].r_km - b[ca["t_index"]].r_km))
    assert ca["miss_km"] <= coarse + 1e-9
    assert np.isfinite(ca["miss_km"]) and ca["miss_km"] > 0.0


def test_conformal_band_is_estimate_and_brackets_miss():
    """When calibration miss-distances are supplied, each event carries an honest
    conformal ESTIMATE band whose lower bound brackets the observed miss."""
    cat = demo_catalog()
    jd0, fr0 = jday(2019, 12, 9, 12, 0, 0)
    calib = np.array([7990.0, 6500.0, 8800.0, 7200.0, 9100.0])
    events = screen_catalog(cat, jd0, fr0, n_steps=180, step_minutes=0.5,
                            threshold_km=5.0, calib_scores=calib)
    ev = events[0]
    assert ev.confidence is not None
    assert ev.confidence["label"] == "ESTIMATE"           # never a certainty
    assert ev.confidence["lo"] <= ev.miss_distance_km <= ev.confidence["hi"]
    assert "Pc" in ev.confidence["note"]                  # honest: not a Pc


def test_honest_limit_disclaims_pc_grade():
    """Module-level HONEST_LIMIT must disclaim Pc/covariance-grade explicitly."""
    assert "SCREENING-GRADE" in HONEST_LIMIT
    assert "Pc" in HONEST_LIMIT and "covariance" in HONEST_LIMIT.lower()


if __name__ == "__main__":
    test_derived_companion_tle_is_valid()
    test_screen_finds_real_iss_companion_conjunction()
    test_no_false_conjunction_at_tight_threshold()
    test_closest_approach_geometry_is_real()
    test_conformal_band_is_estimate_and_brackets_miss()
    test_honest_limit_disclaims_pc_grade()
    print("test_orbit_screen: PASS (real sgp4 conjunction screen, honest band)")
