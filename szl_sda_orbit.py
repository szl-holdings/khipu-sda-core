"""
szl_sda_orbit.py — SZL SDA (Space Domain Awareness) ROADMAP-SEED orbital stub.

================================ ATTRIBUTION ================================
CLEAN-ROOM STATEMENT
--------------------
Clean-room, SZL-native. INSPIRED BY the *publicly described capability* of True
Anomaly Inc.'s "Mosaic" platform (Space Domain Awareness, Detect/Track/ID of
on-orbit objects, conjunction/RPO awareness). NO proprietary code seen or copied.

Orbit propagation uses **python-sgp4** (brandon-rhodes/python-sgp4, MIT license)
to propagate a Two-Line-Element set (TLE) — this is the standard, openly-licensed
SGP4/SDP4 propagator. We add a small, transparent conjunction/anomaly check on top
in numpy. No proprietary ephemeris or catalog is bundled.
============================================================================

HONEST POSTURE & SCOPE LABEL (Doctrine v11):
  *** THIS IS THE SDA ROADMAP SEED — the SPACE-DOMAIN EXTENSION. ***
  TODAY'S killinchu does drone / vessel / counter-UAS (air + maritime). Orbital
  DTID/SDA is ROADMAP, NOT a shipped capability. This module is an honest seed:
  it propagates a TLE and flags a close-approach (conjunction) and a simple orbital
  anomaly. It is NOT a certified conjunction-assessment system; outputs are
  ADVISORY (Λ = Conjecture 1) under human-on-the-loop.
  - Sovereign / own-metal: pure local sgp4 + numpy, no network, 0 CDN.
  - We NEVER fabricate orbital ground truth; example TLEs below are public,
    well-known sample TLEs included for a runnable demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sgp4.api import Satrec, jday


# Public, well-known sample TLEs (for a runnable, sovereign demo only).
# ISS (ZARYA) — a canonical public example TLE distributed with sgp4 docs.
SAMPLE_TLE_ISS = (
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
)


@dataclass
class OrbitState:
    """Propagated orbital state at a UTC instant (TEME frame, km / km-s)."""
    jd: float
    fr: float
    r_km: np.ndarray            # position vector [x, y, z] km (TEME)
    v_km_s: np.ndarray          # velocity vector [vx, vy, vz] km/s
    error_code: int = 0


@dataclass
class ConjunctionEvent:
    """An advisory conjunction (close-approach) finding between two objects."""
    t_index: int
    jd: float
    miss_distance_km: float
    threshold_km: float
    relative_speed_km_s: float
    advisory: str               # Λ verdict: allow | advisory | deny
    note: str = ("ADVISORY only (Λ = Conjecture 1). Not a certified CA product. "
                 "Human-on-the-loop. SDA = SZL roadmap seed, not shipped.")


def propagate_tle(tle_line1: str, tle_line2: str,
                  jd0: float, fr0: float,
                  n_steps: int = 90, step_minutes: float = 1.0) -> list:
    """Propagate a TLE with SGP4 (python-sgp4, MIT) over a time window.

    Returns a list of OrbitState samples. Time advances by `step_minutes` per step.
    """
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    states = []
    for i in range(n_steps):
        # advance fractional day by step_minutes
        fr = fr0 + (i * step_minutes) / (24.0 * 60.0)
        jd = jd0 + int(fr)        # roll integer-day overflow into jd
        fr = fr - int(fr)
        e, r, v = sat.sgp4(jd, fr)
        states.append(OrbitState(jd=jd, fr=fr,
                                 r_km=np.array(r), v_km_s=np.array(v),
                                 error_code=e))
    return states


def detect_conjunction(states_a: list, states_b: list,
                       threshold_km: float = 50.0) -> list:
    """Flag advisory conjunctions where two propagated objects come within
    `threshold_km`. Pure-numpy close-approach screen over matched time samples.
    """
    events = []
    n = min(len(states_a), len(states_b))
    for i in range(n):
        ra, rb = states_a[i].r_km, states_b[i].r_km
        va, vb = states_a[i].v_km_s, states_b[i].v_km_s
        if states_a[i].error_code or states_b[i].error_code:
            continue
        miss = float(np.linalg.norm(ra - rb))
        rel_speed = float(np.linalg.norm(va - vb))
        if miss <= threshold_km:
            # advisory severity by distance band
            if miss <= threshold_km * 0.25:
                verdict = "deny"      # advisory-deny: high-concern close approach
            elif miss <= threshold_km * 0.6:
                verdict = "advisory"
            else:
                verdict = "advisory"
            events.append(ConjunctionEvent(
                t_index=i, jd=states_a[i].jd + states_a[i].fr,
                miss_distance_km=miss, threshold_km=threshold_km,
                relative_speed_km_s=rel_speed, advisory=verdict))
    return events


def detect_orbital_anomaly(states: list, sigma: float = 4.0) -> list:
    """Flag timesteps where orbital RADIUS deviates anomalously (robust z-score).

    A maneuver / decay / bad element set shows up as a radius jump. This reuses the
    same robust median/MAD idea as the core z-score detector (tsod lineage). Returns
    indices of anomalous samples. ADVISORY only.
    """
    radii = np.array([np.linalg.norm(s.r_km) for s in states
                      if s.error_code == 0])
    if radii.size < 5:
        return []
    med = np.median(radii)
    mad = np.median(np.abs(radii - med)) + 1e-9
    z = np.abs(radii - med) / (1.4826 * mad)
    return [int(i) for i in np.where(z > sigma)[0]]


if __name__ == "__main__":
    jd0, fr0 = jday(2019, 12, 9, 12, 0, 0)
    states = propagate_tle(*SAMPLE_TLE_ISS, jd0, fr0, n_steps=90, step_minutes=1.0)
    ok = sum(1 for s in states if s.error_code == 0)
    print(f"SGP4 propagated {len(states)} samples ({ok} ok). "
          f"r0={np.round(states[0].r_km,1)} km")

    # build a synthetic 'shadowing' second object offset by ~30 km to force a
    # conjunction advisory (demo only — clearly synthetic).
    shadow = []
    for s in states:
        s2 = OrbitState(jd=s.jd, fr=s.fr,
                        r_km=s.r_km + np.array([20.0, 20.0, 10.0]),
                        v_km_s=s.v_km_s, error_code=s.error_code)
        shadow.append(s2)
    events = detect_conjunction(states, shadow, threshold_km=50.0)
    print(f"conjunction advisories: {len(events)} (first miss "
          f"~{events[0].miss_distance_km:.1f} km, verdict={events[0].advisory})"
          if events else "no conjunction advisories")
    print(f"orbital-anomaly samples: {detect_orbital_anomaly(states)}")
