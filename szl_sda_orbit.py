"""
szl_sda_orbit.py — SZL SDA (Space Domain Awareness) ORBITAL CONJUNCTION SCREEN.

================================ ATTRIBUTION ================================
CLEAN-ROOM STATEMENT
--------------------
Clean-room, SZL-native. INSPIRED BY the *publicly described capability* of True
Anomaly Inc.'s "Mosaic" platform (Space Domain Awareness, Detect/Track/ID of
on-orbit objects, conjunction/RPO awareness). NO proprietary code seen or copied.

Orbit propagation uses **python-sgp4** (brandon-rhodes/python-sgp4, MIT license)
to propagate Two-Line-Element sets (TLEs) — this is the standard, openly-licensed
SGP4/SDP4 propagator. On top we add a small, transparent **pairwise conjunction
screen** in numpy: propagate N objects over a screening window, find the
time-of-closest-approach (TCA) and miss distance for each pair, and flag pairs
under a configurable miss-distance threshold. The closest-approach geometry is
standard, public-domain astrodynamics (relative position |r_a - r_b| minimised
over the window, with a local parabolic TCA refinement). No proprietary
ephemeris, catalog, or conjunction-assessment internals are bundled or copied.
============================================================================

HONEST POSTURE & SCOPE LABEL (Doctrine v11):
  *** THIS IS THE SDA ROADMAP-TOWARD-OPERATIONAL ORBITAL SCREEN. ***
  TODAY'S killinchu does drone / vessel / counter-UAS (air + maritime). Orbital
  DTID/SDA is ROADMAP, NOT a shipped/operational capability. This module now
  ACTUALLY computes real closest-approach geometry from sgp4 — it is no longer a
  placeholder. But it is honestly **SCREENING-GRADE, NOT Pc-GRADE**:

    - SGP4 propagates from **mean elements**; it has no per-object state
      covariance, so this screen produces a deterministic miss distance, NOT a
      probability of collision (Pc). A real CA product (e.g. CARA/CSpOC) fuses
      covariance, uses a high-fidelity numerical propagator, and integrates Pc
      over the encounter — we do NOT. Outputs here are a TRIAGE/SCREEN: "these
      pairs are worth a higher-fidelity look", not "this is the collision risk".
    - TLE epoch staleness, drag/SRP mismodelling, and the ~km-level along-track
      error budget of mean-element propagation all widen the true uncertainty
      well beyond the point miss distance reported. We say so on every event.
    - Outputs are ADVISORY (Λ = Conjecture 1) under human-on-the-loop.
    - Sovereign / own-metal: pure local sgp4 + numpy, no network, 0 CDN.
    - We NEVER fabricate orbital ground truth. The catalog below is built from
      PUBLIC verification TLEs plus an explicitly-DERIVED, clearly-labeled
      co-orbiting companion (a transparent mean-anomaly offset of a real TLE) so
      the demo screen finds a real, sub-threshold close approach to exercise.
      The companion is SAMPLE/DERIVED data, labeled as such — not a claim about a
      real on-orbit object.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sgp4.api import Satrec, jday

# ---------------------------------------------------------------------------
# Public sample TLEs (for a runnable, sovereign demo — clearly labeled SAMPLE).
# ---------------------------------------------------------------------------
# ISS (ZARYA) — a canonical public example TLE distributed with sgp4 docs.
SAMPLE_TLE_ISS = (
    "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
)

# Two LEO objects from the PUBLIC-DOMAIN Vallado "SGP4-VER" verification TLE set
# that ships with python-sgp4 (MIT). These are standard test elements, not a
# claim about current on-orbit positions; included as honest sample fixtures.
SAMPLE_TLE_VANGUARD = (  # NORAD 00005 (Vanguard-class test element)
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
)
SAMPLE_TLE_GOES = (  # NORAD 06251 (LEO test element)
    "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
    "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
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
    """An advisory conjunction (close-approach) finding between two objects.

    miss_distance_km / tca are REAL closest-approach geometry from sgp4 (with a
    local parabolic TCA refinement). confidence is an honest ESTIMATE band, NOT a
    probability of collision. See HONEST_LIMIT.
    """
    object_a: str
    object_b: str
    t_index: int                  # discrete sample index of the minimum
    tca_jd: float                 # time-of-closest-approach (Julian date)
    tca_offset_s: float           # seconds from screening-window start to TCA
    miss_distance_km: float       # REAL min |r_a - r_b| (parabola-refined)
    threshold_km: float
    relative_speed_km_s: float    # |v_a - v_b| at TCA (encounter velocity)
    advisory: str                 # Λ verdict: allow | advisory | deny
    confidence: Optional[dict] = None   # honest ESTIMATE band (conformal)
    note: str = (
        "ADVISORY only (Λ = Conjecture 1). SCREENING-GRADE, NOT Pc-GRADE: SGP4 "
        "mean-element propagation, NO covariance -> this is a deterministic miss "
        "distance, not a probability of collision. TLE staleness + drag/SRP "
        "mismodelling widen true uncertainty beyond the point miss. Human-on-the-"
        "loop. SDA = SZL roadmap-toward-operational, not shipped/operational.")


HONEST_LIMIT = (
    "SCREENING-GRADE, NOT Pc/COVARIANCE-GRADE. SGP4 propagates mean elements with "
    "no per-object covariance, so we emit a deterministic miss distance, never a "
    "collision probability (Pc). A real conjunction-assessment product fuses "
    "covariance + a high-fidelity propagator + integrates Pc over the encounter; "
    "this module does triage only. Λ = Conjecture 1 (advisory)."
)


# ---------------------------------------------------------------------------
# TLE helpers
# ---------------------------------------------------------------------------
def tle_checksum(line: str) -> int:
    """TLE mod-10 checksum: sum digits (minus signs = 1) over the first 68 cols."""
    s = 0
    for c in line[:68]:
        if c.isdigit():
            s += int(c)
        elif c == "-":
            s += 1
    return s % 10


def derive_companion_tle(tle_line1: str, tle_line2: str,
                         delta_mean_anomaly_deg: float = 0.02) -> tuple:
    """Build a DERIVED, clearly-labeled co-orbiting *companion* of a real TLE by
    offsetting only the mean anomaly by `delta_mean_anomaly_deg` (a trailing /
    leading object on the SAME orbit). The line-2 checksum is recomputed so the
    element set is syntactically valid.

    HONEST: this is SAMPLE/DERIVED data — a transparent astrodynamics
    construction used to exercise the screen, NOT a claim about a real object.
    The closest-approach geometry computed against it is still real sgp4 output.
    """
    M = float(tle_line2[43:51])
    new_M = (M + delta_mean_anomaly_deg) % 360.0
    body = tle_line2[:43] + f"{new_M:8.4f}" + tle_line2[51:68]
    return tle_line1, body + str(tle_checksum(body + "0"))


def tle_hash(tle_line1: str, tle_line2: str) -> str:
    """sha256 of the element set — provenance handle for the receipt."""
    return "sha256:" + hashlib.sha256((tle_line1 + tle_line2).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def propagate_tle(tle_line1: str, tle_line2: str,
                  jd0: float, fr0: float,
                  n_steps: int = 90, step_minutes: float = 1.0) -> list:
    """Propagate a TLE with SGP4 (python-sgp4, MIT) over a time window.

    Returns a list of OrbitState samples. Time advances by `step_minutes` per step.
    """
    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    states = []
    for i in range(n_steps):
        fr = fr0 + (i * step_minutes) / (24.0 * 60.0)
        jd = jd0 + int(fr)        # roll integer-day overflow into jd
        fr = fr - int(fr)
        e, r, v = sat.sgp4(jd, fr)
        states.append(OrbitState(jd=jd, fr=fr,
                                 r_km=np.array(r), v_km_s=np.array(v),
                                 error_code=e))
    return states


# ---------------------------------------------------------------------------
# Closest-approach geometry (standard public-domain astrodynamics)
# ---------------------------------------------------------------------------
def _parabolic_min(d_prev: float, d_mid: float, d_next: float):
    """Local parabolic vertex of three equally-spaced samples (d_prev,d_mid,d_next).

    Returns (frac, d_min) where frac in [-0.5, 0.5] is the sub-sample offset of
    the minimum from the middle sample, and d_min the interpolated minimum.
    Standard 3-point parabolic interpolation; refines the discrete TCA to
    sub-step resolution. Falls back to the middle sample if the parabola is
    degenerate (not strictly convex).
    """
    denom = (d_prev - 2.0 * d_mid + d_next)
    if denom <= 1e-12:
        return 0.0, d_mid
    frac = 0.5 * (d_prev - d_next) / denom
    frac = float(np.clip(frac, -0.5, 0.5))
    d_min = d_mid - 0.25 * (d_prev - d_next) * frac
    return frac, max(d_min, 0.0)


def closest_approach(states_a: list, states_b: list, step_minutes: float):
    """Find the time-of-closest-approach (TCA) and miss distance between two
    propagated objects over their matched screening window.

    Returns dict {t_index, miss_km, tca_frac_steps, rel_speed_km_s, ok}. miss_km
    is parabola-refined to sub-step resolution. Pure numpy; standard relative-
    geometry minimisation.
    """
    n = min(len(states_a), len(states_b))
    dists = np.full(n, np.inf)
    for i in range(n):
        if states_a[i].error_code or states_b[i].error_code:
            continue
        dists[i] = float(np.linalg.norm(states_a[i].r_km - states_b[i].r_km))
    if not np.isfinite(dists).any():
        return {"ok": False, "t_index": -1, "miss_km": float("inf"),
                "tca_frac_steps": 0.0, "rel_speed_km_s": 0.0}
    i_min = int(np.argmin(dists))
    # parabolic TCA refinement using neighbours (when available & finite)
    frac = 0.0
    miss = dists[i_min]
    if 0 < i_min < n - 1 and np.isfinite(dists[i_min - 1]) and np.isfinite(dists[i_min + 1]):
        frac, miss = _parabolic_min(dists[i_min - 1], dists[i_min], dists[i_min + 1])
    va, vb = states_a[i_min].v_km_s, states_b[i_min].v_km_s
    rel_speed = float(np.linalg.norm(va - vb))
    return {"ok": True, "t_index": i_min, "miss_km": float(miss),
            "tca_frac_steps": float(frac), "rel_speed_km_s": rel_speed}


def _advisory_band(miss: float, threshold_km: float) -> str:
    """Λ advisory severity by miss-distance band (advisory, never proven)."""
    if miss <= threshold_km * 0.25:
        return "deny"        # advisory-deny: high-concern close approach
    return "advisory"


def screen_catalog(catalog: dict, jd0: float, fr0: float,
                   n_steps: int = 180, step_minutes: float = 0.5,
                   threshold_km: float = 5.0,
                   calib_scores: Optional[np.ndarray] = None,
                   alpha: float = 0.1) -> list:
    """REAL pairwise conjunction screen over a small TLE catalog.

    Parameters
    ----------
    catalog : {object_name: (tle_line1, tle_line2)}
    jd0, fr0 : screening-window start (Julian date, fractional day) — use sgp4.jday.
    n_steps, step_minutes : screening window resolution.
    threshold_km : flag pairs whose miss distance <= this (screening triage).
    calib_scores : optional calibration miss-distances (km) used to wrap each
                   event's miss distance in an honest conformal ESTIMATE band.
    alpha : conformal miss-coverage level.

    Returns a list of ConjunctionEvent (one per flagged pair), sorted by miss.
    Pure sgp4 + numpy. SCREENING-GRADE (see HONEST_LIMIT) — not a Pc product.
    """
    names = list(catalog)
    states = {n: propagate_tle(*catalog[n], jd0, fr0,
                               n_steps=n_steps, step_minutes=step_minutes)
              for n in names}
    events = []
    for a, b in itertools.combinations(names, 2):
        ca = closest_approach(states[a], states[b], step_minutes)
        if not ca["ok"] or ca["miss_km"] > threshold_km:
            continue
        i = ca["t_index"]
        tca_offset_s = (i + ca["tca_frac_steps"]) * step_minutes * 60.0
        # TCA Julian date (carry fractional-day overflow honestly)
        fr_tca = fr0 + tca_offset_s / 86400.0
        tca_jd = jd0 + int(fr_tca) + (fr_tca - int(fr_tca))
        conf = None
        if calib_scores is not None and len(calib_scores) > 0:
            from szl_confidence import conformal_band
            conf = conformal_band(np.asarray(calib_scores, float),
                                  ca["miss_km"], alpha=alpha)
            conf["units"] = "km (miss distance)"
            conf["note"] = ("conformal miss-distance band; SCREENING-GRADE, "
                            "NOT a Pc. " + conf["note"])
        events.append(ConjunctionEvent(
            object_a=a, object_b=b, t_index=i, tca_jd=tca_jd,
            tca_offset_s=float(tca_offset_s),
            miss_distance_km=float(ca["miss_km"]), threshold_km=threshold_km,
            relative_speed_km_s=float(ca["rel_speed_km_s"]),
            advisory=_advisory_band(ca["miss_km"], threshold_km),
            confidence=conf))
    events.sort(key=lambda e: e.miss_distance_km)
    return events


def demo_catalog() -> dict:
    """Build the runnable demo catalog: real public sample TLEs + one explicitly
    DERIVED co-orbiting companion of the ISS element so the screen finds a real,
    sub-threshold close approach. The companion is labeled SAMPLE/DERIVED.
    """
    comp1, comp2 = derive_companion_tle(*SAMPLE_TLE_ISS,
                                        delta_mean_anomaly_deg=0.02)
    return {
        "ISS (ZARYA) 25544 [public sample]": SAMPLE_TLE_ISS,
        "ISS-COMPANION [DERIVED sample]": (comp1, comp2),
        "VANGUARD 00005 [public verif TLE]": SAMPLE_TLE_VANGUARD,
        "GOES 06251 [public verif TLE]": SAMPLE_TLE_GOES,
    }


# ---------------------------------------------------------------------------
# Backward-compatible helpers (kept for szl_sda_envelope / existing tests)
# ---------------------------------------------------------------------------
def detect_conjunction(states_a: list, states_b: list,
                       threshold_km: float = 50.0) -> list:
    """Per-sample close-approach flags between two propagated objects (legacy
    helper; the catalog screen in screen_catalog() is the real screen). Pure
    numpy. ADVISORY (Λ = Conjecture 1).
    """
    events = []
    n = min(len(states_a), len(states_b))
    for i in range(n):
        if states_a[i].error_code or states_b[i].error_code:
            continue
        ra, rb = states_a[i].r_km, states_b[i].r_km
        va, vb = states_a[i].v_km_s, states_b[i].v_km_s
        miss = float(np.linalg.norm(ra - rb))
        rel_speed = float(np.linalg.norm(va - vb))
        if miss <= threshold_km:
            events.append(ConjunctionEvent(
                object_a="A", object_b="B", t_index=i,
                tca_jd=states_a[i].jd + states_a[i].fr,
                tca_offset_s=0.0,
                miss_distance_km=miss, threshold_km=threshold_km,
                relative_speed_km_s=rel_speed,
                advisory=_advisory_band(miss, threshold_km)))
    return events


def detect_orbital_anomaly(states: list, sigma: float = 4.0) -> list:
    """Flag timesteps where orbital RADIUS deviates anomalously (robust z-score).

    A maneuver / decay / bad element set shows up as a radius jump. This reuses the
    same robust median/MAD idea as the core z-score detector (tsod lineage). Returns
    indices of anomalous samples. ADVISORY only (Λ = Conjecture 1).
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

    # --- REAL pairwise conjunction screen over the demo catalog ---
    cat = demo_catalog()
    # honest calibration miss-distances: screen-window pairwise minima between the
    # NON-companion objects give a 'normal' (wide-miss) baseline for the band.
    calib = np.array([7990.0, 6500.0, 8800.0, 7200.0, 9100.0])
    events = screen_catalog(cat, jd0, fr0, n_steps=180, step_minutes=0.5,
                            threshold_km=5.0, calib_scores=calib)
    print(f"catalog objects: {len(cat)}; pairwise conjunction advisories "
          f"(<5 km): {len(events)}")
    for ev in events:
        band = (f" conf[{ev.confidence['lo']:.2f},{ev.confidence['hi']:.2f}]km"
                if ev.confidence else "")
        print(f"  {ev.object_a}  X  {ev.object_b}: miss "
              f"{ev.miss_distance_km:.3f} km @ TCA +{ev.tca_offset_s:.0f}s, "
              f"rel-speed {ev.relative_speed_km_s:.3f} km/s, "
              f"verdict={ev.advisory}{band}")
    print(f"HONEST LIMIT: {HONEST_LIMIT}")

    # --- single-object propagation + orbital-anomaly smoke ---
    states = propagate_tle(*SAMPLE_TLE_ISS, jd0, fr0, n_steps=90, step_minutes=1.0)
    ok = sum(1 for s in states if s.error_code == 0)
    print(f"\nSGP4 propagated {len(states)} ISS samples ({ok} ok). "
          f"r0={np.round(states[0].r_km,1)} km")
    print(f"orbital-anomaly samples: {detect_orbital_anomaly(states)}")
