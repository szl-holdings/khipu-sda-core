"""
test_receipt_roundtrip.py — the §3.0 envelope round-trips and the in-toto
receipt is well-formed, HONEST (UNSIGNED, verified=false), and sovereign.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from szl_mosaic_core import SZLMosaicCore
from szl_sda_envelope import evaluate_timed, VALID_STAGES


def _fitted_core():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, size=(300, 6))
    core = SZLMosaicCore().fit(Xtr)
    calib = core._combined_scores(Xtr)
    return core, calib


def test_envelope_roundtrip_json():
    core, calib = _fitted_core()
    rng = np.random.default_rng(1)
    feats = rng.normal(0, 1, size=(1, 6)); feats[0, 0] += 9.0
    env = evaluate_timed({"track_id": "TRK-0001", "features": feats},
                         core, calib, stage="TWA")
    # serialises and deserialises losslessly
    s = json.dumps(env)
    back = json.loads(s)
    assert back["track_id"] == "TRK-0001"
    assert back["stage"] in VALID_STAGES
    assert 0.0 <= back["anomaly_score"] <= 1.0
    assert back["lambda_axis"]["advisory"] is True       # Λ = Conjecture 1
    assert back["confidence"]["label"] == "ESTIMATE"
    assert back["confidence"]["lo"] <= back["confidence"]["hi"]


def test_receipt_intoto_v1_and_honest():
    core, calib = _fitted_core()
    feats = np.zeros((1, 6)); feats[0, 0] = 9.0
    env = evaluate_timed({"track_id": "TRK-0002", "features": feats},
                         core, calib, stage="FUSE")
    r = env["receipt"]
    assert r["_type"] == "https://in-toto.io/Statement/v1"
    assert r["predicateType"].endswith("/sda-anomaly/v1")
    assert r["subject"][0]["name"] == "TRK-0002"
    assert "sha256" in r["subject"][0]["digest"]
    # HONESTY invariants
    assert r["predicate"]["verified"] is False           # UNSIGNED
    assert r["predicate"]["sovereign"] is True
    assert env["_signing"]["status"] == "UNSIGNED"
    assert "alibi" not in json.dumps(r).lower()           # alibi-detect excluded


def test_orbital_track_has_sgp4_hash():
    core, calib = _fitted_core()
    feats = np.zeros((1, 6))
    tle = ("1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
           "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482")
    env = evaluate_timed({"track_id": "SAT-25544", "features": feats, "tle": tle},
                         core, calib, stage="DTID", confidence_method="PAC-Bayes")
    assert env["sgp4"] is not None
    assert env["sgp4"]["tle_hash"].startswith("sha256:")
    assert env["confidence"]["method"] == "PAC-Bayes"


def test_air_track_sgp4_null():
    core, calib = _fitted_core()
    feats = np.zeros((1, 6))
    env = evaluate_timed({"track_id": "AIR-1", "features": feats},
                         core, calib, stage="DTID")
    assert env["sgp4"] is None


if __name__ == "__main__":
    test_envelope_roundtrip_json()
    test_receipt_intoto_v1_and_honest()
    test_orbital_track_has_sgp4_hash()
    test_air_track_sgp4_null()
    print("test_receipt_roundtrip: PASS (envelope round-trips; receipt is "
          "in-toto v1, HONEST UNSIGNED/verified=false, sgp4 hash present for "
          "orbital, null for air)")
