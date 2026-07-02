"""
szl_sda_envelope.py — the FROZEN §3.0 interface contract for khipu-sda-core.

This module emits the SDA verdict envelope exactly as specified in the CTO build
spec (estate_audit/mosaic_build_spec.md §3.0), so Dev2 (killinchu wiring), Dev3
(a11oy governance), and Dev4 (UDS packaging) can build against a stable shape.

Envelope shape (frozen v0):
{
  "track_id": "TRK-0001",
  "stage": "DTID|CHARACTERIZE|TWA|FUSE",
  "anomaly_score": 0.0,                       # [0,1]
  "lambda_axis": {"name":"anomaly_twa","value":0.0,"advisory":true},
  "confidence": {"lo":0.0,"hi":0.0,"method":"PAC-Bayes|conformal","label":"ESTIMATE"},
  "sgp4": {"tle_hash":"sha256:…","propagated":true} | null,
  "receipt": { in-toto Statement v1, predicateType .../sda-anomaly/v1 },
  "_signing": {"status":"UNSIGNED|SIGNED","signed_by":"szl_lake/khipu DSSE (Ed25519/P-256)"}
}

HONESTY (Doctrine v11):
  - UNSIGNED is STRUCTURAL-ONLY; never a false "verified/green". No fabricated
    signatures, ever. receipt.predicate.verified = false until a real DSSE signs.
  - Λ = Conjecture 1 -> lambda_axis.advisory = true always; the anomaly axis is
    advisory, never "proven trust" / never folded into locked-proven=8.
  - confidence.label = "ESTIMATE" always.
  - sovereign = true (own-metal, no network in this module).

Attribution: the in-toto Statement v1 receipt shape is DELEGATED to the shared
szl_receipt.attest library (single source of truth; szl_mechanics/szl_pinn are
sibling emitters that should converge there too). Engine lineage libs:
PyOD(BSD-2), Merlion(BSD-3), TODS(Apache-2),
tsod(MIT), GDN/GraGOD(MIT), PyGOD(BSD-2), python-sgp4(MIT). alibi-detect(BSL-1.1)
EXCLUDED. True Anomaly Mosaic = inspiration only (public descriptions); no code.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

import numpy as np

from szl_mosaic_core import SZLMosaicCore, hash_inputs
from szl_confidence import conformal_band, pac_bayes_catoni

LIB_LINEAGE = ["pyod-BSD2", "merlion-BSD3", "tods-Apache2", "tsod-MIT",
               "gdn-MIT", "gragod-MIT", "pygod-BSD2", "sgp4-MIT"]

# Stage labels derived from True Anomaly's public SDA four-function decomposition.
VALID_STAGES = {"DTID", "CHARACTERIZE", "TWA", "FUSE"}


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# The in-toto Statement v1 shape emitted here is DELEGATED to the shared
# szl_receipt.attest library — the single source of truth for this shape across
# the SZL substrate (gim, szl-energy-attest, szl-router already build on it).
# We do NOT re-implement the shape; szl_mechanics/szl_pinn are sibling emitters
# that should also converge here. See CONSOLIDATION notes / szl-receipt v0.2.0.
PREDICATE_TYPE = "https://szlholdings.com/attestations/sda-anomaly/v1"


def _build_statement_local(track_id: str, subject_digest: str,
                           predicate: dict) -> dict:
    """Byte-identical fallback for szl_receipt.attest.build_statement.

    Used ONLY when the shared szl-receipt library is not installed. The import
    is lazy so this package stays zero-hard-dependency and NON-BREAKING; the
    fallback is a thin mirror of the canonical shape, never an independent copy.
    """
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": track_id, "digest": {"sha256": subject_digest}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": predicate,
    }


def make_intoto_receipt(track_id: str, subject_digest: str,
                        model_hash: str, seed: int,
                        anomaly_score: float) -> dict:
    """Build the in-toto Statement v1 receipt.

    The domain-specific predicate is assembled here; the in-toto Statement v1
    ENVELOPE shape is DELEGATED to szl_receipt.attest.build_statement (the shared
    single source of truth) via a lazy import, with a byte-identical local
    fallback if szl-receipt is absent. Output is unchanged either way.

    Honest: predicate.verified = False (UNSIGNED). predicate.sovereign = True.
    """
    predicate = {
        "engine": "khipu-sda-core",
        "model_hash": model_hash,
        "seed": seed,
        "anomaly_score": float(anomaly_score),
        "lib_lineage": LIB_LINEAGE,
        "inspiration": "True Anomaly Mosaic (public descriptions only; no code)",
        "doctrine": "v11",
        "verified": False,        # UNSIGNED until a real DSSE signs downstream
        "sovereign": True,        # own-metal, 0 CDN
    }
    try:
        from szl_receipt.attest import build_statement as _shared_build_statement
    except Exception:
        return _build_statement_local(track_id, subject_digest, predicate)
    return _shared_build_statement(
        subject_name=track_id,
        subject_digest=subject_digest,
        predicate=predicate,
        predicate_type=PREDICATE_TYPE,
    )


def lambda_verdict(score: float, allow_thr: float = 0.35,
                   deny_thr: float = 0.65) -> str:
    if score < allow_thr:
        return "allow"
    if score < deny_thr:
        return "advisory"
    return "deny"


def evaluate(track: dict, core: SZLMosaicCore,
             calib_scores: np.ndarray,
             stage: str = "TWA",
             confidence_method: str = "conformal",
             contamination_rate: float = 0.05,
             n_calib: Optional[int] = None) -> dict:
    """Evaluate ONE track and return the FROZEN §3.0 SDA verdict envelope.

    Parameters
    ----------
    track : dict with keys:
        "track_id" : str
        "features" : array-like (1 x F) multivariate feature vector (the channels
                     the `core` was fit on, e.g. speed/rcs/accel/heading-rate)
        "tle"      : optional (line1, line2) — if present, sgp4 hash is recorded
    core : a FITTED SZLMosaicCore
    calib_scores : combined ensemble scores on normal/calibration data (for the
                   conformal band + PAC-Bayes n).
    stage : one of DTID|CHARACTERIZE|TWA|FUSE.
    confidence_method : "conformal" or "PAC-Bayes".

    HONEST: emits an UNSIGNED receipt with a pointer to where signing happens.
    """
    assert stage in VALID_STAGES, f"stage must be one of {VALID_STAGES}"
    track_id = str(track.get("track_id", "TRK-UNKNOWN"))
    feats = np.atleast_2d(np.asarray(track["features"], dtype=float))

    res = core.score(feats)
    score = float(res["combined"][0])
    comps = {k: float(v[0]) for k, v in res["components"].items()}

    # confidence band (honest, ESTIMATE)
    if confidence_method == "PAC-Bayes":
        n = int(n_calib if n_calib is not None else len(calib_scores))
        conf = pac_bayes_catoni(empirical_rate=contamination_rate, n=n)
    else:
        conf = conformal_band(calib_scores, score, alpha=0.1)

    # sgp4 block (null for air/maritime; hash present for orbital tracks)
    sgp4_block = None
    if track.get("tle"):
        l1, l2 = track["tle"]
        sgp4_block = {"tle_hash": "sha256:" + _sha256_hex((l1 + l2).encode()),
                      "propagated": True}

    subject_digest = hash_inputs(feats)
    model_hash = _sha256_hex(("khipu-sda-core|" + "|".join(LIB_LINEAGE)).encode())
    receipt = make_intoto_receipt(track_id, subject_digest, model_hash,
                                  seed=core.seed, anomaly_score=score)

    envelope = {
        "track_id": track_id,
        "stage": stage,
        "anomaly_score": round(score, 6),
        "lambda_axis": {"name": "anomaly_twa", "value": round(score, 6),
                        "advisory": True,   # Λ = Conjecture 1, ALWAYS advisory
                        "verdict": lambda_verdict(score, core.allow_thr, core.deny_thr)},
        "confidence": conf,
        "component_scores": comps,
        "sgp4": sgp4_block,
        "receipt": receipt,
        "_signing": {
            "status": "UNSIGNED",   # honest: structural-only until real DSSE signs
            "signed_by": "szl_lake/khipu DSSE (Ed25519/P-256)",
            "note": ("UNSIGNED is STRUCTURAL-ONLY at a11oy verify-api; never a "
                     "false 'verified/green'. No fabricated signatures, ever."),
        },
        "_walltime_s": None,  # filled by caller-timed wrapper below
    }
    return envelope


def evaluate_timed(track: dict, core: SZLMosaicCore, calib_scores: np.ndarray,
                   **kw) -> dict:
    t0 = time.time()
    env = evaluate(track, core, calib_scores, **kw)
    env["_walltime_s"] = round(time.time() - t0, 6)
    return env


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, size=(300, 6))
    core = SZLMosaicCore().fit(Xtr)
    calib = core._combined_scores(Xtr)

    # air/maritime track (no TLE)
    anomalous = rng.normal(0, 1, size=(1, 6)); anomalous[0, 0] += 9.0
    env = evaluate_timed({"track_id": "TRK-0001", "features": anomalous},
                         core, calib, stage="TWA", confidence_method="conformal")
    print(json.dumps(env, indent=2))

    # orbital track (with TLE) using PAC-Bayes confidence
    env2 = evaluate_timed(
        {"track_id": "SAT-25544", "features": anomalous,
         "tle": ("1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
                 "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482")},
        core, calib, stage="DTID", confidence_method="PAC-Bayes")
    print(json.dumps(env2, indent=2))
