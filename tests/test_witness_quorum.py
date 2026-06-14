"""
test_witness_quorum.py — REAL assertions on multi-witness verified detection.

No mocks: 4 independent witnesses really re-score each detection. Tests assert
the 3-of-4 quorum behaves honestly — a strong detection is witnessed in the deny
band, a clearly-normal detection is witnessed in the allow band, a genuinely
split (bimodal-evidence) detection FAILS the quorum, and the honest Conjecture-2
posture (advisory, verified=false, no fabricated signature) is preserved.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from szl_witness import (witness_quorum, witness_envelope, verdict_band,
                         QuorumReceipt, DEFAULT_QUORUM, DEFAULT_N_WITNESSES)


def test_quorum_rule_is_3_of_4():
    assert (DEFAULT_QUORUM, DEFAULT_N_WITNESSES) == (3, 4)


def test_strong_detection_is_witnessed_deny():
    """A high-score detection with strong evidence reaches a deny-band quorum."""
    r = witness_quorum("TRK-STRONG", base_score=0.92,
                       evidence=np.array([0.95, 0.9, 0.97, 0.93]))
    assert r.witnessed is True
    assert r.agreed_band == "deny"
    assert r.agree_count >= 3
    # agreed score band is well-formed and inside [0,1]
    assert 0.0 <= r.agreed_score_lo <= r.agreed_score_hi <= 1.0


def test_normal_detection_is_witnessed_allow():
    """A clearly-normal detection reaches an allow-band quorum."""
    r = witness_quorum("TRK-NORMAL", base_score=0.05,
                       evidence=np.array([0.04, 0.06, 0.03, 0.05]))
    assert r.witnessed is True
    assert r.agreed_band == "allow"
    assert r.agree_count >= 3


def test_split_detection_can_fail_quorum():
    """A genuinely ambiguous detection (bimodal evidence near a band boundary)
    can FAIL the quorum — the gate is real, not always-True. We assert that at
    least one seed produces a no-quorum (split) outcome."""
    ev = np.array([0.1, 0.2, 0.9, 0.95])   # high-variance bootstrap
    outcomes = [witness_quorum("T", base_score=0.31, evidence=ev,
                               base_seed=s).witnessed
                for s in (3000, 3100, 3200, 3300)]
    assert any(o is False for o in outcomes), \
        "expected at least one split (no-quorum) outcome on bimodal evidence"


def test_witnesses_are_independent_and_real():
    """Each witness uses a distinct seed and produces a real (non-identical)
    re-score — they are not echoing one fixed value."""
    r = witness_quorum("TRK", base_score=0.5,
                       evidence=np.array([0.3, 0.5, 0.7, 0.4]))
    seeds = [w.seed for w in r.witnesses]
    scores = [w.score for w in r.witnesses]
    assert len(set(seeds)) == 4                       # distinct seeds
    assert len(set(round(s, 6) for s in scores)) > 1  # not all identical
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_receipt_is_honest_conjecture2():
    """Quorum receipt must stay advisory, verified=false, sovereign, and carry
    the Conjecture-2 (proposed, not proven) posture — never a proven/signed claim."""
    r = witness_quorum("TRK", base_score=0.8, evidence=np.array([0.8, 0.82, 0.79]))
    d = r.to_dict()
    assert d["advisory"] is True                      # Λ = Conjecture 1
    assert d["verified"] is False                     # not a proven claim
    assert d["sovereign"] is True
    assert "Conjecture 2" in d["conjecture"]
    assert "not proven" in d["note"].lower()
    # digest is a sha256 handle, NOT a fabricated signature
    assert d["digest"].startswith("sha256:")
    assert d["quorum_rule"] == "3-of-4"


def test_witness_envelope_is_nondestructive_and_no_false_green():
    """Attaching a quorum to a §3.0-style envelope adds _witness without flipping
    a false 'verified/green' signing state."""
    env = {"track_id": "TRK-9", "anomaly_score": 0.85,
           "component_scores": {"iforest": 0.9, "autoencoder": 0.8,
                                "robust_zscore": 0.85},
           "_signing": {"status": "UNSIGNED"}}
    out = witness_envelope(env)
    assert "_witness" in out
    assert out["_signing"]["status"] == "UNSIGNED"    # unchanged, no false green
    assert out["_witness"]["verified"] is False
    assert out["track_id"] == "TRK-9"                 # original preserved


def test_verdict_band_boundaries():
    assert verdict_band(0.30) == "allow"
    assert verdict_band(0.50) == "advisory"
    assert verdict_band(0.90) == "deny"


if __name__ == "__main__":
    test_quorum_rule_is_3_of_4()
    test_strong_detection_is_witnessed_deny()
    test_normal_detection_is_witnessed_allow()
    test_split_detection_can_fail_quorum()
    test_witnesses_are_independent_and_real()
    test_receipt_is_honest_conjecture2()
    test_witness_envelope_is_nondestructive_and_no_false_green()
    test_verdict_band_boundaries()
    print("test_witness_quorum: PASS (real 3-of-4 quorum; honest Conjecture-2)")
