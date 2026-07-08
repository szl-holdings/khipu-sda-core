"""
test_witness_dedup.py — adversarial: a single node cannot stuff the quorum.

Real assertions (no mocks) that when witnesses are supplied EXTERNALLY, the
quorum counts each witness_id at most once (input hygiene). A single node that
echoes its opinion K times is counted ONCE, so it cannot single-handedly
manufacture a quorum — while a NAIVE per-opinion tally (kept only as an explicit
demonstrator) would be fooled. Honest posture (Conjecture 2, verified=false) is
preserved throughout.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from szl_witness import (WitnessOpinion, witness_quorum, dedup_witness_opinions,
                         naive_stuffed_agree_count)

EV = np.array([0.8, 0.82, 0.79])


def _op(wid, band, score=0.8, seed=0):
    return WitnessOpinion(witness_id=wid, seed=seed, score=score, band=band)


def test_dedup_keeps_first_per_id():
    ops = [_op("a", "deny"), _op("a", "deny"), _op("a", "deny"), _op("b", "allow")]
    kept = dedup_witness_opinions(ops)
    assert [w.witness_id for w in kept] == ["a", "b"]      # distinct only, order kept


def test_single_node_cannot_stuff_quorum():
    """One malicious node echoing 'deny' 5x + one honest 'allow' node → only 2
    DISTINCT witnesses, so the 3-of-4 quorum is NOT reached by the stuffer."""
    stuffed = [_op("evil", "deny") for _ in range(5)] + [_op("honest", "allow")]
    r = witness_quorum("TRK-STUFF", base_score=0.8, evidence=EV,
                       quorum=3, external_opinions=stuffed)
    assert r.n_opinions_submitted == 6                     # 6 opinions submitted
    assert r.n_witnesses == 2                              # deduped to 2 distinct
    assert r.witnessed is False                            # stuffing failed
    assert r.agree_count < 3


def test_naive_tally_would_have_been_fooled():
    """The demonstrator proves the attack is real: a naive per-opinion count of
    the stuffed 'deny' opinions clears the quorum threshold, but the shipped
    (deduped) path does not."""
    stuffed = [_op("evil", "deny") for _ in range(5)] + [_op("honest", "allow")]
    naive = naive_stuffed_agree_count(stuffed, "deny")
    assert naive == 5 and naive >= 3                       # naive: fooled
    r = witness_quorum("TRK-STUFF", base_score=0.8, evidence=EV,
                       quorum=3, external_opinions=stuffed)
    assert r.witnessed is False                            # hardened: not fooled


def test_distinct_witnesses_still_reach_quorum():
    """Three genuinely distinct witnesses agreeing on 'deny' DO reach quorum."""
    honest = [_op("w0", "deny"), _op("w1", "deny"), _op("w2", "deny")]
    r = witness_quorum("TRK-OK", base_score=0.8, evidence=EV,
                       quorum=3, external_opinions=honest)
    assert r.n_witnesses == 3
    assert r.witnessed is True
    assert r.agreed_band == "deny"


def test_empty_external_opinions_is_not_witnessed():
    r = witness_quorum("TRK-EMPTY", base_score=0.8, evidence=EV,
                       quorum=3, external_opinions=[])
    assert r.witnessed is False
    assert r.n_witnesses == 0
    assert r.agreed_band is None


def test_external_path_stays_honest_conjecture2():
    honest = [_op("w0", "deny"), _op("w1", "deny"), _op("w2", "deny")]
    d = witness_quorum("TRK", base_score=0.8, evidence=EV,
                       quorum=3, external_opinions=honest).to_dict()
    assert d["verified"] is False
    assert d["advisory"] is True
    assert "Conjecture 2" in d["conjecture"]
    assert d["distinct_witnesses"] == 3
    assert d["n_opinions_submitted"] == 3
    assert d["digest"].startswith("sha256:")


if __name__ == "__main__":
    test_dedup_keeps_first_per_id()
    test_single_node_cannot_stuff_quorum()
    test_naive_tally_would_have_been_fooled()
    test_distinct_witnesses_still_reach_quorum()
    test_empty_external_opinions_is_not_witnessed()
    test_external_path_stays_honest_conjecture2()
    print("test_witness_dedup: PASS (single node cannot stuff the quorum)")
