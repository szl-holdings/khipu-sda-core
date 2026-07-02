"""
test_receipt_delegation.py — the in-toto Statement v1 receipt shape is
DELEGATED to the shared szl_receipt.attest library (the single source of truth),
not re-implemented locally. This proves the delegated statement is byte-identical
to a direct szl_receipt.attest.build_statement call, binds the subject digest,
and is honestly rejected on a wrong digest (never a false green).

Skips (does not fail) if szl-receipt is not installed, since make_intoto_receipt
has a byte-identical local fallback for the zero-dependency path.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

attest = pytest.importorskip("szl_receipt.attest")

from szl_sda_envelope import PREDICATE_TYPE, make_intoto_receipt


def test_receipt_is_built_by_shared_library():
    dg = "a" * 64
    r = make_intoto_receipt("TRK-DELEG", dg, model_hash="m", seed=0,
                            anomaly_score=0.5)
    # Byte-identical to a direct szl_receipt.attest.build_statement call: the
    # shape is owned by the shared library, not this package.
    expect = attest.build_statement(
        subject_name="TRK-DELEG",
        subject_digest=dg,
        predicate=r["predicate"],
        predicate_type=PREDICATE_TYPE,
    )
    assert r == expect
    assert r["_type"] == attest.IN_TOTO_STATEMENT_TYPE
    assert r["predicateType"].endswith("/sda-anomaly/v1")


def test_shared_verify_statement_binds_subject_and_type():
    dg = "b" * 64
    r = make_intoto_receipt("TRK-VERIFY", dg, model_hash="m", seed=1,
                            anomaly_score=0.9)
    ok, why = attest.verify_statement(
        r, expected_digest=dg, predicate_type=PREDICATE_TYPE)
    assert ok and why == "ok", (ok, why)
    # HONESTY: a wrong digest is rejected, never a false green.
    bad, _ = attest.verify_statement(r, expected_digest="deadbeef")
    assert bad is False
