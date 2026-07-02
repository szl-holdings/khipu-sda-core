"""
test_sda_spine.py — focused tests for the emit_receipt-style spine fold that
binds an SDA verdict envelope and a Khipu witness-quorum receipt onto the
org-canonical szl-receipt shape.

REAL computation only (no mocks/stubs): a real SZLMosaicCore scores a real
feature vector, real independent witnesses re-score real evidence, and REAL
throwaway P-256 keys (never written to disk) exercise the signed path. The tests
assert the honest binding — subject (track / witness-round id), input digest of
the evidence/features, output digest of the verdict/agreed band, governing policy
id, energy == "UNAVAILABLE" (joules None, never fabricated), unsigned-honest when
keyless, a real signature verifies + rebinds, tamper is rejected, and the receipt
asserts integrity/reproducibility NOT correctness (Λ = Conjecture 1 advisory,
Khipu BFT = Conjecture 2 proposed-not-proven).

``szl_receipt`` is an optional dependency; if it is not importable these tests are
skipped rather than failing (the core SDA/witness paths never depend on it).
"""
import base64
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

szl_receipt = pytest.importorskip("szl_receipt")

from szl_mosaic_core import SZLMosaicCore  # noqa: E402
from szl_sda_envelope import evaluate_timed  # noqa: E402
from szl_witness import witness_quorum  # noqa: E402
from szl_sda_spine import (  # noqa: E402
    DEFAULT_POLICY_ID_SDA,
    DEFAULT_POLICY_ID_WITNESS,
    ENERGY_UNAVAILABLE,
    PCGI_SCHEMA_SDA,
    PCGI_SCHEMA_WITNESS,
    build_sda_receipt_body,
    build_witness_receipt_body,
    emit_sda_receipt,
    emit_witness_receipt,
    sda_input_digest,
    sda_output_digest,
    sda_receipt_body_digest,
    verify_sda_receipt,
    verify_witness_receipt,
    witness_input_digest,
    witness_output_digest,
    witness_receipt_body_digest,
)


def _fitted_core():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, size=(300, 6))
    core = SZLMosaicCore().fit(Xtr)
    calib = core._combined_scores(Xtr)
    return core, calib


def _sda_envelope():
    core, calib = _fitted_core()
    feats = np.zeros((1, 6))
    feats[0, 0] = 9.0
    return evaluate_timed({"track_id": "TRK-0404", "features": feats},
                          core, calib, stage="TWA")


# --- SDA verdict envelope fold --------------------------------------------

def test_sda_body_binds_the_pcgi_tuple_with_energy_unavailable():
    env = _sda_envelope()
    body = build_sda_receipt_body(env)

    assert body["schema"] == PCGI_SCHEMA_SDA
    assert body["subject"]["track_id"] == "TRK-0404"
    assert body["policy_id"] == DEFAULT_POLICY_ID_SDA
    assert body["input_digest"] == sda_input_digest(env)
    assert body["output_digest"] == sda_output_digest(env)

    # Energy is the honest sentinel — never a fabricated joule.
    assert body["energy"]["status"] == ENERGY_UNAVAILABLE
    assert body["energy"]["status"] == "UNAVAILABLE"
    assert body["energy"]["joules"] is None

    # Λ stays advisory (Conjecture 1); receipt = evidence, not correctness.
    assert body["verdict"]["advisory"] is True
    assert body["honesty"]["asserts"].endswith("NOT correctness")
    assert "Conjecture 1" in body["honesty"]["lambda_axis"]


def test_sda_body_is_deterministic_for_a_fixed_envelope():
    env = _sda_envelope()
    assert sda_receipt_body_digest(env) == sda_receipt_body_digest(dict(env))


def test_sda_keyless_is_unsigned_honest_never_faked():
    env = _sda_envelope()
    rcpt = emit_sda_receipt(env)
    assert rcpt["signed"] is False
    assert rcpt["signature"] == ""          # honest: no fabricated signature
    ok, detail = verify_sda_receipt(rcpt)
    assert ok is False
    assert detail == "unsigned-honest"


def test_sda_signed_receipt_verifies_and_rebinds():
    env = _sda_envelope()
    priv, pub = szl_receipt.generate_keypair()
    rcpt = emit_sda_receipt(env, private_key_pem=priv, organ="khipu-sda-core")
    assert rcpt["signed"] is True

    ok, detail = verify_sda_receipt(rcpt, public_key_pem=pub)
    assert ok, detail
    ok2, detail2 = verify_sda_receipt(rcpt, public_key_pem=pub, source_envelope=env)
    assert ok2, detail2


def test_sda_tamper_is_rejected():
    env = _sda_envelope()
    priv, pub = szl_receipt.generate_keypair()
    rcpt = emit_sda_receipt(env, private_key_pem=priv)

    payload = json.loads(base64.b64decode(rcpt["payload"]).decode("utf-8"))
    payload["policy_id"] = "attacker-swapped-policy"
    rcpt["payload"] = base64.b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode()

    ok, _ = verify_sda_receipt(rcpt, public_key_pem=pub)
    assert ok is False


def test_sda_rebind_fails_after_output_edit():
    env = _sda_envelope()
    priv, pub = szl_receipt.generate_keypair()
    rcpt = emit_sda_receipt(env, private_key_pem=priv)

    tampered = dict(env)
    tampered["lambda_axis"] = dict(env["lambda_axis"])
    tampered["lambda_axis"]["verdict"] = "allow-but-lied"
    ok, detail = verify_sda_receipt(rcpt, public_key_pem=pub, source_envelope=tampered)
    assert ok is False
    assert detail == "output-digest-rebind-mismatch"


# --- Khipu witness-quorum receipt fold ------------------------------------

def test_witness_body_binds_the_pcgi_tuple_with_energy_unavailable():
    q = witness_quorum("TRK-STRONG", base_score=0.92,
                       evidence=np.array([0.95, 0.9, 0.97, 0.93]))
    body = build_witness_receipt_body(q)

    assert body["schema"] == PCGI_SCHEMA_WITNESS
    assert body["subject"]["round_id"] == "TRK-STRONG"
    assert body["policy_id"] == DEFAULT_POLICY_ID_WITNESS
    assert body["input_digest"] == witness_input_digest(q)
    assert body["output_digest"] == witness_output_digest(q)

    # Energy is the honest sentinel — never a fabricated joule.
    assert body["energy"]["status"] == ENERGY_UNAVAILABLE
    assert body["energy"]["status"] == "UNAVAILABLE"
    assert body["energy"]["joules"] is None

    # Khipu BFT = Conjecture 2 (proposed, not proven); Λ advisory (Conjecture 1).
    assert body["quorum"]["agreed_band"] == "deny"
    assert body["honesty"]["asserts"].endswith("NOT correctness")
    assert "Conjecture 2" in body["honesty"]["consensus"]
    assert "not proven" in body["honesty"]["consensus"].lower()
    assert "Conjecture 1" in body["honesty"]["lambda_axis"]


def test_witness_body_is_deterministic_for_a_fixed_receipt():
    q = witness_quorum("TRK", base_score=0.8, evidence=np.array([0.8, 0.82, 0.79]))
    assert witness_receipt_body_digest(q) == witness_receipt_body_digest(q.to_dict())


def test_witness_keyless_is_unsigned_honest_never_faked():
    q = witness_quorum("TRK", base_score=0.8, evidence=np.array([0.8, 0.82, 0.79]))
    rcpt = emit_witness_receipt(q)
    assert rcpt["signed"] is False
    assert rcpt["signature"] == ""          # honest: no fabricated signature
    ok, detail = verify_witness_receipt(rcpt)
    assert ok is False
    assert detail == "unsigned-honest"


def test_witness_signed_receipt_verifies_and_rebinds():
    q = witness_quorum("TRK-STRONG", base_score=0.92,
                       evidence=np.array([0.95, 0.9, 0.97, 0.93]))
    priv, pub = szl_receipt.generate_keypair()
    rcpt = emit_witness_receipt(q, private_key_pem=priv)
    assert rcpt["signed"] is True

    ok, detail = verify_witness_receipt(rcpt, public_key_pem=pub)
    assert ok, detail
    ok2, detail2 = verify_witness_receipt(rcpt, public_key_pem=pub, receipt=q)
    assert ok2, detail2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
