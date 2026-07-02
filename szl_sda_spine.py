"""
szl_sda_spine.py — fold the SDA verdict envelope and the Khipu witness-quorum
receipt onto the ORG-CANONICAL szl-receipt spine (an emit_receipt-style binding).

WHY THIS EXISTS
  `szl_sda_envelope.evaluate_timed()` already emits an honest in-toto Statement v1
  receipt (UNSIGNED, verified=false) and `szl_witness.witness_quorum()` already
  emits an honest Khipu-style quorum receipt. NEITHER, however, is bound onto the
  single org-canonical `szl-receipt` shape that every other SZL decision producer
  (a11oy, yarqa, governed-inference-meter, khipu-consensus, ...) folds onto. This
  module adds that emit_receipt-style binding so an SDA verdict / a witnessed
  detection becomes a first-class receipt on the SAME spine — WITHOUT touching or
  breaking the existing in-toto / quorum paths (this is strictly ADDITIVE).

An emit_receipt-style binding carries, in ONE canonical record:
  * subject        — the track id (SDA) or the witness-round id (quorum),
  * input_digest   — SHA-256 over the canonical INPUTS (the evidence/features
                     digest + track/stage/model/seed for SDA; the per-witness
                     independent re-scores + base band for the quorum),
  * output_digest  — SHA-256 over the agreed VALUE (the verdict/band for SDA;
                     the agreed band + witnessed flag for the quorum),
  * policy_id      — the governing policy id this receipt is bound under,
  * energy         — the literal string "UNAVAILABLE" (this package measures NO
                     joules; a joule is NEVER fabricated).

It invents NO new receipt shape and re-implements NO cryptography: it reuses
`szl_receipt.Receipt` for the canonical body + digest and
`szl_receipt.sign_receipt` / `szl_receipt.verify_receipt` for the DSSE envelope.
The shared library is the ONE source of truth for canonicalization and signing.

HONESTY (Doctrine v11, binding):
  * The receipt is EVIDENCE binding a decision (subject+input+output+policy+
    energy), NOT a proof the verdict/agreed band is correct. Re-deriving a digest
    is evidence the exact inputs/outputs are bound, not a re-run of the detector.
  * Λ = Conjecture 1 -> the anomaly axis is advisory ALWAYS, never proven trust.
  * Khipu BFT = Conjecture 2 (proposed, not proven) -> a witnessed verdict is a
    stronger ADVISORY only, never a proven-safe consensus.
  * Keyless => UNSIGNED-honest (signed=False); a signature is NEVER faked.
  * Energy is the literal "UNAVAILABLE" (joules=None); a joule is NEVER fabricated.

`szl_receipt` (v0.2.0) is an OPTIONAL dependency imported lazily, so importing
this module never requires it. Producing a canonical receipt DOES require it; its
absence raises :class:`SpineUnavailable` rather than fabricating a receipt.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from szl_witness import CONJECTURE_2_NOTE

# Canonical receipt kinds + PCGI schemas for the two folds.
RECEIPT_KIND_SDA = "khipu-sda-anomaly"
RECEIPT_KIND_WITNESS = "khipu-sda-witness-quorum"
PCGI_SCHEMA_SDA = "szl.pcgi.receipt/khipu-sda-anomaly/v1"
PCGI_SCHEMA_WITNESS = "szl.pcgi.receipt/khipu-sda-witness-quorum/v1"

# Governing policy ids (the policies these receipts are bound under).
DEFAULT_POLICY_ID_SDA = "szl.pcgi.policy/khipu-sda-anomaly/v1"
DEFAULT_POLICY_ID_WITNESS = "szl.pcgi.policy/khipu-bft-quorum/v1"

# Default logical signing-authority label stamped onto the envelope.
DEFAULT_ORGAN = "khipu-sda-core"

# Honest sentinel for energy that was not measured (never a fabricated joule).
ENERGY_UNAVAILABLE = "UNAVAILABLE"

# Λ = Conjecture 1: the anomaly axis is advisory, never proven trust.
LAMBDA_ADVISORY_NOTE = (
    "anomaly axis is advisory (Λ = Conjecture 1), never proven trust, never "
    "folded into locked-proven"
)


class SpineUnavailable(RuntimeError):
    """Raised when the shared ``szl_receipt`` library is not importable.

    Callers MUST treat this as "no canonical receipt here", never as a reason to
    fabricate a receipt or duplicate the library's shapes locally.
    """


def _require_szl_receipt():
    """Lazily import the shared ``szl_receipt`` library; fail honestly if absent."""
    try:
        import szl_receipt  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without the lib
        raise SpineUnavailable(
            "szl_receipt (v0.2.0) is not installed; install "
            "`szl-receipt @ git+https://github.com/szl-holdings/szl-receipt.git"
            "@v0.2.0` (already pinned in requirements.txt) to fold SDA/witness "
            "decisions onto the canonical szl-receipt spine. Refusing to "
            "duplicate the shared receipt shapes."
        ) from exc
    return szl_receipt


def _as_dict(obj: Any) -> dict[str, Any]:
    """Accept an object with ``to_dict()`` (e.g. a QuorumReceipt) or a plain dict."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(obj)


def _digest(body: dict[str, Any]) -> str:
    """SHA-256 hex over the shared canonical JSON of ``body``.

    Uses ``szl_receipt.Receipt.digest`` (SHA-256 over the library's canonical
    JSON) so the digest is byte-for-byte the same primitive that binds every
    other SZL receipt — nothing is re-implemented here.
    """
    szl_receipt = _require_szl_receipt()
    return szl_receipt.Receipt(kind="_digest", body=dict(body)).digest()


# --- SDA verdict envelope fold --------------------------------------------

def sda_input(envelope: dict[str, Any]) -> dict[str, Any]:
    """The canonical INPUTS (the evidence/features) an SDA verdict depends on.

    Binds the feature/evidence digest (the ``hash_inputs`` of the multivariate
    feature vector, carried on the in-toto receipt subject) together with the
    track id, stage, model hash and seed. Re-deriving this reproduces
    ``input_digest``.
    """
    r = envelope.get("receipt") or {}
    subjects = r.get("subject") or [{}]
    feature_digest = (subjects[0].get("digest") or {}).get("sha256")
    pred = r.get("predicate") or {}
    return {
        "track_id": envelope.get("track_id"),
        "stage": envelope.get("stage"),
        "feature_digest": feature_digest,
        "model_hash": pred.get("model_hash"),
        "seed": pred.get("seed"),
    }


def sda_output(envelope: dict[str, Any]) -> dict[str, Any]:
    """The canonical agreed VALUE (the verdict / advisory band) of an SDA envelope."""
    la = envelope.get("lambda_axis") or {}
    conf = envelope.get("confidence") or {}
    return {
        "anomaly_score": envelope.get("anomaly_score"),
        "verdict": la.get("verdict"),
        "advisory": la.get("advisory"),
        "confidence": {
            "lo": conf.get("lo"),
            "hi": conf.get("hi"),
            "label": conf.get("label"),
        },
    }


def sda_input_digest(envelope: dict[str, Any]) -> str:
    """SHA-256 hex over :func:`sda_input`."""
    return _digest(sda_input(envelope))


def sda_output_digest(envelope: dict[str, Any]) -> str:
    """SHA-256 hex over :func:`sda_output`."""
    return _digest(sda_output(envelope))


def build_sda_receipt_body(
    envelope: dict[str, Any],
    *,
    policy_id: str = DEFAULT_POLICY_ID_SDA,
) -> dict[str, Any]:
    """Assemble the canonical PCGI receipt body for one SDA verdict.

    Binds the spine tuple — subject (track id), input digest (the evidence/
    features), output digest (the verdict/band), governing policy id, and energy
    (honest ``UNAVAILABLE`` — this package measures no joules). The body is
    deterministic for a fixed input envelope, so the same envelope always yields
    byte-identical canonical JSON and the same digest.
    """
    la = envelope.get("lambda_axis") or {}
    return {
        "schema": PCGI_SCHEMA_SDA,
        "kind": RECEIPT_KIND_SDA,
        "subject": {
            "type": "khipu-sda-track",
            "track_id": envelope.get("track_id"),  # the track/subject id
            "stage": envelope.get("stage"),
        },
        "input_digest": sda_input_digest(envelope),
        "output_digest": sda_output_digest(envelope),
        "policy_id": policy_id,
        "energy": {
            "status": ENERGY_UNAVAILABLE,
            "joules": None,
            "reason": (
                "khipu-sda-core measures no joules here; energy reported "
                "UNAVAILABLE, never fabricated."
            ),
        },
        "verdict": {
            "anomaly_score": envelope.get("anomaly_score"),
            "band": la.get("verdict"),
            "advisory": la.get("advisory"),
        },
        "honesty": {
            "asserts": "integrity/reproducibility of the verdict binding, NOT correctness",
            "receipt_is": (
                "evidence trail binding this verdict (subject+input+output+"
                "policy+energy), not a proof the verdict is correct"
            ),
            "lambda_axis": LAMBDA_ADVISORY_NOTE,
        },
    }


def sda_receipt_body_digest(
    envelope: dict[str, Any],
    *,
    policy_id: str = DEFAULT_POLICY_ID_SDA,
) -> str:
    """Independently (re-)derive the SDA receipt's content digest."""
    return _digest(build_sda_receipt_body(envelope, policy_id=policy_id))


def emit_sda_receipt(
    envelope: dict[str, Any],
    *,
    private_key_pem: Optional[Union[str, bytes]] = None,
    policy_id: str = DEFAULT_POLICY_ID_SDA,
    organ: str = DEFAULT_ORGAN,
    keyid: str = "",
) -> dict[str, Any]:
    """Emit ONE canonical szl-receipt for an SDA verdict (the PCGI spine).

    Wraps :func:`build_sda_receipt_body` in a shared :class:`szl_receipt.Receipt`
    and signs it via :func:`szl_receipt.sign_receipt` (DSSE/ECDSA-P256-SHA256).
    With a PEM ECDSA-P256 ``private_key_pem`` the envelope is signed; keyless it
    is UNSIGNED-honest (``signed=False``) — never a fabricated signature.

    The returned envelope binds subject (track id) + input digest (evidence/
    features) + output digest (verdict/band) + governing policy id + honest
    ``UNAVAILABLE`` energy. It is an EVIDENCE trail, not a proof of correctness.
    """
    szl_receipt = _require_szl_receipt()
    body = build_sda_receipt_body(envelope, policy_id=policy_id)
    return szl_receipt.sign_receipt(
        szl_receipt.Receipt(kind=RECEIPT_KIND_SDA, body=body),
        private_key_pem,
        organ=organ,
        keyid=keyid,
    )


def verify_sda_receipt(
    envelope_receipt: dict[str, Any],
    *,
    public_key_pem: Optional[Union[str, bytes]] = None,
    source_envelope: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Verify a canonical SDA receipt (and optionally rebind it to its envelope).

    Delegates the cryptographic check to :func:`szl_receipt.verify_receipt`
    (keyless envelopes honestly return ``(False, "unsigned-honest")``; a tampered
    payload returns a failure). When ``source_envelope`` is supplied, additionally
    confirms the signed body's ``input_digest`` / ``output_digest`` re-derive from
    that envelope — so any post-hoc edit flips a digest and fails the rebind.
    """
    szl_receipt = _require_szl_receipt()
    ok, detail = szl_receipt.verify_receipt(envelope_receipt, public_key_pem)
    if not ok:
        return ok, detail
    if source_envelope is not None:
        import base64
        import json

        try:
            body = json.loads(
                base64.b64decode(envelope_receipt["payload"]).decode("utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"payload decode error: {exc}"
        if body.get("input_digest") != sda_input_digest(source_envelope):
            return False, "input-digest-rebind-mismatch"
        if body.get("output_digest") != sda_output_digest(source_envelope):
            return False, "output-digest-rebind-mismatch"
    return True, "ok"


# --- Khipu witness-quorum receipt fold ------------------------------------

def witness_input(receipt: Any) -> dict[str, Any]:
    """The canonical INPUTS (the evidence: independent per-witness re-scores).

    Binds the base score/band and the per-witness independent re-scores that the
    quorum outcome depends on. Re-deriving this reproduces ``input_digest``.
    """
    r = _as_dict(receipt)
    return {
        "subject": r.get("subject"),
        "base_score": r.get("base_score"),
        "base_band": r.get("base_band"),
        "quorum_rule": r.get("quorum_rule"),
        "witnesses": r.get("witnesses"),
    }


def witness_output(receipt: Any) -> dict[str, Any]:
    """The canonical agreed VALUE (the agreed band + witnessed flag)."""
    r = _as_dict(receipt)
    return {
        "agreed_band": r.get("agreed_band"),
        "agree_count": r.get("agree_count"),
        "witnessed": r.get("witnessed"),
        "agreed_score_band": r.get("agreed_score_band"),
    }


def witness_input_digest(receipt: Any) -> str:
    """SHA-256 hex over :func:`witness_input`."""
    return _digest(witness_input(receipt))


def witness_output_digest(receipt: Any) -> str:
    """SHA-256 hex over :func:`witness_output`."""
    return _digest(witness_output(receipt))


def build_witness_receipt_body(
    receipt: Any,
    *,
    policy_id: str = DEFAULT_POLICY_ID_WITNESS,
) -> dict[str, Any]:
    """Assemble the canonical PCGI receipt body for one witness-quorum decision.

    Binds the spine tuple — subject (witness-round id), input digest (the
    independent per-witness re-scores), output digest (the agreed band), governing
    policy id, and energy (honest ``UNAVAILABLE``). Khipu BFT = Conjecture 2
    (proposed, not proven): a witnessed verdict is a stronger ADVISORY only.
    """
    r = _as_dict(receipt)
    return {
        "schema": PCGI_SCHEMA_WITNESS,
        "kind": RECEIPT_KIND_WITNESS,
        "subject": {
            "type": "khipu-sda-witness-round",
            "round_id": r.get("subject"),  # the witness-round id
        },
        "input_digest": witness_input_digest(r),
        "output_digest": witness_output_digest(r),
        "policy_id": policy_id,
        "energy": {
            "status": ENERGY_UNAVAILABLE,
            "joules": None,
            "reason": (
                "khipu quorum measures no joules here; energy reported "
                "UNAVAILABLE, never fabricated."
            ),
        },
        "quorum": {
            "rule": r.get("quorum_rule"),
            "agreed_band": r.get("agreed_band"),
            "agree_count": r.get("agree_count"),
            "witnessed": r.get("witnessed"),
        },
        "honesty": {
            "asserts": "integrity/reproducibility of the quorum, NOT correctness",
            "receipt_is": (
                "evidence trail binding this witnessed verdict (subject+input+"
                "output+policy+energy), not a proof the agreed band is correct"
            ),
            "consensus": CONJECTURE_2_NOTE,
            "lambda_axis": LAMBDA_ADVISORY_NOTE,
        },
    }


def witness_receipt_body_digest(
    receipt: Any,
    *,
    policy_id: str = DEFAULT_POLICY_ID_WITNESS,
) -> str:
    """Independently (re-)derive the witness receipt's content digest."""
    return _digest(build_witness_receipt_body(receipt, policy_id=policy_id))


def emit_witness_receipt(
    receipt: Any,
    *,
    private_key_pem: Optional[Union[str, bytes]] = None,
    policy_id: str = DEFAULT_POLICY_ID_WITNESS,
    organ: str = DEFAULT_ORGAN,
    keyid: str = "",
) -> dict[str, Any]:
    """Emit ONE canonical szl-receipt for a witness-quorum decision (PCGI spine).

    Wraps :func:`build_witness_receipt_body` in a shared
    :class:`szl_receipt.Receipt` and signs it via
    :func:`szl_receipt.sign_receipt`. Keyless => UNSIGNED-honest
    (``signed=False``) — never a fabricated signature. The receipt is an EVIDENCE
    trail; Khipu BFT stays Conjecture 2 (a witnessed verdict is a stronger
    ADVISORY only, never proven trust).
    """
    szl_receipt = _require_szl_receipt()
    body = build_witness_receipt_body(receipt, policy_id=policy_id)
    return szl_receipt.sign_receipt(
        szl_receipt.Receipt(kind=RECEIPT_KIND_WITNESS, body=body),
        private_key_pem,
        organ=organ,
        keyid=keyid,
    )


def verify_witness_receipt(
    envelope_receipt: dict[str, Any],
    *,
    public_key_pem: Optional[Union[str, bytes]] = None,
    receipt: Optional[Any] = None,
) -> tuple[bool, str]:
    """Verify a canonical witness receipt (and optionally rebind it).

    Delegates to :func:`szl_receipt.verify_receipt` (keyless => honest
    ``(False, "unsigned-honest")``). When ``receipt`` is supplied, additionally
    confirms the signed body's digests re-derive from that quorum receipt.
    """
    szl_receipt = _require_szl_receipt()
    ok, detail = szl_receipt.verify_receipt(envelope_receipt, public_key_pem)
    if not ok:
        return ok, detail
    if receipt is not None:
        import base64
        import json

        try:
            body = json.loads(
                base64.b64decode(envelope_receipt["payload"]).decode("utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"payload decode error: {exc}"
        if body.get("input_digest") != witness_input_digest(receipt):
            return False, "input-digest-rebind-mismatch"
        if body.get("output_digest") != witness_output_digest(receipt):
            return False, "output-digest-rebind-mismatch"
    return True, "ok"
