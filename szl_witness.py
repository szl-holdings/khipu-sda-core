"""
szl_witness.py — MULTI-WITNESS VERIFIED DETECTION (Khipu-style BFT quorum).

================================ ATTRIBUTION ================================
CLEAN-ROOM STATEMENT
--------------------
Clean-room, SZL-native. The quorum mechanism is the textbook Byzantine-fault-
tolerant idea — require agreement of a super-majority of independent replicas
before a value is accepted — re-implemented minimally in numpy. No proprietary
consensus code copied. The N-of-M quorum threshold (here 3-of-4) is standard
distributed-systems practice; the SZL "Khipu" framing (a quipu-knot Merkle DAG
of attested records) is our own naming, not a borrowed implementation.
============================================================================

WHAT THIS DOES
  Take ONE detection/verdict (a score + Λ verdict from the engine) and have N
  *independent witness nodes* INDEPENDENTLY re-score it. Each witness sees the
  same evidence but carries its own deterministic perturbation (a different
  random seed → a different bootstrap/jitter of the evidence), simulating
  independent re-derivation rather than blind echoing. A Khipu-style **3-of-4
  quorum** must agree on the verdict band before the detection is marked
  "witnessed". We emit a QUORUM RECEIPT: which witnesses agreed, the agreed
  score band, and the honest posture.

HONEST POSTURE & SCOPE LABEL (Doctrine v11):
  *** Khipu BFT = CONJECTURE 2 (PROPOSED, NOT PROVEN). ***
  This is the ENGINEERING MECHANISM (an N-of-M agreement gate), NOT a proven-safe
  consensus protocol. We do NOT claim Byzantine safety/liveness proofs, partial-
  synchrony bounds, equivocation resistance, or sybil resistance. The witnesses
  here are SIMULATED independent re-scorers in one process — there is no network,
  no real replication, no adversary model exercised. A "witnessed" verdict is a
  stronger ADVISORY (more independent scorers concurred), NEVER "proven trust"
  and NEVER folded into locked-proven = 8. Λ stays Conjecture 1 (advisory).
  Sovereign / own-metal: pure local numpy, no network, 0 CDN. Numbers are REAL
  (each witness really re-scores) — nothing fabricated, no fabricated signatures.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Khipu BFT quorum: 3-of-4 (super-majority; tolerates 1 faulty/dissenting witness).
DEFAULT_N_WITNESSES = 4
DEFAULT_QUORUM = 3

CONJECTURE_2_NOTE = (
    "Khipu BFT = Conjecture 2 (PROPOSED, NOT PROVEN). This is an N-of-M agreement "
    "GATE (engineering mechanism), not a proven-safe consensus: no Byzantine "
    "safety/liveness proof, no adversary/sybil model exercised, witnesses are "
    "SIMULATED independent re-scorers in-process (no network/replication). A "
    "'witnessed' verdict is a stronger ADVISORY, never proven trust, never "
    "locked-proven. Λ remains Conjecture 1 (advisory)."
)


def verdict_band(score: float, allow_thr: float = 0.35,
                 deny_thr: float = 0.65) -> str:
    """Map a score to the Λ advisory band (same gate as the envelope)."""
    if score < allow_thr:
        return "allow"
    if score < deny_thr:
        return "advisory"
    return "deny"


@dataclass
class WitnessOpinion:
    """One witness node's independent re-score of a detection."""
    witness_id: str
    seed: int
    score: float
    band: str            # allow | advisory | deny


@dataclass
class QuorumReceipt:
    """Honest quorum receipt for a multi-witness verified detection.

    witnessed = (count of witnesses in the modal band) >= quorum. This is a
    stronger ADVISORY (Conjecture 2), NEVER proven trust.
    """
    subject: str
    base_score: float
    base_band: str
    n_witnesses: int
    quorum: int
    agreed_band: Optional[str]          # modal band (None if no clear mode)
    agree_count: int                    # witnesses in the agreed band
    witnessed: bool                     # quorum reached on agreed band
    agreed_score_lo: float              # min score among AGREEING witnesses
    agreed_score_hi: float              # max score among AGREEING witnesses
    agreed_score_mean: float
    witnesses: list = field(default_factory=list)   # list[WitnessOpinion]
    digest: str = ""                    # sha256 over the agreeing witness opinions
    advisory: bool = True               # Λ stays advisory always
    conjecture: str = "Conjecture 2 (proposed, not proven)"
    note: str = CONJECTURE_2_NOTE
    _walltime_s: Optional[float] = None
    n_opinions_submitted: Optional[int] = None   # set for external opinions;
    #   > distinct_witnesses ⇒ duplicate witness_ids were deduped (input hygiene)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "base_score": round(self.base_score, 6),
            "base_band": self.base_band,
            "quorum_rule": f"{self.quorum}-of-{self.n_witnesses}",
            "agreed_band": self.agreed_band,
            "agree_count": self.agree_count,
            "witnessed": self.witnessed,
            "distinct_witnesses": self.n_witnesses,
            "n_opinions_submitted": self.n_opinions_submitted,
            "agreed_score_band": {
                "lo": round(self.agreed_score_lo, 6),
                "hi": round(self.agreed_score_hi, 6),
                "mean": round(self.agreed_score_mean, 6),
            },
            "witnesses": [
                {"witness_id": w.witness_id, "seed": w.seed,
                 "score": round(w.score, 6), "band": w.band}
                for w in self.witnesses
            ],
            "digest": self.digest,
            "advisory": self.advisory,            # Λ = Conjecture 1
            "conjecture": self.conjecture,        # Khipu BFT = Conjecture 2
            "verified": False,                    # honest: NOT a proven/signed claim
            "sovereign": True,
            "note": self.note,
            "_walltime_s": self._walltime_s,
        }


def default_rescorer(evidence: np.ndarray, seed: int,
                     base_score: float, jitter: float = 0.04) -> float:
    """Default independent witness re-scorer.

    Each witness deterministically bootstraps the evidence vector with its own
    seed (resample-with-replacement + small zero-mean jitter) and recomputes a
    score as the bootstrapped-evidence energy mapped through a logistic, anchored
    to the base score. Independent seeds → genuinely different draws, so an honest
    witness will USUALLY-but-not-always land in the same band — which is exactly
    what a quorum is for. Returns a score in [0, 1].
    """
    rng = np.random.default_rng(seed)
    ev = np.asarray(evidence, dtype=float).ravel()
    if ev.size == 0:
        ev = np.array([base_score])
    idx = rng.integers(0, ev.size, size=ev.size)          # bootstrap resample
    boot = ev[idx] + rng.normal(0.0, jitter, size=ev.size)  # independent jitter
    # robust energy of the bootstrapped evidence, blended with the base score so
    # the witness is anchored to the same detection (not scoring noise).
    energy = float(np.tanh(np.mean(np.abs(boot))))
    score = 0.5 * base_score + 0.5 * energy
    return float(np.clip(score, 0.0, 1.0))


def dedup_witness_opinions(opinions: list) -> list:
    """INPUT HYGIENE: keep at most ONE opinion per witness_id (first-seen wins).

    A quorum tally must count DISTINCT witnesses. Without this, a single node
    that submits its opinion K times (echo / duplicate stuffing) would be counted
    K times and could single-handedly manufacture a quorum. Deduping by
    witness_id closes that stuffing vector.

    HONEST SCOPE (Doctrine v11): this is INPUT HYGIENE only. It is NOT sybil
    resistance (a node forging K *distinct* spoofed identities is not stopped
    here) and NOT a Byzantine-safety proof. Khipu BFT stays Conjecture 2.
    """
    seen: set = set()
    out: list = []
    for op in opinions:
        if op.witness_id in seen:
            continue
        seen.add(op.witness_id)
        out.append(op)
    return out


def naive_stuffed_agree_count(opinions: list, agreed_band: str) -> int:
    """DEMONSTRATOR of the vulnerability `dedup_witness_opinions` closes.

    A naive tally counts EVERY submitted opinion in the band with no dedup, so a
    single node echoing its opinion K times inflates the count K-fold. Exposed
    only to make the adversarial test explicit — the shipped `witness_quorum`
    path always dedups external opinions first.
    """
    return sum(1 for op in opinions if op.band == agreed_band)


def witness_quorum(subject: str, base_score: float, evidence: np.ndarray,
                   n_witnesses: int = DEFAULT_N_WITNESSES,
                   quorum: int = DEFAULT_QUORUM,
                   allow_thr: float = 0.35, deny_thr: float = 0.65,
                   rescorer: Optional[Callable] = None,
                   base_seed: int = 1000,
                   external_opinions: Optional[list] = None) -> QuorumReceipt:
    """Run an N-of-M multi-witness quorum over one detection.

    Each of `n_witnesses` independent witnesses re-scores `evidence` with its own
    seed; their verdict bands are tallied; the modal band wins; the detection is
    "witnessed" iff that modal band has >= `quorum` agreeing witnesses.

    Returns a QuorumReceipt with REAL per-witness scores. Khipu BFT = Conjecture 2
    (proposed, not proven) — a witnessed verdict is a stronger ADVISORY only.
    """
    t0 = time.time()
    rescorer = rescorer or default_rescorer
    base_band = verdict_band(base_score, allow_thr, deny_thr)

    n_opinions_submitted: Optional[int] = None
    if external_opinions is not None:
        # Externally-supplied witnesses (e.g. real independent nodes). Apply
        # INPUT HYGIENE: count each witness_id at most once so a single node
        # cannot stuff the tally by echoing its opinion K times. (Honest scope:
        # input hygiene only — NOT sybil resistance, NOT a Byzantine-safety
        # proof; Khipu BFT stays Conjecture 2.)
        submitted = list(external_opinions)
        n_opinions_submitted = len(submitted)
        witnesses = dedup_witness_opinions(submitted)
        n_witnesses = len(witnesses)
    else:
        witnesses = []
        for w in range(n_witnesses):
            seed = base_seed + w
            sc = rescorer(evidence, seed, base_score)
            band = verdict_band(sc, allow_thr, deny_thr)
            witnesses.append(WitnessOpinion(witness_id=f"witness-{w}", seed=seed,
                                            score=sc, band=band))

    # honest edge case: no witnesses ⇒ nothing is witnessed
    if not witnesses:
        empty = QuorumReceipt(
            subject=subject, base_score=float(base_score), base_band=base_band,
            n_witnesses=0, quorum=quorum, agreed_band=None, agree_count=0,
            witnessed=False, agreed_score_lo=0.0, agreed_score_hi=0.0,
            agreed_score_mean=0.0, witnesses=[], digest="sha256:",
            n_opinions_submitted=n_opinions_submitted)
        empty._walltime_s = round(time.time() - t0, 6)
        return empty

    # tally bands → modal band (over DISTINCT witnesses)
    bands = [w.band for w in witnesses]
    uniq, counts = np.unique(bands, return_counts=True)
    j = int(np.argmax(counts))
    agreed_band = str(uniq[j])
    agree_count = int(counts[j])
    witnessed = agree_count >= quorum

    agreeing = [w for w in witnesses if w.band == agreed_band]
    scores = np.array([w.score for w in agreeing], dtype=float)
    lo = float(scores.min()); hi = float(scores.max()); mean = float(scores.mean())

    # honest digest over the AGREEING witness opinions (provenance handle, not a
    # signature — no fabricated signatures, ever)
    payload = "|".join(f"{w.witness_id}:{w.seed}:{w.score:.6f}:{w.band}"
                       for w in agreeing)
    digest = "sha256:" + hashlib.sha256(
        (subject + "|" + payload).encode()).hexdigest()

    receipt = QuorumReceipt(
        subject=subject, base_score=float(base_score), base_band=base_band,
        n_witnesses=n_witnesses, quorum=quorum,
        agreed_band=agreed_band, agree_count=agree_count, witnessed=witnessed,
        agreed_score_lo=lo, agreed_score_hi=hi, agreed_score_mean=mean,
        witnesses=witnesses, digest=digest,
        n_opinions_submitted=n_opinions_submitted)
    receipt._walltime_s = round(time.time() - t0, 6)
    return receipt


def witness_envelope(envelope: dict, evidence: Optional[np.ndarray] = None,
                     **kw) -> dict:
    """Attach a quorum receipt to a §3.0 SDA verdict envelope (non-destructive).

    Uses the envelope's anomaly_score as the base score and its component_scores
    (or the score itself) as the evidence vector. Adds a `_witness` block. Λ stays
    advisory; this NEVER flips _signing.status to a false "verified/green".
    """
    base_score = float(envelope.get("anomaly_score", 0.0))
    if evidence is None:
        comps = envelope.get("component_scores") or {}
        evidence = (np.array(list(comps.values()), dtype=float)
                    if comps else np.array([base_score]))
    subject = str(envelope.get("track_id", "TRK-UNKNOWN"))
    rcpt = witness_quorum(subject, base_score, evidence, **kw)
    out = dict(envelope)
    out["_witness"] = rcpt.to_dict()
    return out


if __name__ == "__main__":
    import json

    # high-concern detection → witnesses should reach a 'deny'-band quorum
    rng = np.random.default_rng(0)
    strong_evidence = np.array([0.9, 0.85, 0.95, 0.8])
    r1 = witness_quorum("TRK-STRONG", base_score=0.82,
                        evidence=strong_evidence)
    print("STRONG detection:")
    print(json.dumps(r1.to_dict(), indent=2))

    # borderline detection → quorum may or may not be reached (honest)
    r2 = witness_quorum("TRK-BORDERLINE", base_score=0.5,
                        evidence=np.array([0.5, 0.48, 0.52, 0.5]))
    print("\nBORDERLINE detection: witnessed =", r2.witnessed,
          "agreed_band =", r2.agreed_band, f"({r2.agree_count}/4)")
