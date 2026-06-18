# szl_mosaic / khipu-sda-core — SZL-native clean-room anomaly-detection + SDA CORE

**SZL Holdings · Dev 1 · Doctrine v11 · clean-room · sovereign · honest**

SZL's sovereign answer to True Anomaly's **Mosaic** ("the operating system for
space superiority"). This is the **clean-room anomaly/SDA CORE organ**: a
multivariate + graph anomaly-detection ensemble, multi-sensor track fusion into a
Common Operating Picture (COP), a **real orbital conjunction screen** (sgp4
time-of-closest-approach + miss distance over a small TLE catalog, honestly
labeled SCREENING-GRADE / roadmap-toward-operational), and **multi-witness
verified detection** (a Khipu 3-of-4 quorum, Conjecture 2) — each verdict emitting
a structured, **honestly UNSIGNED** provenance receipt with an **Λ-gated,
bounded-confidence** advisory.

> **One honest sentence:** today killinchu does **counter-UAS / drone / vessel**
> track→classify→evaluate→signed-receipt at the air/maritime edge; the **orbital-
> SDA / Threat-Warning** extension is **roadmap-toward-operational** — the orbital
> screen now computes **real** sgp4 closest-approach geometry but is honestly
> labeled **SCREENING-GRADE, NOT \(P_c\)/covariance-grade** and never claims SZL flies.

---

## CLEAN-ROOM STATEMENT (read first)

This engine is **inspired by the publicly described CAPABILITY** of True Anomaly
Inc.'s proprietary **Mosaic** platform (SDA / C2 / Threat-Warning) — the public
four-function decomposition **Detect/Track/ID → Characterize → ML Threat-Warning
& Assessment → fuse/forecast into a COP** ([Mosaic page](https://www.trueanomaly.space/mosaic);
[Hilmer/True Anomaly, LinkedIn](https://www.linkedin.com/posts/erichilmer_true-anomaly-lands-174m-contract-from-us-activity-7110684034724233216-371t)).
**No proprietary Mosaic code, assets, or internals were seen, copied, or
referenced.** We clean-room the *capability* only, from public descriptions plus
**verified-permissive** open-source methods. See `ATTRIBUTION.md` +
`THIRD_PARTY_NOTICES`. **alibi-detect (BSL-1.1) is deliberately excluded.**

Lineage adopted (methods only, implemented from scratch): **PyOD** (BSD-2),
**PyGOD** (BSD-2), **Merlion** (BSD-3), **TODS** (Apache-2.0), **tsod** (MIT),
**GDN** (MIT), **GraGOD** (MIT), **python-sgp4** (MIT). Runtime deps: numpy, scipy,
scikit-learn (BSD-3), matplotlib, optional torch (BSD-3-style).

---

## Modules

| File | Role | Lineage |
|---|---|---|
| `szl_mosaic_core.py` | Multivariate anomaly ENSEMBLE (Isolation-Forest + autoencoder + robust z-score) + **GraphDeviationDetector** (track-relational) + provenance receipt + Λ-gate + conformal CI | PyOD, Merlion/TODS, tsod, GDN/PyGOD |
| `szl_track_fusion.py` | Multi-sensor track fusion: global-nearest-neighbour association + tiny constant-velocity **Kalman** update → fused **COP** track list | textbook MTT (numpy) |
| `szl_sda_orbit.py` | **SDA orbital screen (roadmap-toward-operational)**: `python-sgp4` TLE propagation + **REAL pairwise conjunction screen** (time-of-closest-approach + miss distance over a small catalog) + conformal band + advisory orbital-anomaly flag | python-sgp4 (MIT) |
| `szl_witness.py` | **Multi-witness verified detection**: N independent witness re-scorers + **Khipu 3-of-4 quorum** + quorum receipt (Conjecture 2, proposed) | textbook BFT quorum (numpy) |
| `szl_confidence.py` | Honest **bounded confidence**: split-conformal band + **PAC-Bayes (Catoni/McAllester)** bound — labeled `ESTIMATE` | conformal; PAC-Bayes (cited) |
| `szl_sda_envelope.py` | The FROZEN **§3.0 interface contract**: `evaluate(track) -> envelope` with in-toto **Statement v1** receipt | spec §3.0 |
| `szl_mosaic_validate.py` | Synthetic multi-track generator + injected anomalies + **honest precision/recall** + **orbital conjunction screen** + **witness quorum** + 6-panel matplotlib figure | — |
| `tests/` | `test_no_mock.py` (no placeholder logic; alibi-detect absent) + `test_receipt_roundtrip.py` + `test_orbit_screen.py` (real sgp4 conjunction) + `test_witness_quorum.py` (real 3-of-4 quorum) | — |

---

## The math (clean-room formulas)

### 1. Robust z-score detector (tsod / PyOD lineage)
Per channel \(j\), using median and Median-Absolute-Deviation (MAD) for robustness:
\[
z_{ij} = \frac{|x_{ij} - \mathrm{median}_j|}{1.4826\,\mathrm{MAD}_j + \epsilon},
\qquad s_i^{\text{z}} = \max_j z_{ij}.
\]
(The constant \(1.4826\) makes MAD a consistent estimator of \(\sigma\) for Gaussians.)

### 2. Autoencoder reconstruction detector (Merlion / TODS lineage)
A small autoencoder \(g\circ f\) (torch MLP, or numpy PCA fallback) trained on
standardized normal data; anomaly score is the RMS reconstruction error:
\[
s_i^{\text{ae}} = \sqrt{\tfrac{1}{F}\sum_{j=1}^{F}\bigl(x_{ij} - (g\!\circ\! f)(x_i)_j\bigr)^2}.
\]

### 3. Isolation Forest detector (PyOD lineage, via scikit-learn)
\(s_i^{\text{if}} = -\,\text{decisionfn}(x_i)\) (negated so larger ⇒ more outlying).

### 4. Ensemble combination
Each detector's score is min–max normalized to \([0,1]\) using a training-data
range \((\ell,h)\): \(\tilde s = \mathrm{clip}\!\big((s-\ell)/(h-\ell),0,1\big)\).
The combined per-cell score is the mean:
\[
S_i = \tfrac{1}{3}\bigl(\tilde s_i^{\text{if}} + \tilde s_i^{\text{ae}} + \tilde s_i^{\text{z}}\bigr)\in[0,1].
\]

### 5. Graph-deviation detector (GDN / PyGOD lineage, clean-room)
Nodes = tracks; edges = \(k\)-nearest neighbours in **kinematic (velocity) space**
at a reference snapshot. Each node's features are PREDICTED as the mean of its
neighbours (a 1-hop graph smoother / message-passing predictor); the deviation
score is
\[
d_i = \frac{\bigl\|x_i - \frac{1}{k}\sum_{j\in N(i)} x_j\bigr\|}{\text{ref-scale}}.
\]
A track that maneuvers unlike its kinematic neighbourhood gets a large \(d_i\).
This is the explainable "which track/sensor deviated" idea of GDN, done minimally.

### 6. Track fusion (textbook MTT)
Constant-velocity state \(\mathbf{x}=[x,y,v_x,v_y]\); predict
\(\mathbf{x}\leftarrow F\mathbf{x},\ P\leftarrow FPF^\top+Q\); associate
measurements to tracks by minimizing total Euclidean cost (Hungarian if scipy,
else greedy) within a gate; Kalman-correct with gain \(K=PH^\top(HPH^\top+R)^{-1}\).

### 7. Honest bounded confidence
- **Split-conformal band:** lower/upper = \(\alpha/2,\ 1-\alpha/2\) calibration
  quantiles, widened to contain the observed score.
- **PAC-Bayes (McAllester):**
  \(\text{risk} \le \hat r + \sqrt{\tfrac{\mathrm{KL} + \ln(2\sqrt{n}/\delta)}{2n}}\).
Both are reported as `ESTIMATE`. They are bounds/intervals — **never a certainty**.

### 8. Λ-gate (HONEST, ADVISORY — Λ = Conjecture 1)
\[
\text{verdict}(S)=\begin{cases}\text{allow}& S<0.35\\ \text{advisory}& 0.35\le S<0.65\\ \text{deny}& S\ge 0.65\end{cases}
\]
Verdicts are **advisories** under human-on-the-loop — **never "proven trust"**,
never folded into locked-proven = 8.

### 9. Orbital conjunction screen (REAL closest-approach geometry)
Propagate each catalog object with SGP4 over a screening window
\(\{t_0+k\,\Delta t\}_{k=0}^{N-1}\) to TEME positions \(\mathbf r_a(t),\mathbf r_b(t)\).
For each pair, the discrete relative range is \(d_k=\lVert\mathbf r_a(t_k)-\mathbf r_b(t_k)\rVert\);
the time-of-closest-approach (TCA) is refined to sub-step resolution by a local
3-point parabolic vertex around the discrete minimum \(k^\star\):
\[
\delta=\tfrac12\,\frac{d_{k^\star-1}-d_{k^\star+1}}{d_{k^\star-1}-2d_{k^\star}+d_{k^\star+1}},\qquad
d_{\min}=d_{k^\star}-\tfrac14\,(d_{k^\star-1}-d_{k^\star+1})\,\delta.
\]
A pair is flagged when \(d_{\min}\le\) `threshold_km`. The encounter (relative)
speed at TCA is \(\lVert\mathbf v_a-\mathbf v_b\rVert\). This is **standard,
public-domain astrodynamics** (relative-range minimisation + parabolic
refinement) — no proprietary CA internals.

> **HONEST LIMIT — SCREENING-GRADE, NOT \(P_c\)/COVARIANCE-GRADE.** SGP4 propagates
> **mean elements with no per-object covariance**, so the screen emits a
> *deterministic miss distance*, **never a probability of collision** \(P_c\). TLE
> epoch staleness and drag/SRP mismodelling widen the true uncertainty well beyond
> the reported point miss (km-level along-track error budget). A real CA product
> (e.g. CARA/CSpOC) fuses covariance, a high-fidelity numerical propagator, and
> integrates \(P_c\) over the encounter — **this module does triage only**, and is
> ADVISORY (Λ = Conjecture 1). The miss is wrapped in the §7 conformal band as an
> `ESTIMATE`, **not** a calibrated risk. The demo catalog uses **real public sample
> TLEs plus one explicitly-DERIVED co-orbiting companion** (a transparent
> mean-anomaly offset of a real TLE, labeled SAMPLE/DERIVED) so the screen finds a
> real, sub-threshold close approach to exercise — we **never fabricate orbital
> ground truth**.

### 10. Multi-witness verified detection (Khipu 3-of-4 quorum)
Given a detection with base score \(S\) and evidence vector \(\mathbf e\), each of
\(M\) **independent witness nodes** \(w\) re-scores it under its own seed: it
bootstrap-resamples the evidence (\(\tilde{\mathbf e}_w=\mathbf e[\mathrm{idx}_w]+\eta_w\),
\(\eta_w\sim\mathcal N(0,\sigma^2)\)) and reports
\(s_w=\tfrac12 S+\tfrac12\tanh(\overline{|\tilde{\mathbf e}_w|})\in[0,1]\), then maps
\(s_w\) to a Λ band. Let \(c\) be the count of witnesses in the modal band; the
detection is **witnessed** iff \(c\ge Q\) with the Khipu quorum \(Q\text{-of-}M=3\text{-of-}4\)
(super-majority, tolerates one dissenting/faulty witness). The quorum receipt
records which witnesses agreed and the agreed score band \([\min_w s_w,\max_w s_w]\).

> **HONEST LIMIT — Khipu BFT = Conjecture 2 (PROPOSED, NOT PROVEN).** This is the
> **engineering mechanism** (an \(N\)-of-\(M\) agreement gate), **not a proven-safe
> consensus**: no Byzantine safety/liveness proof, no partial-synchrony bound, no
> equivocation/sybil model exercised. The witnesses are **simulated independent
> re-scorers in one process** — no network, no real replication, no adversary. A
> *witnessed* verdict is a **stronger ADVISORY** (more independent scorers
> concurred), **never proven trust** and **never folded into locked-proven = 8**.
> The receipt digest is a SHA-256 provenance handle, **not** a signature — no
> signature is ever fabricated. Λ stays Conjecture 1 (advisory).

---

## §3.0 SDA verdict envelope (the frozen interface contract)

`evaluate(track, core, calib_scores, stage=...) -> envelope`:

```jsonc
{
  "track_id": "TRK-0001",
  "stage": "DTID|CHARACTERIZE|TWA|FUSE",
  "anomaly_score": 0.0,                                  // [0,1]
  "lambda_axis": {"name":"anomaly_twa","value":0.0,"advisory":true,"verdict":"allow|advisory|deny"},
  "confidence": {"lo":0.0,"hi":0.0,"method":"PAC-Bayes|conformal","label":"ESTIMATE"},
  "sgp4": {"tle_hash":"sha256:…","propagated":true},     // null for air/maritime
  "receipt": {                                           // in-toto Statement v1
    "_type": "https://in-toto.io/Statement/v1",
    "predicateType": "https://szlholdings.com/attestations/sda-anomaly/v1",
    "subject": [{"name":"TRK-0001","digest":{"sha256":"…"}}],
    "predicate": {"engine":"khipu-sda-core","model_hash":"…","seed":0,
                  "lib_lineage":["pyod-BSD2","gdn-MIT","sgp4-MIT","merlion-BSD3"],
                  "verified": false, "sovereign": true}
  },
  "_signing": {"status":"UNSIGNED","signed_by":"szl_lake/khipu DSSE (Ed25519/P-256)"}
}
```

### Receipt schema — honesty invariants
- `predicate.verified = false` and `_signing.status = "UNSIGNED"` until a **real**
  DSSE signature attaches downstream (a11oy / khipu-consensus, BFT 3-of-4, on a
  SHA-256 Khipu Merkle DAG). **UNSIGNED is structural-only — never a false green.**
- **No signature is ever fabricated.** `real-DSSE-or-honestly-UNSIGNED`.
- `predicate.sovereign = true` (own-metal, 0 CDN). `confidence.label = "ESTIMATE"`.
- `lambda_axis.advisory = true` always (Λ = Conjecture 1).

---

## Validation (REAL numbers on synthetic data)

Run: `python3 szl_mosaic_validate.py` → writes `mosaic_validation.png`, prints
precision/recall and a Λ-gated example receipt/envelope.

Synthetic: **6 tracks × 120 steps × 7 raw features**, reduced to **6 behavioural
features** [speed, rcs, |Δvx|, |Δvy|, |accel|, |heading-rate|]; **23 injected-
anomaly cells** across 3 anomaly types (sustained maneuver, RCS spike, heading
oscillation). Detectors trained on the normal early window; thresholds set from
the calibration (normal) distribution (not tuned to the test anomalies):

| Channel | Precision | Recall | F1 |
|---|---|---|---|
| Point ensemble (iForest + AE + robust-z) | **0.44** | **0.74** | **0.55** |
| Graph-relational (GDN-style velocity-deviation) | **1.00** | **0.35** | **0.52** |
| Fused consensus (0.5·point + 0.5·graph) | **0.40** | **0.70** | **0.51** |
| Track fusion | 6 fused COP tracks from 6 true (2 noisy sensors) | | |

**Honest reading:** the point ensemble carries recall; the graph channel is
high-precision / low-recall (it only fires on velocity-space maneuvers and
correctly MISSES the RCS spike, which has no kinematic signature). False positives
come from genuine normal heading-drift noise — not hidden. These are small-model,
honest numbers, not inflated.

### Orbital conjunction screen (REAL sgp4 numbers)
The same harness now runs the real pairwise conjunction screen over a 4-object
demo catalog (real public sample TLEs + one explicitly-DERIVED co-orbiting
companion) on a 90-minute window (180 steps × 0.5 min):

- **catalog objects = 4; pairwise conjunctions < 5 km = 1**
- **ISS (ZARYA) 25544 × ISS-COMPANION (DERIVED):** miss **2.368 km** @ TCA
  **+2302 s**, encounter rel-speed **0.0027 km/s**, verdict **advisory**, conformal
  `ESTIMATE` band **[2.37, 9040] km** (the wide upper bound honestly reflects the
  no-covariance screening posture — it is not a \(P_c\)). The three unrelated
  public pairs stay at ~10³–10⁴ km and are NOT flagged. **SCREENING-GRADE, NOT
  \(P_c\)-grade** (see math §9).

### Multi-witness verified detection (REAL 3-of-4 quorum)
The harness witnesses the engine's strongest behavioural detection and a
clearly-normal cell through 4 independent re-scorers:

- **STRONG TRK-1@t40:** base 1.000 (deny) → **witnessed**, agreed band **deny 4/4**,
  agreed score-band **[0.942, 0.991]**.
- **NORMAL TRK-0@t5:** base 0.188 (allow) → **witnessed**, agreed band **allow 3/4**.
- On bimodal/boundary evidence the quorum genuinely **FAILS** (a real 2–2 split is
  exercised in `tests/test_witness_quorum.py`) — the gate is real, not always-True.
  **Khipu BFT = Conjecture 2 (proposed, not proven)** — a witnessed verdict is a
  stronger ADVISORY only (see math §10).

See `mosaic_validation.png` for the 6-panel figure (tracks + flagged anomalies +
fused COP + score curves + **orbital conjunction screen** + **witness quorum**).

---

## Run

```bash
pip install -r requirements.txt
python3 szl_mosaic_core.py        # smoke test the ensemble + receipt
python3 szl_track_fusion.py       # smoke test fusion -> COP
python3 szl_sda_orbit.py          # REAL pairwise conjunction screen (sgp4 TCA + miss)
python3 szl_witness.py            # multi-witness 3-of-4 quorum (Conjecture 2)
python3 szl_sda_envelope.py       # emit a §3.0 envelope (air + orbital)
python3 szl_mosaic_validate.py    # full validation -> mosaic_validation.png
python3 -m pytest tests/ -q       # honesty + receipt round-trip tests
```

---

## Operations (ownership · health · rollback)

**Ownership.** SZL Holdings · Dev 1 (anomaly/SDA CORE organ). This is an importable
library + CLI smoke/validation harness — it is *not* a long-running network service
(sovereign, own-metal, 0 CDN, no listening port), so there is no `/healthz` endpoint
to expose. Its "health check" is the test suite + per-module smoke run.

**Health / readiness check.** Treat a green run of the following as readiness:

```bash
python3 -m pytest tests/ -q      # 20 tests: honesty gate + receipt/orbit/quorum
python3 szl_sda_orbit.py         # must print: catalog objects: 4; ... advisories (<5 km): 1
```

CI (`.github/workflows/ci.yml`) gates every push/PR on: the full test suite, a
license-hygiene check (BSL `alibi-detect` forbidden as a dependency/import), and a
no-fabricated-signature check. Main is never merged red.

**Rollback (one-step, tested).** Releases are plain git refs — there is no stateful
migration to unwind, so rollback is a single revert/checkout:

```bash
# revert the most recent merge to main (keeps history; preferred)
git revert -m 1 <merge_commit_sha>

# or pin a known-good tagged release for a redeploy
git checkout <last_good_tag>     # e.g. the commit before the regression
```

Verify the rollback the same way as readiness above (pytest + the orbit smoke line).
Because the engine holds no persistent state and writes only `mosaic_validation.png`
(gitignored), reverting code is a complete rollback.

---

## Doctrine v11 (binding)

Λ = **Conjecture 1** (advisory, never proven trust) · locked-proven = **8**
`{F1,F4,F7,F11,F12,F18,F19,F22}` (this engine is engineering capability, NOT in
the locked count) · Khipu BFT = **Conjecture 2** (proposed, NOT proven — the
`szl_witness.py` 3-of-4 quorum is the engineering mechanism, not a proven-safe
consensus; a 'witnessed' verdict is a stronger ADVISORY only) · orbital screen =
**SCREENING-GRADE roadmap**, honestly NOT operational \(P_c\)/covariance-grade ·
SLSA **L1 honest /
L2 build-attested / L3 roadmap** · **sovereign own-metal, 0 CDN** · **NO
free-energy** · joules MEASURED only · every $/credit = **ESTIMATE** ·
**cite-never-plagiarize** · NEVER fabricate numbers (validation is real on
synthetic data) · honest sensor caveat: broadcast Remote-ID/ADS-B/MAVLink are
**unauthenticated and spoofable** — every decoded field is a *claim*, and the same
skepticism applies to fused anomaly inputs.
