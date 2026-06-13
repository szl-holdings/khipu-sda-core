# szl_mosaic / khipu-sda-core — SZL-native clean-room anomaly-detection + SDA CORE

**SZL Holdings · Dev 1 · Doctrine v11 · clean-room · sovereign · honest**

SZL's sovereign answer to True Anomaly's **Mosaic** ("the operating system for
space superiority"). This is the **clean-room anomaly/SDA CORE organ**: a
multivariate + graph anomaly-detection ensemble, multi-sensor track fusion into a
Common Operating Picture (COP), and an honest orbital-SDA roadmap seed — each
verdict emitting a structured, **honestly UNSIGNED** provenance receipt with an
**Λ-gated, bounded-confidence** advisory.

> **One honest sentence:** today killinchu does **counter-UAS / drone / vessel**
> track→classify→evaluate→signed-receipt at the air/maritime edge; the **orbital-
> SDA / Threat-Warning** extension is **roadmap** — this engine builds the
> capability and labels the orbital surface as roadmap, never claiming SZL flies.

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
| `szl_sda_orbit.py` | **SDA roadmap seed**: `python-sgp4` TLE propagation + advisory conjunction/orbital-anomaly flag | python-sgp4 (MIT) |
| `szl_confidence.py` | Honest **bounded confidence**: split-conformal band + **PAC-Bayes (Catoni/McAllester)** bound — labeled `ESTIMATE` | conformal; PAC-Bayes (cited) |
| `szl_sda_envelope.py` | The FROZEN **§3.0 interface contract**: `evaluate(track) -> envelope` with in-toto **Statement v1** receipt | spec §3.0 |
| `szl_mosaic_validate.py` | Synthetic multi-track generator + injected anomalies + **honest precision/recall** + matplotlib figure | — |
| `tests/` | `test_no_mock.py` (no placeholder logic; alibi-detect absent) + `test_receipt_roundtrip.py` | — |

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
honest numbers, not inflated. See `mosaic_validation.png` for the figure (tracks +
flagged anomalies + fused COP + score curves).

---

## Run

```bash
pip install -r requirements.txt
python3 szl_mosaic_core.py        # smoke test the ensemble + receipt
python3 szl_track_fusion.py       # smoke test fusion -> COP
python3 szl_sda_orbit.py          # sgp4 propagation + conjunction advisory
python3 szl_sda_envelope.py       # emit a §3.0 envelope (air + orbital)
python3 szl_mosaic_validate.py    # full validation -> mosaic_validation.png
python3 -m pytest tests/ -q       # honesty + receipt round-trip tests
```

---

## Doctrine v11 (binding)

Λ = **Conjecture 1** (advisory, never proven trust) · locked-proven = **8**
`{F1,F4,F7,F11,F12,F18,F19,F22}` (this engine is engineering capability, NOT in
the locked count) · Khipu BFT = **Conjecture 2** (open) · SLSA **L1 honest /
L2 build-attested / L3 roadmap** · **sovereign own-metal, 0 CDN** · **NO
free-energy** · joules MEASURED only · every $/credit = **ESTIMATE** ·
**cite-never-plagiarize** · NEVER fabricate numbers (validation is real on
synthetic data) · honest sensor caveat: broadcast Remote-ID/ADS-B/MAVLink are
**unauthenticated and spoofable** — every decoded field is a *claim*, and the same
skepticism applies to fused anomaly inputs.
