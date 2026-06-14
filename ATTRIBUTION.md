# ATTRIBUTION — khipu-sda-core / szl_mosaic (clean-room)

**SZL Holdings · Doctrine v11 · cite-never-plagiarize**

## 1. Inspiration (capability only — NO code)

This engine is SZL's sovereign, clean-room answer to **True Anomaly Inc.'s
"Mosaic"** space-superiority platform. We were INSPIRED BY Mosaic's **publicly
described capability** — fusing space- and ground-based sensors into a dynamic
Common Operating Picture, accelerating the OODA loop, and supporting Space Domain
Awareness, with the public four-function SDA decomposition
**Detect/Track/ID → Characterize → ML Threat-Warning & Assessment → fuse/forecast**.

- True Anomaly — Mosaic product page: https://www.trueanomaly.space/mosaic
- SDA four-function decomposition (Eric Hilmer, True Anomaly, LinkedIn):
  https://www.linkedin.com/posts/erichilmer_true-anomaly-lands-174m-contract-from-us-activity-7110684034724233216-371t

**No proprietary Mosaic source code, assets, screenshots, or internal interfaces
were seen, copied, decompiled, or referenced.** Mosaic itself is proprietary. We
clean-room the *capability* from public descriptions only. This is INSPIRATION,
not derivation.

## 2. Methods adopted (clean-room — implemented from scratch, with license)

We adopt only **verified-permissive** algorithm lineages. Each method below was
re-implemented from public papers/descriptions; upstream code was not vendored.

| Library | Role / idea adopted | License | URL |
|---|---|---|---|
| PyOD (`yzhao062/pyod`) | Isolation-Forest + classical outlier-detector bank pattern | **BSD-2-Clause** | https://github.com/yzhao062/pyod |
| PyGOD (`pygod-team/pygod`) | Graph outlier-detection idea | **BSD-2-Clause** | https://github.com/pygod-team/pygod |
| Merlion (`salesforce/Merlion`) | Autoencoder TSAD + detector ensembling pattern | **BSD-3-Clause** | https://github.com/salesforce/Merlion |
| TODS (`datamllab/tods`) | Automated multivariate TSAD pipeline pattern | **Apache-2.0** | https://github.com/datamllab/tods |
| tsod (`DHI/tsod`) | Lightweight robust statistical (z-score / MAD) detector | **MIT** | https://github.com/DHI/tsod |
| GDN (`d-ailin/GDN`) | Graph Deviation Network — inter-sensor/track graph anomaly (explainable) | **MIT** | https://github.com/d-ailin/GDN |
| GraGOD (`GraGODs/GraGOD`) | GNN time-series anomaly-detection framework idea | **MIT** | https://github.com/GraGODs/GraGOD |
| python-sgp4 (`brandon-rhodes/python-sgp4`) | SGP4/SDP4 orbit propagation from TLE/OMM; ships the public-domain Vallado `SGP4-VER.TLE` verification element sets used as honest sample fixtures | **MIT** | https://github.com/brandon-rhodes/python-sgp4 |
| scikit-learn | IsolationForest, StandardScaler (permissive dependency) | **BSD-3-Clause** | https://scikit-learn.org |
| PyTorch (optional) | tiny autoencoder (permissive dependency) | **BSD-3-style** | https://pytorch.org |

### Science cited (papers / methods — cite, do NOT vendor)
- GDN, AAAI'21 — *Graph Neural Network-Based Anomaly Detection in Multivariate
  Time Series* — arXiv:2106.06947.
- Graph time-series anomaly-detection surveys — arXiv:2302.00058, arXiv:2307.03759.
- PAC-Bayes bounds — McAllester (1999); Catoni (2007). Used for the honest
  confidence band (formula only; not vendored).
- Conformal prediction — Vovk et al.; Lei et al. (split-conformal band).
- **Conjunction screening / closest-approach geometry** — this is **standard,
  public-domain astrodynamics**: relative-range minimisation over a propagated
  screening window with a local 3-point parabolic time-of-closest-approach (TCA)
  refinement. The canonical reference text is **Vallado, _Fundamentals of
  Astrodynamics and Applications_** (SGP4 + conjunction-assessment basics); the
  SGP4 propagator and its verification report (Vallado et al., _Revisiting
  Spacetrack Report #3_, AIAA 2006-6753) are public-domain. We implement the
  screen from these public-domain methods in numpy — **no proprietary
  conjunction-assessment (CA) product code (e.g. CARA/CSpOC internals) was seen,
  copied, or referenced.** The screen is honestly **SCREENING-GRADE, NOT
  \(P_c\)/covariance-grade**.
- **Byzantine-fault-tolerant N-of-M quorum** (`szl_witness.py`) — the
  super-majority agreement gate is **standard distributed-systems practice**
  (textbook BFT / state-machine-replication quorum intuition; lineage: Lamport
  et al., _The Byzantine Generals Problem_, 1982; Castro & Liskov, _PBFT_, 1999).
  We implement only the **N-of-M agreement mechanism** in numpy — NOT a full BFT
  protocol, and we claim **no safety/liveness proofs**. **Khipu BFT = Conjecture 2
  (proposed, not proven).** No consensus-protocol code was vendored.

## 3. EXCLUDED (license hard-reject)

- **alibi-detect (`SeldonIO/alibi-detect`)** — relicensed to **Business Source
  License 1.1** on **2024-01-22** (no production use without a subscription).
  **DELIBERATELY NOT USED. Not imported. CI blocks it.**
  Source: https://www.seldon.io/strengthening-our-commitment-to-open-core/

## 4. House license & posture

This package ships **Apache-2.0** (SZL house license), with `THIRD_PARTY_NOTICES`
reproducing each upstream license. Doctrine v11 applies throughout: Λ = Conjecture 1
(advisory, never proven trust); **Khipu BFT = Conjecture 2 (proposed, not
proven)** — the `szl_witness.py` 3-of-4 quorum is the engineering mechanism only;
the **orbital conjunction screen is SCREENING-GRADE, NOT \(P_c\)/covariance-grade**
(SGP4 mean-elements, no covariance); sovereign own-metal, 0 CDN; honest UNSIGNED
receipts; validation numbers are real (computed by sgp4 / the detectors, on public
sample TLEs + synthetic data); every $/credit = ESTIMATE; **never fabricate orbital
ground truth** (the demo companion is explicitly DERIVED/labeled).
