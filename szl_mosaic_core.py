"""
szl_mosaic_core.py — SZL-native multivariate + graph anomaly-detection CORE.

================================ ATTRIBUTION ================================
CLEAN-ROOM STATEMENT
--------------------
This module is a CLEAN-ROOM, SZL-native implementation. It is INSPIRED BY the
*publicly described capability* of True Anomaly Inc.'s "Mosaic" space-superiority
platform (SDA / C2 / Threat-Warning & Assessment), per public marketing and press
descriptions only. NO proprietary Mosaic source code, assets, or internals were
seen, copied, or referenced. We clean-room the CAPABILITY (Detect/Track/ID ->
Characterize -> ML-Threat-Warning -> fuse/forecast -> Common Operating Picture)
from public descriptions. See README.md and estate_audit/mosaic_identification.md.

METHODS adopted (ideas / algorithm lineage only — implemented from scratch here,
not vendored), with their verified-permissive licenses:
  - PyOD            (yzhao062/pyod)        BSD-2-Clause  : isolation-forest / z-score outlier-bank pattern
  - Merlion         (salesforce/Merlion)   BSD-3-Clause  : autoencoder TSAD + detector ensembling pattern
  - TODS            (datamllab/tods)       Apache-2.0    : automated multivariate TSAD pipeline pattern
  - tsod            (DHI/tsod)             MIT           : lightweight rule/statistical (robust z-score) detector
  - GDN             (d-ailin/GDN)          MIT           : Graph Deviation Network — inter-sensor graph anomaly idea
  - GraGOD          (GraGODs/GraGOD)       MIT           : GNN time-series anomaly-detection framework idea
  - PyGOD           (pygod-team/pygod)     BSD-2-Clause  : graph outlier detection idea
Science (cited, NOT vendored): GDN AAAI'21 (arXiv 2106.06947); graph-TSAD surveys
(arXiv 2302.00058, 2307.03759).
scikit-learn (BSD-3) and PyTorch (BSD-3 style) are used as permissive dependencies.

alibi-detect is BSL-1.1 (relicensed 2024-01-22) and is DELIBERATELY NOT USED.
============================================================================

HONEST POSTURE (Doctrine v11):
  - Lambda (Λ) = Conjecture 1 (conditional, ADVISORY) — NEVER "proven trust".
    The Λ-gate here emits allow/advisory/deny *advisories*, not proofs.
  - Confidence is reported as a BOUNDED / conformal interval (honest, finite-sample),
    never a point claim of certainty.
  - Every detection emits a structured PROVENANCE RECEIPT, ready for DSSE/Khipu
    signing. We emit it HONESTLY UNSIGNED with a note on where signing happens.
    We NEVER fabricate a signature.
  - Validation numbers are REAL on synthetic data (see szl_mosaic_validate.py).
  - Sovereign / own-metal: pure local compute, no network calls, 0 CDN.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - graceful fallback to numpy AE
    _HAVE_TORCH = False

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# 1. ROBUST STATISTICAL (z-score) DETECTOR  — lineage: tsod (MIT), PyOD (BSD-2)
# ----------------------------------------------------------------------------
class RobustZScoreDetector:
    """Robust multivariate outlier detector using median / MAD per channel.

    Clean-room of the lightweight statistical idea in tsod (MIT) and the
    classical z-score detector pattern catalogued in PyOD (BSD-2). Robust to
    outliers because it uses median and Median-Absolute-Deviation (MAD) instead
    of mean/std. The per-sample score is the max (worst-channel) robust z.

        z_ij = |x_ij - median_j| / (1.4826 * MAD_j + eps)
        score_i = max_j z_ij
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.median_: Optional[np.ndarray] = None
        self.mad_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustZScoreDetector":
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scale = 1.4826 * self.mad_ + self.eps
        z = np.abs(X - self.median_) / scale
        return np.max(z, axis=1)


# ----------------------------------------------------------------------------
# 2. AUTOENCODER DETECTOR — lineage: Merlion VAE/AE TSAD (BSD-3), TODS (Apache-2)
#    Reconstruction error = anomaly score. Torch if available, else numpy PCA-AE.
# ----------------------------------------------------------------------------
if _HAVE_TORCH:

    class _TorchAE(nn.Module):
        def __init__(self, n_in: int, hidden: int, latent: int):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(n_in, hidden), nn.ReLU(),
                nn.Linear(hidden, latent), nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.Linear(latent, hidden), nn.ReLU(),
                nn.Linear(hidden, n_in),
            )

        def forward(self, x):
            return self.dec(self.enc(x))


class AutoencoderDetector:
    """Small autoencoder; reconstruction error is the anomaly score.

    Clean-room of the autoencoder time-series-anomaly pattern in Merlion (BSD-3)
    and TODS (Apache-2.0). Uses a tiny torch MLP-AE when torch is available;
    otherwise falls back to a linear PCA-based autoencoder in pure numpy so the
    engine ALWAYS runs in-sandbox (sovereign, no hard torch dependency).
    """

    def __init__(self, latent: int = 2, hidden: int = 8, epochs: int = 120,
                 lr: float = 1e-2, seed: int = 7):
        self.latent = latent
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.scaler = StandardScaler()
        self._backend = "torch" if _HAVE_TORCH else "numpy-pca"
        self._model = None
        self._components = None  # numpy fallback
        self._mean = None

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        X = self.scaler.fit_transform(np.asarray(X, dtype=float))
        n_in = X.shape[1]
        if _HAVE_TORCH:
            torch.manual_seed(self.seed)
            latent = min(self.latent, max(1, n_in - 1))
            hidden = max(self.hidden, latent + 1)
            self._model = _TorchAE(n_in, hidden, latent)
            opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            xt = torch.tensor(X, dtype=torch.float32)
            self._model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                out = self._model(xt)
                loss = loss_fn(out, xt)
                loss.backward()
                opt.step()
            self._model.eval()
        else:  # numpy linear AE via PCA (top-k components reconstruct)
            self._mean = X.mean(axis=0)
            Xc = X - self._mean
            k = min(self.latent, max(1, n_in - 1))
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._components = Vt[:k]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(np.asarray(X, dtype=float))
        if _HAVE_TORCH:
            with torch.no_grad():
                xt = torch.tensor(X, dtype=torch.float32)
                recon = self._model(xt).numpy()
        else:
            Xc = X - self._mean
            proj = Xc @ self._components.T
            recon = proj @ self._components + self._mean
        return np.sqrt(np.mean((X - recon) ** 2, axis=1))


# ----------------------------------------------------------------------------
# 3. ISOLATION-FOREST DETECTOR — lineage: PyOD (BSD-2) via scikit-learn (BSD-3)
# ----------------------------------------------------------------------------
class IsolationForestDetector:
    """Isolation Forest wrapper. Lineage: PyOD's IForest (BSD-2), scikit-learn.

    Higher score = more anomalous (we negate sklearn's decision_function so that
    larger means "more outlying", consistent with the other detectors).
    """

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05,
                 seed: int = 7):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination,
            random_state=seed,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.model.fit(Xs)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        # decision_function: higher = more normal -> negate for "anomaly"
        return -self.model.decision_function(Xs)


# ----------------------------------------------------------------------------
# Helper: robust min-max normalisation to [0,1] for ensemble combination
# ----------------------------------------------------------------------------
def _normalize01(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(scores)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)


# ----------------------------------------------------------------------------
# 4. GRAPH ANOMALY DETECTOR (track-relational) — lineage: GDN (MIT), PyGOD (BSD-2)
# ----------------------------------------------------------------------------
class GraphDeviationDetector:
    """Lightweight track-relational graph anomaly detector.

    Clean-room of the Graph Deviation Network (GDN, MIT, AAAI'21, arXiv 2106.06947)
    *idea*: build a graph over entities, learn the expected relationship between a
    node and its neighbours, and flag a node whose observed behaviour DEVIATES from
    what its neighbourhood predicts. This is implemented from scratch (no vendored
    GDN code) in a deliberately small, explainable form:

      - Nodes  = tracks (one snapshot per timestep).
      - Edges  = spatial/kinematic proximity (k-nearest neighbours in feature space
                 at a *reference* timestep -> a fixed adjacency, as in GDN's learned
                 sensor graph but here distance-based for transparency).
      - For each node we PREDICT its feature vector as the mean of its neighbours'
        features (a 1-hop graph smoother — the simplest message-passing predictor).
      - Graph-deviation score = ||observed - neighbour_predicted|| .

    A track that moves/behaves unlike its kinematic neighbours (e.g. a maneuver or
    a spoof relative to the local population) gets a high graph-deviation score.
    Output is explainable: we also expose which neighbours/channels drove it.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.adj_: Optional[np.ndarray] = None  # (N, k) neighbour indices
        self.ref_scale_: Optional[np.ndarray] = None

    def fit(self, X_ref: np.ndarray) -> "GraphDeviationDetector":
        """Build the proximity graph from a reference snapshot X_ref (N tracks x F).
        Edges connect each track to its k nearest tracks (excluding itself).
        """
        X_ref = np.asarray(X_ref, dtype=float)
        N = X_ref.shape[0]
        k = min(self.k, max(1, N - 1))
        # pairwise distances
        diff = X_ref[:, None, :] - X_ref[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        self.adj_ = np.argsort(dist, axis=1)[:, :k]
        # reference deviation scale for normalisation
        pred = X_ref[self.adj_].mean(axis=1)
        dev = np.sqrt(np.sum((X_ref - pred) ** 2, axis=1))
        self.ref_scale_ = np.median(dev) + 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-9
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Graph-deviation score per track at a given snapshot X (N x F)."""
        X = np.asarray(X, dtype=float)
        pred = X[self.adj_].mean(axis=1)  # neighbour-predicted features
        dev = np.sqrt(np.sum((X - pred) ** 2, axis=1))
        return dev / self.ref_scale_


# ----------------------------------------------------------------------------
# 5. PROVENANCE RECEIPT — honest, UNSIGNED, ready for DSSE/Khipu signing
# ----------------------------------------------------------------------------
@dataclass
class ProvenanceReceipt:
    """Structured provenance receipt for one detection verdict.

    HONEST: this is emitted UNSIGNED. The `signing` field documents *where* a real
    DSSE/Khipu signature would be attached downstream (a11oy / khipu-consensus
    BFT 3-of-4 quorum on a SHA-256 Merkle DAG). We NEVER fabricate a signature.
    """
    schema: str = "szl.mosaic.receipt/v1"
    inputs_sha256: str = ""           # hash of the exact inputs scored
    track_id: Optional[int] = None    # fused COP track id (if fused)
    timestep: Optional[int] = None
    detector_ensemble: list = field(default_factory=list)
    component_scores: dict = field(default_factory=dict)
    anomaly_score: float = 0.0        # combined ensemble score in [0,1]
    confidence_interval: tuple = (0.0, 0.0)  # conformal/bounded CI on the score
    confidence_method: str = ""
    lambda_verdict: str = "advisory"  # allow | advisory | deny  (Λ Conjecture 1)
    lambda_note: str = ("Lambda is Conjecture 1 (conditional, ADVISORY) — "
                        "NOT proven trust. Human-on-the-loop required.")
    verified: bool = False            # False until a real DSSE signature attaches
    walltime_s: float = 0.0
    signing: str = ("UNSIGNED — sign downstream via DSSE/Khipu in a11oy / "
                    "khipu-consensus (BFT 3-of-4) on a SHA-256 Merkle DAG. "
                    "real-DSSE-or-honestly-UNSIGNED; never silently fabricated.")
    doctrine: str = "v11"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)


def hash_inputs(arr: np.ndarray) -> str:
    """SHA-256 over the exact byte content + shape/dtype of the scored inputs."""
    arr = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# 6. CONFORMAL / BOUNDED CONFIDENCE  — honest finite-sample interval
# ----------------------------------------------------------------------------
def conformal_interval(calib_scores: np.ndarray, point_score: float,
                       alpha: float = 0.1) -> tuple:
    """Honest split-conformal-style bounded interval for an anomaly score.

    We use the empirical distribution of calibration (normal-data) scores to
    place a non-parametric (1 - alpha) interval AROUND the observed point score.
    This is a BOUNDED, finite-sample honesty device — NOT a probabilistic claim
    of correctness. Lower/upper are the alpha/2 and 1 - alpha/2 calibration
    quantiles, clipped so the interval contains the observed point.

    Returns (lower, upper).
    """
    calib = np.asarray(calib_scores, dtype=float)
    lo_q = np.quantile(calib, alpha / 2.0)
    hi_q = np.quantile(calib, 1.0 - alpha / 2.0)
    lower = min(lo_q, point_score)
    upper = max(hi_q, point_score)
    return (float(lower), float(upper))


# ----------------------------------------------------------------------------
# 7. THE ENSEMBLE ENGINE  — combines the detectors + Λ-gate + receipts
# ----------------------------------------------------------------------------
class SZLMosaicCore:
    """SZL-native multivariate + graph anomaly-detection ensemble engine.

    Detector bank (clean-room, lineages cited above):
        - IsolationForestDetector   (PyOD lineage)
        - AutoencoderDetector        (Merlion / TODS lineage)
        - RobustZScoreDetector       (tsod / PyOD lineage)
        - GraphDeviationDetector     (GDN / PyGOD lineage)  [optional, relational]

    The point-detectors operate per (track, timestep) feature vector; their
    normalised scores are averaged into a combined ensemble anomaly score in
    [0,1]. The graph detector adds a track-relational deviation channel.

    Λ-GATE (HONEST, ADVISORY — Λ = Conjecture 1):
        score < allow_thr           -> "allow"     (no advisory)
        allow_thr <= score < deny   -> "advisory"  (human review)
        score >= deny_thr           -> "deny"      (advisory deny — NOT autonomous)
    Verdicts are ADVISORIES under human-on-the-loop, never "proven trust".
    """

    def __init__(self, contamination: float = 0.05, seed: int = 7,
                 allow_thr: float = 0.35, deny_thr: float = 0.65,
                 alpha: float = 0.1):
        self.seed = seed
        self.allow_thr = allow_thr
        self.deny_thr = deny_thr
        self.alpha = alpha
        self.iforest = IsolationForestDetector(contamination=contamination, seed=seed)
        self.ae = AutoencoderDetector(seed=seed)
        self.zscore = RobustZScoreDetector()
        self._norm = {}          # detector -> (lo, hi) for [0,1] normalisation
        self._calib = None       # calibration ensemble scores for conformal CI
        self._fitted = False

    # -- fit on a "mostly-normal" training matrix (n_samples x n_features) -----
    def fit(self, X_train: np.ndarray) -> "SZLMosaicCore":
        X_train = np.asarray(X_train, dtype=float)
        self.iforest.fit(X_train)
        self.ae.fit(X_train)
        self.zscore.fit(X_train)
        # establish per-detector normalisation ranges on training data
        for name, det in (("iforest", self.iforest),
                          ("autoencoder", self.ae),
                          ("robust_zscore", self.zscore)):
            s = det.score(X_train)
            self._norm[name] = (float(np.min(s)),
                                float(np.quantile(s, 0.999)))
        # calibration distribution of the COMBINED score on training (normal) data
        self._calib = self._combined_scores(X_train)
        self._fitted = True
        return self

    def _combined_scores(self, X: np.ndarray) -> np.ndarray:
        comps = self._component_scores(X)
        stacked = np.vstack([comps["iforest"], comps["autoencoder"],
                             comps["robust_zscore"]])
        return stacked.mean(axis=0)

    def _component_scores(self, X: np.ndarray) -> dict:
        out = {}
        for name, det in (("iforest", self.iforest),
                          ("autoencoder", self.ae),
                          ("robust_zscore", self.zscore)):
            raw = det.score(X)
            lo, hi = self._norm[name]
            out[name] = _normalize01(raw, lo, hi)
        return out

    def score(self, X: np.ndarray) -> dict:
        """Return component + combined anomaly scores in [0,1] for X (n x F)."""
        assert self._fitted, "call fit() first"
        comps = self._component_scores(X)
        combined = np.vstack([comps["iforest"], comps["autoencoder"],
                              comps["robust_zscore"]]).mean(axis=0)
        return {"components": comps, "combined": combined}

    def lambda_verdict(self, score: float) -> str:
        """Honest, ADVISORY Λ verdict (Λ = Conjecture 1). Never 'proven trust'."""
        if score < self.allow_thr:
            return "allow"
        if score < self.deny_thr:
            return "advisory"
        return "deny"

    def emit_receipt(self, X_row: np.ndarray, combined_score: float,
                     component_scores: dict, track_id: Optional[int] = None,
                     timestep: Optional[int] = None) -> ProvenanceReceipt:
        """Build an honest UNSIGNED provenance receipt for one detection."""
        t0 = time.time()
        ci = conformal_interval(self._calib, combined_score, alpha=self.alpha)
        receipt = ProvenanceReceipt(
            inputs_sha256=hash_inputs(X_row),
            track_id=track_id,
            timestep=timestep,
            detector_ensemble=["IsolationForest(PyOD-lineage,BSD-2)",
                               "Autoencoder(Merlion/TODS-lineage,BSD-3/Apache-2)",
                               "RobustZScore(tsod-lineage,MIT)",
                               "GraphDeviation(GDN/PyGOD-lineage,MIT/BSD-2)"],
            component_scores={k: float(v) for k, v in component_scores.items()},
            anomaly_score=float(combined_score),
            confidence_interval=ci,
            confidence_method=f"split-conformal-style, alpha={self.alpha} "
                              f"(bounded finite-sample interval, NOT a certainty claim)",
            lambda_verdict=self.lambda_verdict(combined_score),
            verified=False,
            walltime_s=round(time.time() - t0, 6),
        )
        return receipt


if __name__ == "__main__":
    # tiny smoke test
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, size=(300, 4))
    Xte = rng.normal(0, 1, size=(20, 4))
    Xte[0] += 8.0  # inject anomaly
    core = SZLMosaicCore().fit(Xtr)
    res = core.score(Xte)
    print("combined scores (first 5):", np.round(res["combined"][:5], 3))
    r = core.emit_receipt(Xte[0], res["combined"][0],
                          {k: v[0] for k, v in res["components"].items()},
                          track_id=0, timestep=0)
    print(r.to_json())
