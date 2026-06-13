"""
szl_confidence.py — HONEST bounded confidence for SZL Mosaic / khipu-sda-core.

Implements the spec's confidence contract:
    confidence = {lo, hi, method: "PAC-Bayes|conformal", label: "ESTIMATE"}

Two honest, finite-sample bounded methods (NEVER a certainty claim):
  1. split-conformal band   — non-parametric calibration-quantile interval.
  2. PAC-Bayes (Catoni)     — high-probability risk bound; here used as an
     honest bounded confidence on the empirical anomaly rate. Implemented from
     the published Catoni / McAllester PAC-Bayes bound (cited; NOT vendored).

HONESTY: every interval is labeled ESTIMATE. Λ = Conjecture 1 (advisory). These
are bounds/intervals, not guarantees of correctness.

Lineage / attribution:
  - split-conformal: standard conformal-prediction method (Vovk; Lei et al.).
  - PAC-Bayes (Catoni 2007; McAllester 1999) — bound formula only, cited.
  The SZL spec maps this to szl_shared_formulas/pac_bayes.py (Lean
  pacBayesBound_eq_add_slack). This module is the clean-room numpy reference.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def conformal_band(calib_scores: np.ndarray, point_score: float,
                   alpha: float = 0.1) -> dict:
    """Split-conformal-style bounded interval around an observed anomaly score.

    lo/hi are the alpha/2 and 1-alpha/2 quantiles of the calibration (normal)
    score distribution, widened to contain the observed point. Honest finite-
    sample band, NOT a probability of correctness.
    """
    calib = np.asarray(calib_scores, dtype=float)
    lo_q = float(np.quantile(calib, alpha / 2.0))
    hi_q = float(np.quantile(calib, 1.0 - alpha / 2.0))
    lo = min(lo_q, float(point_score))
    hi = max(hi_q, float(point_score))
    return {"lo": round(lo, 6), "hi": round(hi, 6),
            "method": "conformal", "label": "ESTIMATE",
            "alpha": alpha,
            "note": "split-conformal-style band; finite-sample; NOT a certainty"}


def pac_bayes_catoni(empirical_rate: float, n: int, kl: float = math.log(2),
                     delta: float = 0.05) -> dict:
    """Honest PAC-Bayes (Catoni/McAllester) high-probability upper bound on the
    true risk given an empirical rate over n samples.

    We use the McAllester (1999) PAC-Bayes bound form:
        risk <= emp + sqrt( (KL + ln(2 sqrt(n) / delta)) / (2 n) )
    with a default KL = ln 2 (a weak, honest prior-posterior slack). This is a
    BOUND (cite: McAllester 1999; Catoni 2007), reported as an ESTIMATE band
    [emp, upper]. NOT a guarantee — Λ = Conjecture 1.

    Returns {lo, hi, method, label, ...}. lo = empirical rate (lower honest
    anchor), hi = the PAC-Bayes upper bound (clipped to 1).
    """
    n = max(int(n), 1)
    emp = float(np.clip(empirical_rate, 0.0, 1.0))
    slack = math.sqrt((kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n))
    upper = float(min(1.0, emp + slack))
    return {"lo": round(emp, 6), "hi": round(upper, 6),
            "method": "PAC-Bayes", "label": "ESTIMATE",
            "delta": delta, "n": n, "kl": round(kl, 6),
            "note": ("McAllester/Catoni PAC-Bayes high-prob bound (cited, not "
                     "vendored); ESTIMATE only; Lambda=Conjecture 1 advisory")}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    calib = rng.beta(2, 8, size=500)
    print("conformal:", conformal_band(calib, 0.92, alpha=0.1))
    print("pac-bayes:", pac_bayes_catoni(empirical_rate=0.04, n=300))
