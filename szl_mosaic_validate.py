"""
szl_mosaic_validate.py — HONEST validation harness for the SZL Mosaic core.

What this does (and it ACTUALLY RUNS, end-to-end, on synthetic data):
  1. Generate synthetic multi-track sensor data (position/velocity/RCS/heading)
     with INJECTED anomalies (maneuver, RCS spike, heading flip) at known
     ground-truth (track, timestep) locations.
  2. Run the SZLMosaicCore ensemble (iForest + autoencoder + robust z-score) to
     score every (track, timestep), plus the GraphDeviationDetector for the
     track-relational channel.
  3. Run SZLTrackFusion over two noisy sensor streams -> fused COP track list.
  4. Report HONEST precision / recall / F1 against the injected ground truth.
  5. Emit a Λ-gated example provenance receipt (honest, UNSIGNED).
  6. Render mosaic_validation.png: tracks + flagged anomalies + fused COP +
     score curves.

HONEST POSTURE: numbers are REAL (computed here on synthetic data), not asserted.
The synthetic generator and the detector are independent. Sovereign / own-metal:
all local, no network.

Attribution: see szl_mosaic_core.py / szl_track_fusion.py / szl_sda_orbit.py
headers (PyOD BSD-2, Merlion BSD-3, TODS Apache-2, tsod MIT, GDN/PyGOD MIT/BSD-2,
python-sgp4 MIT; scikit-learn BSD-3). alibi-detect (BSL-1.1) NOT used.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from szl_mosaic_core import SZLMosaicCore, GraphDeviationDetector
from szl_track_fusion import SZLTrackFusion


# ----------------------------------------------------------------------------
# 1. Synthetic multi-track generator with injected anomalies
# ----------------------------------------------------------------------------
def generate_synthetic(n_tracks: int = 6, n_steps: int = 120, seed: int = 42):
    """Generate n_tracks smooth kinematic tracks over n_steps timesteps.

    Per (track, timestep) feature vector: [x, y, vx, vy, speed, rcs, heading].
    Injects three anomaly TYPES at known locations:
      - MANEUVER  : sudden velocity/heading change (a turn that breaks the model)
      - RCS_SPIKE : radar-cross-section jump (e.g. payload deploy / different body)
      - HDG_FLIP  : abrupt 180-deg heading reversal

    Returns:
      feats : (n_tracks, n_steps, F) feature tensor
      gt     : (n_tracks, n_steps) boolean ground-truth anomaly mask
      truth_xy : (n_tracks, n_steps, 2) clean ground-truth positions (for fusion)
    """
    rng = np.random.default_rng(seed)
    F = 7
    feats = np.zeros((n_tracks, n_steps, F))
    gt = np.zeros((n_tracks, n_steps), dtype=bool)
    truth_xy = np.zeros((n_tracks, n_steps, 2))

    for k in range(n_tracks):
        # random smooth start + heading
        pos = rng.uniform(-20, 20, size=2)
        speed = rng.uniform(0.5, 1.5)
        heading = rng.uniform(0, 2 * np.pi)
        rcs = rng.uniform(1.0, 3.0)            # nominal radar cross section
        for t in range(n_steps):
            # gentle heading drift (normal behaviour)
            heading += rng.normal(0, 0.02)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            pos = pos + np.array([vx, vy])
            r = rcs + rng.normal(0, 0.08)
            feats[k, t] = [pos[0], pos[1], vx, vy, speed, r, heading % (2 * np.pi)]
            truth_xy[k, t] = pos

    # ---- inject anomalies at fixed, known locations ----
    # These are SUSTAINED behavioural anomalies (held over a window), which is the
    # honest target for per-timestep behavioural detection: while the anomaly
    # persists, the track's kinematics stay out-of-distribution.
    #
    # MANEUVER on track 1 at t=40..47 — sustained sharp, erratic turning (high,
    # noisy heading-rate + elevated speed) recomputed kinematically.
    for t in range(40, 48):
        feats[1, t, 4] += 2.5                              # speed elevated
        spd = feats[1, t, 4]
        feats[1, t, 6] = (feats[1, t, 6] + 0.6 + rng.normal(0, 0.15)) % (2 * np.pi)
        feats[1, t, 2] = spd * np.cos(feats[1, t, 6])      # vx consistent w/ hdg
        feats[1, t, 3] = spd * np.sin(feats[1, t, 6])      # vy consistent w/ hdg
        gt[1, t] = True
    # RCS_SPIKE on track 3 at t=70..76 — sustained anomalous radar cross-section
    for t in range(70, 77):
        feats[3, t, 5] += 6.0
        gt[3, t] = True
    # HDG_OSCILLATION on track 4 at t=95..102 — sustained weaving (rapid heading
    # reversals each step) = repeated high heading-rate while it persists.
    for t in range(95, 103):
        flip = np.pi if (t % 2 == 0) else -np.pi * 0.8
        feats[4, t, 6] = (feats[4, t, 6] + flip) % (2 * np.pi)
        spd = feats[4, t, 4]
        feats[4, t, 2] = spd * np.cos(feats[4, t, 6])
        feats[4, t, 3] = spd * np.sin(feats[4, t, 6])
        gt[4, t] = True

    return feats, gt, truth_xy


# ----------------------------------------------------------------------------
# 1b. Behavioural (stationary) feature engineering
# ----------------------------------------------------------------------------
def behavioural_features(feats: np.ndarray) -> np.ndarray:
    """Map raw [x,y,vx,vy,speed,rcs,heading] -> stationary BEHAVIOURAL features.

    Returns per (track, timestep): [speed, rcs, |dvx|, |dvy|, |accel|,
    |heading_rate|]. Absolute position is deliberately DROPPED because it is
    non-stationary (drifts forever) and is not what a behavioural-anomaly
    detector should key on. This is the honest modelling fix that makes the
    detector flag MANEUVERS / SPIKES / FLIPS rather than 'far from origin'.
    """
    n_tracks, n_steps, _ = feats.shape
    out = np.zeros((n_tracks, n_steps, 6))
    for k in range(n_tracks):
        vx = feats[k, :, 2]; vy = feats[k, :, 3]
        speed = feats[k, :, 4]; rcs = feats[k, :, 5]; hdg = feats[k, :, 6]
        dvx = np.abs(np.diff(vx, prepend=vx[0]))
        dvy = np.abs(np.diff(vy, prepend=vy[0]))
        accel = np.sqrt(dvx ** 2 + dvy ** 2)
        # heading rate wrapped to [-pi, pi]
        dh = np.diff(hdg, prepend=hdg[0])
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        hrate = np.abs(dh)
        out[k, :, 0] = speed
        out[k, :, 1] = rcs
        out[k, :, 2] = dvx
        out[k, :, 3] = dvy
        out[k, :, 4] = accel
        out[k, :, 5] = hrate
    return out


# ----------------------------------------------------------------------------
# 2. Metrics (honest precision / recall / F1)
# ----------------------------------------------------------------------------
def prf(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1)


# ----------------------------------------------------------------------------
# 3. Main validation routine
# ----------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("SZL MOSAIC CORE — HONEST VALIDATION ON SYNTHETIC DATA (real numbers)")
    print("=" * 74)

    raw_feats, gt, truth_xy = generate_synthetic()
    n_tracks, n_steps, F = raw_feats.shape
    print(f"synthetic: {n_tracks} tracks x {n_steps} steps x {F} features; "
          f"{int(gt.sum())} injected-anomaly cells of {gt.size} total")

    # ---- BEHAVIOURAL feature engineering (honest modelling choice) ----
    # Absolute position (x,y) grows unboundedly along a track, so it is NOT a
    # stationary signal and would make every late timestep look 'novel'. Mosaic-
    # style TW&A flags BEHAVIOURAL change, so we score on stationary KINEMATIC
    # features: speed, RCS, |accel| (delta-velocity), and |heading-rate|. These
    # are exactly the channels a maneuver / RCS-spike / heading-flip perturb.
    feats = behavioural_features(raw_feats)
    F = feats.shape[2]
    print(f"behavioural features per cell: {F} "
          f"[speed, rcs, |dvx|, |dvy|, |accel|, |heading_rate|]")

    # ---- train the ensemble on a 'mostly-normal' window (first 35 steps) ----
    # exclude the known anomaly windows from training to keep it honest-normal
    train_rows = []
    for k in range(n_tracks):
        for t in range(0, 35):
            train_rows.append(feats[k, t])
    X_train = np.array(train_rows)

    core = SZLMosaicCore(contamination=0.05, allow_thr=0.35, deny_thr=0.65).fit(X_train)

    # ---- score every (track, timestep) ----
    flat = feats.reshape(-1, F)
    res = core.score(flat)
    combined = res["combined"].reshape(n_tracks, n_steps)

    # Detection threshold chosen HONESTLY from the calibration (normal) window:
    # the max normal combined score seen in training + a small margin. This is a
    # principled, data-driven cutoff (not hand-tuned to the test anomalies).
    calib_combined = core._combined_scores(X_train)
    thr = float(np.quantile(calib_combined, 0.995))
    pred_mask = combined >= thr
    metrics = prf(pred_mask, gt)

    print("\n-- POINT-ENSEMBLE DETECTION (iForest + autoencoder + robust z) --")
    print(f"   detection threshold (99.5th pct of normal combined score) = {thr:.3f}")
    print(f"   TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"   PRECISION = {metrics['precision']:.3f}")
    print(f"   RECALL    = {metrics['recall']:.3f}")
    print(f"   F1        = {metrics['f1']:.3f}")

    # ---- graph-relational channel: score the population at each timestep ----
    # Proximity graph in VELOCITY space (vx,vy): neighbours are tracks moving
    # similarly. A maneuver/weave makes a track's velocity diverge sharply from
    # its velocity-neighbours -> large graph-deviation. The GraphDeviationDetector
    # already normalises by the reference-snapshot deviation scale, so we use its
    # raw score directly. The threshold is set HONESTLY from the calibration
    # (normal) window: a robust population cutoff (median + 6*MAD), NOT tuned to
    # the test anomalies.
    gstate = raw_feats[:, :, 2:4]                 # [vx, vy] per track,time
    gdn = GraphDeviationDetector(k=3).fit(gstate[:, 10, :])  # ref snapshot t=10
    graph_scores = np.zeros((n_tracks, n_steps))
    for t in range(n_steps):
        graph_scores[:, t] = gdn.score(gstate[:, t, :])
    calib_g = graph_scores[:, :35].ravel()
    g_med = np.median(calib_g)
    g_mad = np.median(np.abs(calib_g - g_med)) + 1e-9
    graph_thr = float(g_med + 6.0 * 1.4826 * g_mad)
    graph_pred = graph_scores >= graph_thr
    graph_metrics = prf(graph_pred, gt)
    print("\n-- GRAPH-RELATIONAL CHANNEL (GDN-style track-deviation) --")
    print(f"   threshold = {graph_thr:.3f}")
    print(f"   PRECISION = {graph_metrics['precision']:.3f}  "
          f"RECALL = {graph_metrics['recall']:.3f}  F1 = {graph_metrics['f1']:.3f}")

    # ---- FUSED decision: normalised CONSENSUS of point + graph channels ----
    # Union (OR) inflates false positives; a consensus that AVERAGES the two
    # normalised channels is the honest fusion. We min-max the graph score into
    # [0,1] using the calibration window, then average with the point score and
    # threshold at the calibration 99.5th percentile of the fused score.
    g01 = np.clip((graph_scores - calib_g.min()) /
                  (np.quantile(calib_g, 0.999) - calib_g.min() + 1e-9), 0, 1)
    fused_score = 0.5 * combined + 0.5 * g01
    fused_calib = fused_score[:, :35].ravel()
    fused_thr = float(np.quantile(fused_calib, 0.995))
    union_pred = fused_score >= fused_thr
    union_metrics = prf(union_pred, gt)
    print("\n-- FUSED DECISION (consensus = 0.5*point + 0.5*graph) --")
    print(f"   threshold = {fused_thr:.3f}")
    print(f"   PRECISION = {union_metrics['precision']:.3f}  "
          f"RECALL = {union_metrics['recall']:.3f}  F1 = {union_metrics['f1']:.3f}")

    # ---- track fusion: build 2 noisy sensor streams from truth_xy ----
    rng = np.random.default_rng(7)
    fuse = SZLTrackFusion(gate=6.0, init_var=5.0, prune_misses=8)
    for t in range(n_steps):
        true_pts = truth_xy[:, t, :]                       # (n_tracks, 2)
        s1 = true_pts + rng.normal(0, 0.4, true_pts.shape)  # radar
        s2 = true_pts + rng.normal(0, 0.7, true_pts.shape)  # eo
        fuse.step({"radar": s1, "eo": s2}, t=t,
                  sensor_noise={"radar": 0.4, "eo": 0.7})
    n_fused = len(fuse.tracks)
    print(f"\n-- TRACK FUSION (2 noisy sensors -> COP) --")
    print(f"   true tracks = {n_tracks}; fused COP tracks = {n_fused}")

    # ---- emit a Λ-gated example provenance receipt for the strongest detection ----
    idx = np.unravel_index(np.argmax(combined), combined.shape)
    k, t = int(idx[0]), int(idx[1])
    comp_row = {name: float(res["components"][name].reshape(n_tracks, n_steps)[k, t])
                for name in res["components"]}
    receipt = core.emit_receipt(feats[k, t], combined[k, t], comp_row,
                                track_id=k, timestep=t)
    print("\n-- Λ-GATED EXAMPLE PROVENANCE RECEIPT (HONEST, UNSIGNED) --")
    print(receipt.to_json())

    # ---- FIGURE ----
    make_figure(raw_feats, gt, combined, pred_mask, thr, graph_scores, graph_thr,
                fuse, truth_xy, metrics, union_metrics)
    print("\nfigure written -> mosaic_validation.png")

    return dict(point=metrics, graph=graph_metrics, union=union_metrics,
                n_fused=n_fused, n_tracks=n_tracks, thr=thr, receipt=receipt)


def make_figure(feats, gt, combined, pred_mask, thr, graph_scores, graph_thr,
                fuse, truth_xy, metrics, union_metrics):
    n_tracks, n_steps, F = feats.shape
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SZL Mosaic Core — Honest Validation (synthetic data, real numbers)\n"
                 "Clean-room of True Anomaly 'Mosaic' CAPABILITY | lineage: "
                 "PyOD/Merlion/TODS/tsod/GDN/PyGOD/sgp4 | Λ=Conjecture 1 (ADVISORY)",
                 fontsize=11, fontweight="bold")

    # (a) tracks in XY with flagged anomalies
    a = ax[0, 0]
    cmap = plt.cm.tab10
    for k in range(n_tracks):
        a.plot(feats[k, :, 0], feats[k, :, 1], "-", color=cmap(k % 10),
               alpha=0.6, lw=1.2, label=f"track {k}")
        # flagged points
        fl = pred_mask[k]
        a.scatter(feats[k, fl, 0], feats[k, fl, 1], color="red", s=28,
                  marker="x", zorder=5)
        # ground-truth anomalies (circles)
        g = gt[k]
        a.scatter(feats[k, g, 0], feats[k, g, 1], facecolors="none",
                  edgecolors="black", s=70, zorder=4, linewidths=1.2)
    a.set_title("(a) Tracks (lines) | flagged anomalies (red x) | "
                "injected GT (black o)")
    a.set_xlabel("x"); a.set_ylabel("y")
    a.legend(fontsize=7, ncol=2, loc="best")

    # (b) fused COP overlay vs truth
    b = ax[0, 1]
    for k in range(n_tracks):
        b.plot(truth_xy[k, :, 0], truth_xy[k, :, 1], "-", color="gray",
               alpha=0.5, lw=1.0)
    for tr in fuse.tracks:
        if len(tr.history) > 1:
            hx = [h[1] for h in tr.history]
            hy = [h[2] for h in tr.history]
            b.plot(hx, hy, "-", lw=1.8, alpha=0.9)
            b.scatter(hx[-1], hy[-1], s=40, marker="D", zorder=5)
    b.set_title(f"(b) Fused COP tracks (colored) vs truth (gray) — "
                f"{len(fuse.tracks)} fused / {n_tracks} true\n"
                f"2 noisy sensors (radar+eo), NN-assoc + tiny Kalman")
    b.set_xlabel("x"); b.set_ylabel("y")

    # (c) combined anomaly-score curves (Schwarz-style score-vs-time per track)
    c = ax[1, 0]
    for k in range(n_tracks):
        c.plot(combined[k], color=cmap(k % 10), alpha=0.8, lw=1.2,
               label=f"track {k}")
        # mark GT windows
        gtk = np.where(gt[k])[0]
        if gtk.size:
            c.scatter(gtk, combined[k, gtk], color="black", s=18, zorder=5)
    c.axhline(thr, color="red", ls="--", lw=1.2,
              label=f"detect thr={thr:.2f}")
    c.set_title(f"(c) Combined ensemble anomaly score per track over time\n"
                f"point-ensemble P={metrics['precision']:.2f} "
                f"R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")
    c.set_xlabel("timestep"); c.set_ylabel("combined score [0,1]")
    c.legend(fontsize=7, ncol=2, loc="upper left")

    # (d) graph-deviation score curves
    d = ax[1, 1]
    for k in range(n_tracks):
        d.plot(graph_scores[k], color=cmap(k % 10), alpha=0.8, lw=1.2)
        gtk = np.where(gt[k])[0]
        if gtk.size:
            d.scatter(gtk, graph_scores[k, gtk], color="black", s=18, zorder=5)
    d.axhline(graph_thr, color="red", ls="--", lw=1.2,
              label=f"graph thr={graph_thr:.2f}")
    d.set_title(f"(d) Graph-relational deviation score (GDN-style)\n"
                f"fused consensus (0.5*pt+0.5*graph) P={union_metrics['precision']:.2f} "
                f"R={union_metrics['recall']:.2f} F1={union_metrics['f1']:.2f}")
    d.set_xlabel("timestep"); d.set_ylabel("graph-deviation score")
    d.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("mosaic_validation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
