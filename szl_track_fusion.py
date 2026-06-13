"""
szl_track_fusion.py — SZL-native multi-sensor track fusion -> Common Operating
Picture (COP).

================================ ATTRIBUTION ================================
CLEAN-ROOM STATEMENT
--------------------
Clean-room, SZL-native. INSPIRED BY the *publicly described capability* of True
Anomaly Inc.'s "Mosaic" platform — specifically its public claim of "fusing data
from space- and ground-based sensors ... into a complete and dynamic Common
Operating Picture." NO proprietary code seen or copied. We clean-room the
CAPABILITY only.

The fusion math here (global-nearest-neighbour association + a scalar/diagonal
Kalman update) is STANDARD, textbook multi-target tracking — implemented from
scratch with numpy. No third-party tracking library is vendored. scikit-learn
(BSD-3) provides the linear_sum_assignment-free Hungarian via scipy if available;
we fall back to greedy nearest-neighbour so the module always runs.
============================================================================

HONEST POSTURE (Doctrine v11):
  - Sovereign / own-metal: pure local numpy, no network, 0 CDN.
  - Sensor inputs are CLAIMS, not ground truth (broadcast Remote-ID / ADS-B /
    MAVLink are unauthenticated and spoofable). Fusion does NOT launder a spoofed
    measurement into "truth"; it produces a fused *estimate* with covariance.
  - Honest delineation: this is a SIMPLE NN-association + tiny Kalman update, NOT
    a full JPDA/MHT tracker. It is the COP seed for the SZL Mosaic organ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------------
# Fused COP track: a tiny constant-position Kalman filter on the state [x, y, vx, vy]
# ----------------------------------------------------------------------------
@dataclass
class FusedTrack:
    """One fused Common-Operating-Picture track.

    State is [x, y, vx, vy] with a diagonal covariance P (kept simple/diagonal for
    transparency). Each update is a scalar-gain Kalman correction toward the
    associated measurement.
    """
    track_id: int
    state: np.ndarray                       # [x, y, vx, vy]
    P: np.ndarray                           # 4x4 diagonal covariance
    n_updates: int = 0
    last_t: float = 0.0
    history: list = field(default_factory=list)        # [(t, x, y)]
    contributing_sensors: list = field(default_factory=list)

    def predict(self, dt: float, q: float = 0.05):
        """Constant-velocity predict step."""
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        self.state = F @ self.state
        Q = np.diag([q, q, q, q]) * dt
        self.P = F @ self.P @ F.T + Q

    def update(self, z_xy: np.ndarray, r: float, t: float, sensor_id: str):
        """Kalman correction with a position-only measurement z_xy = [x, y].

        Measurement model H maps state->position. We use a diagonal R = r*I.
        """
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * r
        y = z_xy - H @ self.state                  # innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)        # Kalman gain
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.n_updates += 1
        self.last_t = t
        self.history.append((t, float(self.state[0]), float(self.state[1])))
        if sensor_id not in self.contributing_sensors:
            self.contributing_sensors.append(sensor_id)


# ----------------------------------------------------------------------------
# Association: global-nearest-neighbour (Hungarian if scipy, else greedy)
# ----------------------------------------------------------------------------
def _associate(track_positions: np.ndarray, meas_positions: np.ndarray,
               gate: float) -> list:
    """Associate measurements to tracks by minimising total Euclidean distance.

    Returns list of (track_idx, meas_idx) pairs within the gate; unmatched
    measurements/tracks are left out (caller spawns/coasts).
    """
    if len(track_positions) == 0 or len(meas_positions) == 0:
        return []
    cost = np.linalg.norm(
        track_positions[:, None, :] - meas_positions[None, :, :], axis=2)
    pairs = []
    if _HAVE_SCIPY:
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= gate:
                pairs.append((int(r), int(c)))
    else:  # greedy
        used_t, used_m = set(), set()
        flat = sorted(((cost[i, j], i, j)
                       for i in range(cost.shape[0])
                       for j in range(cost.shape[1])))
        for d, i, j in flat:
            if d > gate:
                break
            if i in used_t or j in used_m:
                continue
            used_t.add(i); used_m.add(j)
            pairs.append((i, j))
    return pairs


# ----------------------------------------------------------------------------
# The fusion engine
# ----------------------------------------------------------------------------
class SZLTrackFusion:
    """Multi-sensor track-fusion engine producing a fused COP track list.

    Consumes measurements from 2+ noisy sensor streams, associates them to
    existing fused tracks by global-nearest-neighbour, runs a tiny Kalman
    update, spawns new tracks for unassociated measurements, and coasts/prunes
    stale tracks. Output is the fused Common-Operating-Picture track list.

    Each sensor stream is a dict: sensor_id -> ndarray (n_meas x 2) of [x, y]
    positions at the current timestep.
    """

    def __init__(self, gate: float = 3.0, init_var: float = 5.0,
                 prune_misses: int = 5):
        self.gate = gate
        self.init_var = init_var
        self.prune_misses = prune_misses
        self.tracks: list = []
        self._next_id = 0
        self._misses: dict = {}   # track_id -> consecutive miss count

    def _spawn(self, z_xy: np.ndarray, t: float, sensor_id: str) -> FusedTrack:
        tr = FusedTrack(
            track_id=self._next_id,
            state=np.array([z_xy[0], z_xy[1], 0.0, 0.0], dtype=float),
            P=np.diag([self.init_var] * 4).astype(float),
        )
        tr.history.append((t, float(z_xy[0]), float(z_xy[1])))
        tr.contributing_sensors.append(sensor_id)
        self._misses[self._next_id] = 0
        self._next_id += 1
        self.tracks.append(tr)
        return tr

    def step(self, sensor_streams: dict, t: float, dt: float = 1.0,
             sensor_noise: Optional[dict] = None) -> list:
        """Process one timestep of measurements from all sensors.

        sensor_streams : {sensor_id: ndarray(n x 2)}
        sensor_noise   : {sensor_id: r} measurement variance per sensor (optional)
        Returns the current fused COP track list.
        """
        sensor_noise = sensor_noise or {}
        # 1. predict all existing tracks forward
        for tr in self.tracks:
            tr.predict(dt)

        updated_ids = set()
        # 2. process each sensor stream in turn (sequential fusion)
        for sensor_id, meas in sensor_streams.items():
            meas = np.atleast_2d(np.asarray(meas, dtype=float))
            if meas.size == 0 or meas.shape[1] != 2:
                continue
            r = sensor_noise.get(sensor_id, 1.0)
            track_pos = np.array([[tr.state[0], tr.state[1]] for tr in self.tracks]) \
                if self.tracks else np.empty((0, 2))
            pairs = _associate(track_pos, meas, self.gate)
            matched_meas = set()
            for ti, mi in pairs:
                self.tracks[ti].update(meas[mi], r, t, sensor_id)
                updated_ids.add(self.tracks[ti].track_id)
                matched_meas.add(mi)
            # spawn for unassociated measurements
            for mi in range(meas.shape[0]):
                if mi not in matched_meas:
                    tr = self._spawn(meas[mi], t, sensor_id)
                    updated_ids.add(tr.track_id)

        # 3. update miss counters + prune stale tracks
        survivors = []
        for tr in self.tracks:
            if tr.track_id in updated_ids:
                self._misses[tr.track_id] = 0
                survivors.append(tr)
            else:
                self._misses[tr.track_id] += 1
                if self._misses[tr.track_id] <= self.prune_misses:
                    survivors.append(tr)
        self.tracks = survivors
        return self.tracks

    def cop_snapshot(self) -> np.ndarray:
        """Return current fused positions (n_tracks x 2) as the COP snapshot."""
        if not self.tracks:
            return np.empty((0, 2))
        return np.array([[tr.state[0], tr.state[1]] for tr in self.tracks])


if __name__ == "__main__":
    # tiny smoke test: two sensors observing 2 crossing targets
    rng = np.random.default_rng(1)
    fuse = SZLTrackFusion(gate=4.0)
    for t in range(10):
        true_a = np.array([t, 2 * t])
        true_b = np.array([10 - t, t])
        s1 = np.vstack([true_a + rng.normal(0, 0.3, 2),
                        true_b + rng.normal(0, 0.3, 2)])
        s2 = np.vstack([true_a + rng.normal(0, 0.5, 2),
                        true_b + rng.normal(0, 0.5, 2)])
        fuse.step({"radar": s1, "eo": s2}, t=t,
                  sensor_noise={"radar": 0.3, "eo": 0.5})
    print(f"fused COP tracks: {len(fuse.tracks)}")
    for tr in fuse.tracks:
        print(f"  track {tr.track_id}: pos=({tr.state[0]:.1f},{tr.state[1]:.1f}) "
              f"updates={tr.n_updates} sensors={tr.contributing_sensors}")
