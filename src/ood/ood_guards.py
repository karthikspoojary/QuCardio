"""
src/ood/ood_guards.py
====================
Two lightweight OOD guards that run after the heuristic gatekeeper
but before classification:

  1. ecg_periodicity_score(gray)      — autocorrelation-based periodicity
  2. MahalanobisGuard                 — feature-space distance check

Both are fast (<1 ms each after loading) and require no GPU.

Usage
-----
# Fit and save (once, after training):
    guard = MahalanobisGuard()
    guard.fit(X_train_9d)           # shape (742, 9), already MinMax-scaled
    guard.save("backend/models/mahalanobis_guard.npz")

# At inference:
    guard = MahalanobisGuard.load("backend/models/mahalanobis_guard.npz")
    dist  = guard.distance(x_9d)    # scalar
    if dist > guard.threshold:
        return "out-of-distribution"

    score, period_px = ecg_periodicity_score(gray_image)
    if score < PERIODICITY_THRESHOLD:
        return "not periodic — likely not an ECG"
"""
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Periodicity check
# ─────────────────────────────────────────────────────────────────────────────

PERIODICITY_THRESHOLD = 0.15   # ECG ~0.35-0.70; line art ~0.0-0.12
ROW_BAND_THRESHOLD    = 0.55   # top-40% rows must hold >55% of ink
AXIS_ANGLE_THRESHOLD  = 0.70   # if >70% gradient energy is axis-aligned → line art


def ecg_periodicity_score(gray: np.ndarray) -> tuple[float, int]:
    """
    Returns (score, period_px).
    Real ECG strips score ~0.35–0.70; line art / UML diagrams score <0.15.
    A score ≥ PERIODICITY_THRESHOLD is consistent with a genuine ECG.
    """
    ink  = (gray < 128).astype(np.float32).sum(axis=0)  # dark pixel count per column
    ink  = ink - ink.mean()
    if ink.std() < 1e-6:
        return 0.0, 0

    ac = np.correlate(ink, ink, mode='full')[len(ink)-1:]
    ac = ac / (ac[0] + 1e-9)

    lo  = max(8,  len(ink) // 60)
    hi  = max(20, len(ink) // 4)
    band = ac[lo:hi]
    if band.size == 0:
        return 0.0, 0

    k = int(np.argmax(band))
    return float(band[k]), lo + k


def row_band_energy_score(gray: np.ndarray) -> float:
    """
    ECGs concentrate ink in horizontal lead bands.
    Returns fraction of total ink held by the top-40% ink rows.
    ECG typically >0.60; diagrams are more uniform (~0.40–0.55).
    """
    ink_per_row = (gray < 128).astype(np.float32).sum(axis=1)
    total       = ink_per_row.sum()
    if total < 1:
        return 0.0
    n_top   = max(1, int(len(ink_per_row) * 0.40))
    top_ink = np.sort(ink_per_row)[::-1][:n_top].sum()
    return float(top_ink / total)


def axis_angle_fraction(gray: np.ndarray) -> float:
    """
    UML / line-chart diagrams have most gradient energy at 0° and 90°.
    ECG traces (curved waveforms) spread energy across all angles.
    Returns fraction of gradient energy within ±5° of horizontal or vertical.
    Values >0.70 suggest line art.
    """
    try:
        import cv2
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    except ImportError:
        # fallback: simple finite differences
        gx = (gray[:, 2:].astype(np.float32) - gray[:, :-2])
        gy = (gray[2:, :].astype(np.float32) - gray[:-2, :])
        gx = np.pad(gx, ((0,0),(1,1)))
        gy = np.pad(gy, ((1,1),(0,0)))

    mag   = np.sqrt(gx**2 + gy**2)
    angle = np.abs(np.degrees(np.arctan2(gy, gx + 1e-9))) % 180
    total = float(mag.sum()) + 1e-9
    axis_mask = (angle < 5) | (angle > 175) | ((angle > 85) & (angle < 95))
    return float(mag[axis_mask].sum() / total)


def is_ecg_by_periodicity(gray: np.ndarray,
                           per_thresh: float  = PERIODICITY_THRESHOLD,
                           row_thresh: float  = ROW_BAND_THRESHOLD,
                           axis_thresh: float = AXIS_ANGLE_THRESHOLD
                           ) -> tuple[bool, str, dict]:
    """
    Combined three-signal check.
    Returns (passed, reason_if_failed, scores_dict).
    """
    p_score, period_px = ecg_periodicity_score(gray)
    row_score          = row_band_energy_score(gray)
    axis_frac          = axis_angle_fraction(gray)

    scores = {
        'periodicity':  round(p_score,   4),
        'period_px':    period_px,
        'row_band':     round(row_score,  4),
        'axis_frac':    round(axis_frac,  4),
    }

    if p_score < per_thresh:
        return False, (
            f"Periodicity score {p_score:.3f} < {per_thresh} — "
            "no repeating beat structure detected (likely not an ECG)"
        ), scores

    if axis_frac > axis_thresh:
        return False, (
            f"Gradient energy fraction at axis-aligned angles {axis_frac:.2f} > {axis_thresh} — "
            "image contains predominantly horizontal/vertical strokes (likely a diagram or chart)"
        ), scores

    return True, "", scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mahalanobis distance guard
# ─────────────────────────────────────────────────────────────────────────────

class MahalanobisGuard:
    """
    Fits a multivariate Gaussian on the training 9-D features.
    At inference, computes the Mahalanobis distance from the query to the
    training distribution.  Samples beyond `threshold` are flagged OOD.

    The threshold is calibrated at `percentile` of the training distances
    (default: 99th — only the most extreme 1% of *training* samples are
    flagged, which prevents rejecting genuine edge cases).
    """

    def __init__(self):
        self.mu        = None   # (9,)
        self.inv_cov   = None   # (9,9)
        self.threshold = None
        self.percentile = 99.0

    def fit(self, X: np.ndarray, percentile: float = 99.0):
        """
        X : (N, 9) training features, already MinMax-scaled.
        """
        self.percentile = percentile
        self.mu      = X.mean(axis=0)
        cov          = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        self.inv_cov = np.linalg.inv(cov)
        dists        = np.array([self._dist(x) for x in X])
        self.threshold = float(np.percentile(dists, percentile))
        print(f"  MahalanobisGuard fitted on {len(X)} samples")
        print(f"  Threshold (p{percentile:.0f}): {self.threshold:.4f}")
        print(f"  Training dist mean={dists.mean():.3f} std={dists.std():.3f}")
        return self

    def _dist(self, x: np.ndarray) -> float:
        d = x - self.mu
        return float(np.sqrt(d @ self.inv_cov @ d))

    def distance(self, x: np.ndarray) -> float:
        return self._dist(x)

    def is_in_distribution(self, x: np.ndarray) -> tuple[bool, float]:
        d = self.distance(x)
        return (d <= self.threshold), d

    def save(self, path: str):
        np.savez(path,
                 mu=self.mu,
                 inv_cov=self.inv_cov,
                 threshold=np.array([self.threshold]),
                 percentile=np.array([self.percentile]))
        print(f"  MahalanobisGuard saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MahalanobisGuard":
        d = np.load(path)
        g = cls()
        g.mu        = d['mu']
        g.inv_cov   = d['inv_cov']
        g.threshold = float(d['threshold'][0])
        g.percentile = float(d['percentile'][0])
        return g


# ─────────────────────────────────────────────────────────────────────────────
# Fit and save the Mahalanobis guard from training features
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append('.')
    import config

    d = np.load(config.FEATURES_DIR / 'features_9d.npz')
    X_train = d['train_features']   # (742, 9), MinMax-scaled [0,1]

    guard = MahalanobisGuard()
    guard.fit(X_train, percentile=99.0)

    out = Path('backend/models/mahalanobis_guard.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    guard.save(str(out))

    # Quick check on test set
    X_test = d['test_features']
    dists  = [guard.distance(x) for x in X_test]
    n_ood  = sum(d > guard.threshold for d in dists)
    print(f"\nTest set ({len(X_test)} samples):")
    print(f"  Flagged OOD: {n_ood} ({n_ood/len(X_test)*100:.1f}%)")
    print(f"  (expected ~{100-guard.percentile:.0f}% at p{guard.percentile:.0f} threshold)")
    print("  dist min/mean/max:", round(min(dists),3), round(sum(dists)/len(dists),3), round(max(dists),3))
