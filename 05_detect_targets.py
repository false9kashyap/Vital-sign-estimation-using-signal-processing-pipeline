"""
05_detect_targets.py  -  Algorithm 1, line 5:  T = DetectTargets(RAmap, M, mD, B, N_theta, R_mean)

Pick the M strongest PEAKS of the range-angle map (the MUSIC-sharpened map from script 04),
rejecting any peak that is too close to an already accepted one (too close in range AND in angle):

    r_min     = ceil(mD / dR)                       dR = metres per range bin
    a_min     = ceil( asin(mD / R_mean) / (180 / N_THETA) )

Only local maxima are candidates, so one patient gives one candidate, and the second target is
a real peak of the data, not a cell chosen by the separation rule.

Input : range_angle_map.npy     (N_BINS, N_THETA)
Output: targets.npy             (M, 2)  [range bin, angle bin]  sorted by angle (left to right)
        targets.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from config import *


# ============================================================
# SETTINGS OF THIS SCRIPT   (the paper does not give these values)
# ============================================================

MD = 0.5                    # minimum physical separation of two patients (m)
RMEAN = 2.0                 # nominal distance of the patients (m), for the angle limit

# Minimum separation, in bins (None = computed from MD / RMEAN as in the paper).
# Small on purpose: with MUSIC the peaks are narrow, so patients only a few degrees apart can be
# told apart. With the paper's rule (MD = 0.5 m) the exclusion zone is 26 range bins x 11 angle bins.
R_MIN_BINS = 6
A_MIN_BINS = 4

# A candidate must be the maximum of the map inside a window of (2*LOCAL_R+1) x (2*LOCAL_A+1) cells
LOCAL_R = 3
LOCAL_A = 2

# Search window (m): ignore very close clutter and, if you like, far walls
SEARCH_MIN_M = 0.3
SEARCH_MAX_M = None         # None = up to the end of the map

# The paper rejects cells with r < r_min or r > R - r_min. With the range scale of this
# dataset (about 2 cm per bin), r_min is 26 bins, so that literal rule would reject every
# target farther than about 2.0 m. Default False: only keep the beamforming slab inside the map.
PAPER_EDGE_RULE = False
SLAB_HALF_WIDTH = 1         # must match WIN in script 06

# TEST A GUESS ABOUT WHERE A PATIENT IS: give one angle (degrees) per patient, left to right,
# or None to keep the detected one, e.g.  FORCE_ANGLES_DEG = [None, 45]
# The range bin is taken from the detection. If HR/RR of that patient then become correct,
# the guess was right; if not, it was wrong. Use 09_find_patients.py to find the positions from the data.
FORCE_ANGLES_DEG = None

# The same for the range: one range bin per patient (same order as above) or None, e.g. [None, 120].
FORCE_RANGE_BINS = None


# ============================================================
# 1. LOAD MAP
# ============================================================

RA = np.load("range_angle_map.npy")

R, A = RA.shape

print("Range-angle map shape:", RA.shape)


# ============================================================
# 2. SEPARATION LIMITS
# ============================================================

dR = RANGE_BIN_M

r_min = int(np.ceil(MD / dR)) if R_MIN_BINS is None else R_MIN_BINS

d_theta = np.degrees(np.arcsin(min(MD / RMEAN, 1.0)))

a_min = int(np.ceil(d_theta / (180.0 / N_THETA))) if A_MIN_BINS is None else A_MIN_BINS

print(f"\nMinimum separation: {r_min} range bins ({r_min * dR:.2f} m), "
      f"{a_min} angle bins ({a_min * 180 / N_THETA:.1f} deg)")

lo = max(int(SEARCH_MIN_M / dR), SLAB_HALF_WIDTH)
hi = R - 1 - SLAB_HALF_WIDTH if SEARCH_MAX_M is None else min(int(SEARCH_MAX_M / dR), R - 1 - SLAB_HALF_WIDTH)

if PAPER_EDGE_RULE:
    lo = max(lo, r_min)
    hi = min(hi, R - 1 - r_min)

print(f"Search window: range bins {lo} ... {hi}")


# ============================================================
# 3. PEAK PICKING (strongest first, local maxima only)
# ============================================================

local_max = maximum_filter(RA, size=(2 * LOCAL_R + 1, 2 * LOCAL_A + 1), mode="nearest")

is_peak = (RA == local_max) & (RA > 0)

cells = np.argwhere(is_peak)

cells = sorted(cells.tolist(), key=lambda c: RA[c[0], c[1]], reverse=True)

targets = []

for r, a in cells:

    if r < lo or r > hi:
        continue

    ok = True

    for (r0, a0) in targets:
        if abs(r - r0) < r_min and abs(a - a0) < a_min:
            ok = False
            break

    if ok:
        targets.append((r, a))

    if len(targets) == M_TARGETS:
        break

if len(targets) < M_TARGETS:
    print(f"\nWARNING: only {len(targets)} of {M_TARGETS} targets found. "
          "Reduce MD / RMEAN / R_MIN_BINS / A_MIN_BINS, or widen the search window.")

SORT_COL = 0 if TARGET_ORDER == "range" else 1                # 1 = by angle (left to right), 0 = by range (near to far)

targets = np.array(sorted(targets, key=lambda t: t[SORT_COL]))

if FORCE_ANGLES_DEG is not None:

    if len(FORCE_ANGLES_DEG) != len(targets):
        raise ValueError(f"FORCE_ANGLES_DEG needs {len(targets)} entries, got {len(FORCE_ANGLES_DEG)}")

    for i, ang in enumerate(FORCE_ANGLES_DEG):
        if ang is not None:
            targets[i, 1] = int(round(N_THETA // 2 + (N_THETA / 2) * np.sin(np.radians(ang))))
            print(f"Target {i + 1}: angle forced to {ang} deg (bin {targets[i, 1]})")

    targets = targets[np.argsort(targets[:, SORT_COL])]

if FORCE_RANGE_BINS is not None:

    if len(FORCE_RANGE_BINS) != len(targets):
        raise ValueError(f"FORCE_RANGE_BINS needs {len(targets)} entries, got {len(FORCE_RANGE_BINS)}")

    for i, rb in enumerate(FORCE_RANGE_BINS):
        if rb is not None:
            targets[i, 0] = int(rb)
            print(f"Target {i + 1}: range bin forced to {rb}")

    targets = targets[np.argsort(targets[:, SORT_COL])]


# ============================================================
# 4. PRINT
# ============================================================

sin_theta = 2.0 * (targets[:, 1] - N_THETA // 2) / N_THETA

angles = np.degrees(np.arcsin(sin_theta))

print("\nDetected targets (" + ("left to right" if TARGET_ORDER != "range" else "near to far") + "):")
for i, ((r, a), ang) in enumerate(zip(targets, angles), start=1):
    print(f"  target {i}: range bin {r}  ({r * dR:.2f} m),  angle bin {a}  ({ang:.1f} deg)")


# A target that sits right on the edge of the exclusion zone of another target was probably
# placed there by the separation rule, not by a patient.
for i in range(len(targets)):
    for j in range(i + 1, len(targets)):
        dr = abs(int(targets[i, 0]) - int(targets[j, 0]))
        da = abs(int(targets[i, 1]) - int(targets[j, 1]))
        if (dr < r_min and da in (a_min, a_min + 1)) or (da < a_min and dr in (r_min, r_min + 1)):
            print(f"\nNOTE: targets {i + 1} and {j + 1} are exactly on the edge of each other's exclusion zone "
                  f"({dr} range bins, {da} angle bins apart; limits {r_min} and {a_min}).\n"
                  "      One of them was probably placed by the separation rule, not by a patient. "
                  "Reduce R_MIN_BINS / A_MIN_BINS and run 09_find_patients.py.")

for ang in angles:
    if abs(ang) < 1.5:
        print("\nNOTE: a target sits at (almost) exactly 0 degrees. That is often a STATIC echo straight ahead "
              "(bed frame, wall, ceiling), not a patient.\n"
              "      Check that static removal is on (REMOVE_STATIC_FOR_MAP in config.py) and run 09_find_patients.py.")
        break


# ============================================================
# 5. PLOT
# ============================================================

axis_angle = np.degrees(np.arcsin(2.0 * (np.arange(N_THETA) - N_THETA // 2) / N_THETA))
axis_range = np.arange(R) * dR

plt.figure(figsize=(9, 6))

plt.pcolormesh(axis_angle, axis_range, 10 * np.log10(RA / RA.max() + 1e-12), shading="auto", vmin=-40, vmax=0)

plt.colorbar(label="Power (dB, relative)")

plt.plot(angles, targets[:, 0] * dR, "wo", markersize=12, markerfacecolor="none", markeredgewidth=2)

for i, (ang, r) in enumerate(zip(angles, targets[:, 0]), start=1):
    plt.text(ang, r * dR + 0.15, str(i), color="white", ha="center", fontsize=11)

plt.xlabel("Angle (deg)")
plt.ylabel("Range (m)   [approximate]")
plt.title("Detected patients (circles)")

plt.xlim(-60, 60)

plt.tight_layout()

plt.savefig("targets.png", dpi=300)

if SHOW_PLOTS:
    plt.show()


# ============================================================
# 6. SAVE
# ============================================================

np.save("targets.npy", targets)

print("\nSaved:")
print("  targets.npy")
print("  targets.png")
