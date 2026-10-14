"""
04_range_angle_map.py  -  Algorithm 1, line 4:  RAmap = ComputeRangeAngle(RP, N_theta)      (+ MUSIC)

Angle processing across the RX antennas for every range bin.

  Bartlett : FFT across the antennas, zero-padded to N_THETA bins (the paper's map). Calibrated
             power, but with 4 antennas the beam is ~25 degrees wide: close patients merge.
  MUSIC    : pseudo-spectrum from the noise subspace of the covariance of the MOVING echoes.
             Very sharp, but not a power measure.
  combined : Bartlett x MUSIC. This is the map used by the target detector (05).
             (USE_MUSIC = False in config.py gives the plain Bartlett map of the paper.)

Input : slow_time.npy               (frames, RX, N_BINS)
Output: range_angle_map.npy         (N_BINS, N_THETA)  detection map (combined)
        range_angle_bartlett.npy    plain Bartlett map
        range_angle_music.npy       MUSIC pseudo-spectrum (max 1 in every range bin)
        range_angle_map.png
"""

import numpy as np
import matplotlib.pyplot as plt
from config import *
from array_tools import compute_maps, sin_axis


# ============================================================
# 1. LOAD SLOW-TIME DATA
# ============================================================

S = np.load("slow_time.npy")                       # [frames, RX, bins]

print("Slow-time shape:", S.shape)

if REMOVE_STATIC_FOR_MAP:
    print("Static echoes removed for the map.")

print("MUSIC:", "ON" if USE_MUSIC else "OFF (plain Bartlett map)")


# ============================================================
# 2. MAPS
# ============================================================

print("\nComputing range-angle maps...")

RA, RA_bartlett, RA_music, n_src = compute_maps(S)

print("Range-angle map shape:", RA.shape)

busy = n_src[8:]
print(f"MUSIC source count per range bin: 1 source in {(busy == 1).sum()} bins, "
      f"2 in {(busy == 2).sum()}, 3 in {(busy >= 3).sum()}")


# ============================================================
# 3. PLOT
# ============================================================

angles = np.degrees(np.arcsin(sin_axis()))

range_axis = np.arange(N_BINS) * RANGE_BIN_M


def db(x):
    return 10 * np.log10(x / x.max() + 1e-12)


fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for a, data, title in ((ax[0], RA_bartlett, "Bartlett (paper)"),
                       (ax[1], RA, "Detection map (Bartlett x MUSIC)" if USE_MUSIC else "Detection map (Bartlett)")):
    im = a.pcolormesh(angles, range_axis, db(data), shading="auto", vmin=-40, vmax=0)
    a.set_xlabel("Angle (deg)")
    a.set_title(title)
    a.set_xlim(-60, 60)
    plt.colorbar(im, ax=a, label="Power (dB, relative)")

ax[0].set_ylabel("Range (m)   [approximate]")

plt.tight_layout()

plt.savefig("range_angle_map.png", dpi=300)

if SHOW_PLOTS:
    plt.show()


# ============================================================
# 4. SAVE
# ============================================================

np.save("range_angle_map.npy", RA)
np.save("range_angle_bartlett.npy", RA_bartlett)
np.save("range_angle_music.npy", RA_music)

print("\nSaved:")
print("  range_angle_map.npy  (detection map)")
print("  range_angle_bartlett.npy, range_angle_music.npy")
print("  range_angle_map.png")
