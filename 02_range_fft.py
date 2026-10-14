"""
02_range_fft.py  -  Algorithm 1, line 3:  RP = FFT(ADC, axis=-1)

Input : radar_cube.npy   (frames, RX, chirps, ADC)
Output: range_fft.npy    same shape (complex64)
        range_axis.npy   range of each kept bin (m)
        range_profile.png
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.format import open_memmap
from config import *


# ============================================================
# SETTINGS OF THIS SCRIPT
# ============================================================

MIN_BIN = 8        # ignore the first bins when looking for the strongest echo (DC / leakage)


# ============================================================
# 1. LOAD CUBE
# ============================================================

print("Loading radar cube...")

X = np.load("radar_cube.npy", mmap_mode="r")

N_FRAMES = X.shape[0]

print("Cube shape:", X.shape)


# ============================================================
# 2. CREATE OUTPUT FILE FOR RANGE FFT
# ============================================================

range_fft = open_memmap(
    "range_fft.npy",
    mode="w+",
    dtype=np.complex64,
    shape=X.shape
)


# ============================================================
# 3. RANGE FFT
# ============================================================

print("\nPerforming Range FFT...")

for f in range(N_FRAMES):

    # One frame: shape = [RX, Chirps, ADC]
    frame = X[f]

    # FFT along the ADC-sample dimension
    rf = np.fft.fft(frame, axis=-1)

    range_fft[f] = rf.astype(np.complex64)

    if (f + 1) % 100 == 0:
        print(f"Processed {f + 1}/{N_FRAMES} frames")

range_fft.flush()

print("\nRange FFT completed.")
print("Range FFT shape:", range_fft.shape)


# ============================================================
# 4. RANGE AXIS  (bin k  ->  k * RANGE_BIN_M)
# ============================================================

bins = np.arange(N_BINS)
range_axis = bins * RANGE_BIN_M

print(f"\nRange bin spacing: {RANGE_BIN_M * 100:.2f} cm")
print(f"Unambiguous range: {range_axis[-1]:.2f} m")


# ============================================================
# 5. EXAMPLE RANGE PROFILE (first 20 frames, all RX and chirps)
# ============================================================

first = np.asarray(range_fft[:20])                 # [20, RX, chirps, ADC]

magnitude = np.abs(first)

range_profile = magnitude[..., :N_BINS].mean(axis=(0, 1, 2))


# ============================================================
# 6. CHECK THE DATA: REAL-LIKE (MIRRORED) SIGNAL?
# ============================================================

# For true I/Q data the range FFT has energy in the positive half only.
# If the negative half carries similar energy, the data behave like a real signal
# and are mirrored. Using only the first N_ADC/2 bins (as we do) is still correct.
power_all = (magnitude ** 2).mean(axis=(0, 1, 2))
pos_energy = power_all[1:N_ADC // 2].sum()
neg_energy = power_all[N_ADC // 2 + 1:].sum()

print(f"\nEnergy in negative half / positive half: {neg_energy / pos_energy:.2f}")
print("(close to 0 = true I/Q data; close to 1 = mirrored, real-like data)")


# ============================================================
# 7. PLOT RANGE PROFILE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(range_axis, range_profile)

plt.xlabel("Range (m)   [approximate, see config.py]")
plt.ylabel("Magnitude")
plt.title("Range Profile (mean of first 20 frames)")

plt.grid(True)
plt.xlim(0, range_axis[-1])

plt.tight_layout()

plt.savefig("range_profile.png", dpi=300)

if SHOW_PLOTS:
    plt.show()


# ============================================================
# 8. PRINT STRONGEST RANGE PEAK
# ============================================================

peak_index = MIN_BIN + np.argmax(range_profile[MIN_BIN:])

peak_range = range_axis[peak_index]

print("\nStrongest range peak (bins below MIN_BIN ignored):")
print(f"Range bin : {peak_index}")
print(f"Range     : {peak_range:.3f} m")


# ============================================================
# 9. SAVE RANGE AXIS
# ============================================================

np.save("range_axis.npy", range_axis)

print("\nSaved:")
print("  range_fft.npy")
print("  range_axis.npy")
print("  range_profile.png")
