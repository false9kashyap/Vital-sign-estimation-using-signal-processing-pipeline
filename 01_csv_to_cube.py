"""
01_csv_to_cube.py  -  CSV (rows = chirps, 1024 columns) -> 4-D radar cube

Output: radar_cube.npy   shape (frames, RX, chirps, ADC samples), complex64
"""

import numpy as np
from config import *


# ============================================================
# 1. READ CSV
# ============================================================

print("Reading CSV...")

data = np.loadtxt(
    CSV_FILE,
    delimiter=",",
    skiprows=1,
    dtype=np.complex64
)

print("CSV shape:", data.shape)


# ============================================================
# 2. CHECK
# ============================================================

expected_columns = N_RX * N_ADC

if data.shape[1] != expected_columns:
    raise ValueError(
        f"Expected {expected_columns} columns, "
        f"found {data.shape[1]}"
    )

total_chirps = data.shape[0]

if total_chirps % N_CHIRPS != 0:
    raise ValueError(
        f"{total_chirps} rows is not divisible by "
        f"{N_CHIRPS} chirps/frame"
    )

N_FRAMES = total_chirps // N_CHIRPS

print("Frames:", N_FRAMES)
print(f"Recording length: {N_FRAMES * FRAME_PERIOD:.1f} s at {FS_SLOW:.1f} frames/s")


# ============================================================
# 3. RESHAPE
# ============================================================

# Current:  (total_chirps, 1024)
# First:    (frames, chirps, RX, ADC)
cube = data.reshape(N_FRAMES, N_CHIRPS, N_RX, N_ADC)

# Reorder:  (frames, RX, chirps, ADC)
X = cube.transpose(0, 2, 1, 3)


# ============================================================
# 4. VERIFY
# ============================================================

print("Final cube shape:", X.shape)

expected_shape = (N_FRAMES, N_RX, N_CHIRPS, N_ADC)

if X.shape == expected_shape:
    print("SUCCESS: Cube created!")
else:
    print("ERROR: Wrong cube shape.")


# ============================================================
# 5. SAVE
# ============================================================

np.save("radar_cube.npy", np.ascontiguousarray(X))

print("Saved as radar_cube.npy")
