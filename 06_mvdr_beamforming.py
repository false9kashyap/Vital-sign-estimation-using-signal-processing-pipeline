"""
06_mvdr_beamforming.py  -  Algorithm 1, line 6:  B = BeamformMVDR(ADC, T, N_theta, fc, win)

For each detected patient (r_m, a_m):
  1. take the slab of range bins [r_m - win, r_m + win] and all frames as snapshots,
  2. covariance matrix R of the MOVING echoes (static echoes removed: only chests shape the beam),
  3. weights   w = R^-1 C (C^H R^-1 C)^-1 f
               C = [a_m, a_j ...]  steering vectors of this patient and (optionally) of the other
                   patients that share its range slab,   f = [1, 0, ...]
               -> unit gain on this patient, a NULL on the neighbour (LCMV; MVDR if no null is used),
  4. y[f] = w^H x[f]     (one complex time series per patient).

The null on a neighbour stops its breathing and heartbeat from leaking into this patient's signal.
It is only used when the neighbour is in the same range slab, is not almost in the same direction
(NULL_MIN_SEP_DEG) and does not amplify the noise too much (MAX_NOISE_GAIN_DB). Otherwise the plain
MVDR of the paper is used (MVDR also suppresses strong interferers by itself).

Input : slow_time.npy, targets.npy
Output: beamformed.npy          (frames, M) complex
        beamformed.png
"""

import numpy as np
import matplotlib.pyplot as plt
from config import *
from array_tools import steering


# ============================================================
# SETTINGS OF THIS SCRIPT
# ============================================================

WIN = 1                     # slab half-width in range bins (paper: not given)
DIAG_LOADING = 1e-2         # added for numerical stability (NOT in the paper); 0 = pure formula

NULL_STEERING = True        # put a null on the other patients (NOT in the paper)
NULL_RANGE_BINS = 12        # a neighbour is nulled only if its range is within this many bins
NULL_MIN_SEP_DEG = 5.0      # ... and its angle differs by at least this much
MAX_NOISE_GAIN_DB = 10.0    # ... and the weights do not raise white noise by more than this (vs a/N)


# ============================================================
# 1. LOAD
# ============================================================

S = np.load("slow_time.npy").astype(np.complex128)      # [frames, RX, bins]

targets = np.load("targets.npy")                        # [M, 2]

N_FRAMES = S.shape[0]

print("Slow-time shape:", S.shape)
print("Targets:", targets.tolist())

S_move = S - S.mean(axis=0, keepdims=True)              # moving echoes, used for the covariance only


def sin_of(a_bin):
    return 2.0 * (a_bin - N_THETA // 2) / N_THETA


# ============================================================
# 2. BEAMFORMER FOR EVERY TARGET
# ============================================================

B = np.zeros((N_FRAMES, len(targets)), dtype=np.complex128)

for m, (r, a_bin) in enumerate(targets):

    # ---- snapshots from the slab ----
    slab = S_move[:, :, r - WIN:r + WIN + 1]                     # [frames, RX, 2WIN+1]

    X = slab.transpose(1, 0, 2).reshape(N_RX, -1)                # [RX, snapshots]

    R_cov = (X @ X.conj().T) / X.shape[1]

    R_cov = R_cov + DIAG_LOADING * np.real(np.trace(R_cov)) / N_RX * np.eye(N_RX)

    # ---- steering vector of this patient's angle ----
    a_vec = steering(sin_of(a_bin))[:, 0]

    # ---- constraints: this patient (gain 1) + neighbours (gain 0) ----
    cols = [a_vec]

    if NULL_STEERING:
        for j, (r2, a2) in enumerate(targets):
            if j == m:
                continue
            sep = abs(np.degrees(np.arcsin(np.clip(sin_of(a2), -1, 1)) - np.arcsin(np.clip(sin_of(a_bin), -1, 1))))
            if abs(int(r2) - int(r)) <= NULL_RANGE_BINS and sep >= NULL_MIN_SEP_DEG and len(cols) < N_RX - 1:
                cols.append(steering(sin_of(a2))[:, 0])

    Ri = np.linalg.inv(R_cov)

    def weights(C):
        f = np.zeros(C.shape[1], dtype=complex)
        f[0] = 1.0
        return Ri @ C @ np.linalg.solve(C.conj().T @ Ri @ C, f)

    w = weights(np.column_stack(cols))

    mode = "MVDR" if len(cols) == 1 else f"LCMV ({len(cols) - 1} null)"

    # ---- check the noise gain, fall back to plain MVDR ----
    gain_db = 10 * np.log10(np.real(w.conj() @ w) * N_RX)

    if len(cols) > 1 and gain_db > MAX_NOISE_GAIN_DB:
        print(f"Target {m + 1}: null would raise the noise by {gain_db:.1f} dB, plain MVDR used instead")
        w = weights(a_vec[:, None])
        mode = "MVDR"
        gain_db = 10 * np.log10(np.real(w.conj() @ w) * N_RX)

    # ---- apply to the gated data of this patient (original signal, static part included) ----
    B[:, m] = w.conj() @ S[:, :, r].T

    print(f"Target {m + 1}: {mode}, noise gain {gain_db:.1f} dB, mean |y| = {np.abs(B[:, m]).mean():.1f}")


# ============================================================
# 3. PLOT (first 20 s)
# ============================================================

t = np.arange(N_FRAMES) / FS_SLOW

n_show = min(N_FRAMES, int(20 * FS_SLOW))

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for m in range(B.shape[1]):
    ax[0].plot(t[:n_show], np.abs(B[:n_show, m]), label=f"target {m + 1}")
    ax[1].plot(t[:n_show], np.unwrap(np.angle(B[:n_show, m])), label=f"target {m + 1}")

ax[0].set_ylabel("|y|")
ax[0].set_title("Beamformed signal of each patient")
ax[0].legend()
ax[0].grid(True)

ax[1].set_ylabel("Unwrapped phase (rad)")
ax[1].set_xlabel("Time (s)")
ax[1].grid(True)

plt.tight_layout()

plt.savefig("beamformed.png", dpi=300)

if SHOW_PLOTS:
    plt.show()


# ============================================================
# 4. SAVE
# ============================================================

np.save("beamformed.npy", B)

print("\nSaved:")
print("  beamformed.npy")
print("  beamformed.png")
