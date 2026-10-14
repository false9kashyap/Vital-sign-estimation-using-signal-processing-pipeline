"""
03_slow_time.py  -  one complex sample per frame (the "slow-time" signal)

Averages the chirps inside each frame (of ONE transmitter only), so every frame gives one
complex number per RX antenna and range bin. This is the signal that carries breathing
and heartbeat.

Input : range_fft.npy   (frames, RX, chirps, ADC)
Output: slow_time.npy   (frames, RX, N_BINS) complex64
        motion_energy.png
"""

import numpy as np
import matplotlib.pyplot as plt
from config import *


# ============================================================
# SETTINGS OF THIS SCRIPT
# ============================================================

MIN_BIN = 8                 # ignore the first bins (DC / leakage) in the checks below
N_CHECK_FRAMES = 50         # frames used for the TX check


# ============================================================
# 1. LOAD RANGE FFT
# ============================================================

print("Loading range FFT...")

rf = np.load("range_fft.npy", mmap_mode="r")

N_FRAMES = rf.shape[0]

print("Range FFT shape:", rf.shape)


# ============================================================
# 2. CHECK: ARE THE CHIRPS OF SEVERAL TRANSMITTERS INTERLEAVED?
# ============================================================

# At the strongest range bin, look at the phase step between consecutive chirps.
# One TX only        -> the step is the same for every chirp.
# 2 or 3 TX (TDM)    -> the step repeats every 2 or 3 chirps with big jumps.
# Mixing TX chirps would corrupt the phase, so N_TX in config.py must be right.

print("\nChecking TX pattern...")

chk = np.asarray(rf[:N_CHECK_FRAMES, 0, :, :N_BINS])           # [frames, chirps, bins]

k_best = MIN_BIN + np.argmax(np.abs(chk).mean(axis=(0, 1))[MIN_BIN:])

x = chk[:, :, k_best]                                          # [frames, chirps]

step = np.angle((x[:, 1:] * np.conj(x[:, :-1])).sum(axis=0))   # phase step per chirp

print(f"Range bin used: {k_best}")
print("Phase step between consecutive chirps (deg), first 12:")
print(np.round(np.degrees(step[:12]), 1))


def circ_std(a):
    return np.sqrt(-2 * np.log(max(np.abs(np.mean(np.exp(1j * a))), 1e-12)))


guess = None

for p in (1, 2, 3, 4):
    within = np.mean([circ_std(step[r::p]) for r in range(p) if len(step[r::p]) > 2])
    print(f"  period {p}: spread inside groups = {within:.2f} rad")
    if guess is None and within < 0.3:
        guess = p

if guess is None:
    print("TX pattern unclear: check the printed phase steps by eye.")
elif guess == N_TX:
    print(f"OK: pattern looks like {guess} TX, matches N_TX = {N_TX}.")
else:
    print(f"WARNING: pattern looks like {guess} TX, but N_TX = {N_TX} in config.py. Check it.")


# ============================================================
# 3. SLOW-TIME SIGNAL
# ============================================================

tx_idx = np.arange(TX_INDEX, N_CHIRPS, N_TX)

print(f"\nAveraging {len(tx_idx)} chirps per frame (TX index {TX_INDEX} of {N_TX})...")

S = np.zeros((N_FRAMES, N_RX, N_BINS), dtype=np.complex64)

for f in range(N_FRAMES):

    # frame: [RX, chirps, ADC] -> keep chosen chirps and first N_BINS bins
    frame = rf[f][:, tx_idx, :N_BINS]

    S[f] = frame.mean(axis=1)

    if (f + 1) % 500 == 0:
        print(f"Processed {f + 1}/{N_FRAMES} frames")

print("Slow-time shape:", S.shape)


# ============================================================
# 4. MOTION-ENERGY PER RANGE BIN  (where do things MOVE?)
# ============================================================

# Furniture is static; a chest moves. The variation of the signal over time, per range bin,
# shows where the patients are even when a wall or bed frame gives a stronger echo.

amplitude = np.abs(S).mean(axis=(0, 1))

motion = np.sqrt((np.abs(S - S.mean(axis=0, keepdims=True)) ** 2).mean(axis=(0, 1)))

range_axis = np.arange(N_BINS) * RANGE_BIN_M

top = MIN_BIN + np.argsort(motion[MIN_BIN:])[::-1][:5]

print("\nBins with the most motion (candidate patient ranges):")
for b in top:
    print(f"  bin {b:3d}   {range_axis[b]:.2f} m   motion {motion[b]:.1f}   mean amplitude {amplitude[b]:.1f}")


# ============================================================
# 5. PLOT
# ============================================================

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax[0].plot(range_axis, amplitude)
ax[0].set_ylabel("Mean amplitude")
ax[0].set_title("Echo strength per range bin")
ax[0].grid(True)

ax[1].plot(range_axis, motion, color="tab:orange")
ax[1].set_ylabel("Motion (std over time)")
ax[1].set_xlabel("Range (m)   [approximate]")
ax[1].set_title("Motion energy per range bin")
ax[1].grid(True)

plt.tight_layout()

plt.savefig("motion_energy.png", dpi=300)

if SHOW_PLOTS:
    plt.show()


# ============================================================
# 6. SAVE
# ============================================================

np.save("slow_time.npy", S)

print("\nSaved:")
print("  slow_time.npy")
print("  motion_energy.png")
