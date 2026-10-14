
"""
09_find_patients.py
Diagnostic for target locations and vital-sign spectral peaks.

Input:
    slow_time.npy

Output:
    find_patients.png
    find_patients.csv

HR_PSD and RR_PSD use the phase-difference signal and the same
Welch PSD settings as the spectral baseline in 07_vital_signs.py.

The additional HR peak list helps diagnose missed heart-rate components.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import maximum_filter
from scipy.signal import (
    butter, sosfiltfilt, welch, detrend, find_peaks
)

from config import *
from array_tools import compute_maps


# ============================================================
# SETTINGS
# ============================================================

WIN = 1
MAX_CANDIDATES = 10
MIN_PEAK_DB = -25.0

R_SEP_BINS = 4
A_SEP_BINS = 4

MIN_RANGE_BIN = 8
MAX_ANGLE_DEG = 75.0

DIAG_LOADING = 1e-2

# Number of strong HR-band spectral peaks to display per candidate
N_HR_PEAKS = 8
HR_BAND = (0.8, 3.0)   # Heart-rate band in Hz
RR_BAND = (0.1, 0.6)   # Respiration band in Hz
FILTER_ORDER = 4


# ============================================================
# 1. LOAD SLOW-TIME DATA
# ============================================================

S = np.load("slow_time.npy").astype(np.complex128)

if S.ndim != 3:
    raise ValueError(
        f"slow_time.npy must have shape [frames, RX, bins], got {S.shape}"
    )

N_FRAMES, N_RX_DATA, N_BINS_DATA = S.shape

if N_RX_DATA != N_RX:
    raise ValueError(
        f"RX mismatch: data has {N_RX_DATA} receivers, "
        f"but config.py specifies N_RX={N_RX}"
    )

print("Slow-time shape:", S.shape)
print(f"Slow-time rate: {FS_SLOW:.2f} Hz")
print(f"Recording duration: {N_FRAMES / FS_SLOW:.2f} seconds")


# ============================================================
# 2. BUILD RANGE-ANGLE MOTION MAP
# ============================================================

sin_theta = (
    2.0 * (np.arange(N_THETA) - N_THETA // 2) / N_THETA
)

sin_theta = np.clip(sin_theta, -1.0, 1.0)
angles = np.degrees(np.arcsin(sin_theta))

steering = np.exp(
    1j * 2 * np.pi * RX_POS[:, None] * sin_theta[None, :]
)

P, _, _, _ = compute_maps(S)

P = np.asarray(P, dtype=float)

if P.shape != (N_BINS_DATA, N_THETA):
    raise ValueError(
        f"Unexpected map shape {P.shape}; expected "
        f"({N_BINS_DATA}, {N_THETA})"
    )

P_db = 10 * np.log10(P / (np.max(P) + 1e-30) + 1e-12)


# ============================================================
# 3. DETECT CANDIDATE RANGE-ANGLE PEAKS
# ============================================================

P_search = P_db.copy()

P_search[:MIN_RANGE_BIN, :] = -np.inf
P_search[:, np.abs(angles) > MAX_ANGLE_DEG] = -np.inf

local_max = maximum_filter(
    P_search,
    size=(2 * R_SEP_BINS + 1, 2 * A_SEP_BINS + 1)
)

cells = np.argwhere(
    (P_search == local_max) & (P_search > MIN_PEAK_DB)
)

# Diagnostic test: evaluate selected range bins at approximately 0 degrees.
# These are investigation points, not confirmed patient locations.
zero_angle_idx = int(np.argmin(np.abs(angles)))

test_bins = [97, 102, 109, 119]

cells = [
    (r, zero_angle_idx)
    for r in test_bins
    if MIN_RANGE_BIN <= r < N_BINS_DATA
]


# ============================================================
# 4. SIGNAL-PROCESSING FUNCTIONS
# ============================================================

def remove_static(y):
    """Subtract the complex mean, matching DC_REMOVAL='mean' in script 07."""
    y = np.asarray(y, dtype=np.complex128)
    return y - np.mean(y)


def bandpass(x, band):
    """Zero-phase Butterworth band-pass."""
    sos = butter(
        FILTER_ORDER,
        band,
        btype="bandpass",
        fs=FS_SLOW,
        output="sos"
    )
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def psd_analysis(x, band):
    """
    Welch PSD, using the baseline settings from script 07.

    Returns:
        rate_bpm: strongest PSD peak in the requested band
        clarity_db: peak PSD / median PSD in the band
        f: frequency axis
        Pxx: PSD
    """

    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 16 or not np.all(np.isfinite(x)):
        return np.nan, np.nan, np.array([]), np.array([])

    nperseg = min(2048, n)
    nfft = max(
        8192,
        2 ** int(np.ceil(np.log2(nperseg)))
    )

    f, Pxx = welch(
        x,
        fs=FS_SLOW,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        detrend="linear",
        scaling="density"
    )

    band_idx = np.where(
        (f >= band[0]) & (f <= band[1])
    )[0]

    if len(band_idx) == 0:
        return np.nan, np.nan, f, Pxx

    k = band_idx[np.argmax(Pxx[band_idx])]
    f_peak = f[k]

    # Parabolic interpolation around the tallest PSD bin
    if 0 < k < len(Pxx) - 1:
        denominator = Pxx[k - 1] - 2 * Pxx[k] + Pxx[k + 1]

        if abs(denominator) > 1e-20:
            f_peak += (
                0.5
                * (Pxx[k - 1] - Pxx[k + 1])
                / denominator
                * (f[1] - f[0])
            )

    peak_power = np.max(Pxx[band_idx])
    median_power = np.median(Pxx[band_idx])

    clarity_db = 10 * np.log10(
        (peak_power + 1e-30) / (median_power + 1e-30)
    )

    return 60.0 * f_peak, clarity_db, f, Pxx


def strongest_hr_peaks(f, Pxx, limit=N_HR_PEAKS):
    """
    Find the strongest local PSD maxima in the HR band.
    Returns peak frequencies in bpm, ordered by PSD power.
    """

    if len(f) == 0 or len(Pxx) == 0:
        return []

    indices, _ = find_peaks(Pxx)

    indices = [
        i for i in indices
        if HR_BAND[0] <= f[i] <= HR_BAND[1]
    ]

    indices.sort(
        key=lambda i: Pxx[i],
        reverse=True
    )

    results = []

    for i in indices[:limit]:
        f_peak = f[i]

        # Apply the same parabolic refinement used by the PSD estimator
        if 0 < i < len(Pxx) - 1:
            denominator = Pxx[i - 1] - 2 * Pxx[i] + Pxx[i + 1]

            if abs(denominator) > 1e-20:
                f_peak += (
                    0.5
                    * (Pxx[i - 1] - Pxx[i + 1])
                    / denominator
                    * (f[1] - f[0])
                )

        results.append(60.0 * f_peak)

    return results



# ============================================================
# DIAGNOSTIC: CHECK EACH RECEIVER AT FIXED RANGE BINS
# ============================================================

print("\n" + "=" * 65)
print("PER-RECEIVER HR CHECK (before MVDR beamforming)")
print("=" * 65)


# ============================================================
# CHECK WHETHER RX CHANNELS CONTAIN DISTINCT SIGNALS
# ============================================================

print("\n" + "=" * 65)
print("RECEIVER SIGNAL CORRELATION CHECK")
print("=" * 65)

for r_check in [97, 109, 119]:

    if not (0 <= r_check < N_BINS_DATA):
        continue

    signals = []

    for rx in range(N_RX):
        y_rx = remove_static(S[:, rx, r_check])
        phase_rx = np.unwrap(np.angle(y_rx))

        dphase_rx = detrend(
            np.diff(phase_rx) * FS_SLOW,
            type="linear"
        )

        signals.append(dphase_rx)

    signals = np.asarray(signals)

    corr = np.corrcoef(signals)

    print(f"\nRange bin {r_check}:")
    print("Phase-difference correlation matrix:")
    print(np.array2string(
        corr,
        precision=3,
        suppress_small=True
    ))

    reference_norm = np.linalg.norm(signals[0]) + 1e-12

    for rx in range(1, N_RX):
        relative_difference = (
            np.linalg.norm(signals[rx] - signals[0])
            / reference_norm
        )

        print(
            f"RX{rx} vs RX0: "
            f"relative difference = {relative_difference:.5f}"
        )

for r_check in [97, 109, 119]:

    if not (0 <= r_check < N_BINS_DATA):
        continue

    print(
        f"\nRange bin {r_check} "
        f"({r_check * RANGE_BIN_M:.3f} m):"
    )

    for rx in range(N_RX):

        # Use this receiver's signal directly, without MVDR
        y_rx = remove_static(S[:, rx, r_check])

        phase_rx = np.unwrap(np.angle(y_rx))

        dphase_rx = detrend(
            np.diff(phase_rx) * FS_SLOW,
            type="linear"
        )

        x_rx = bandpass(dphase_rx, HR_BAND)

        
        # Check whether the near-160 bpm component persists over time
        window_len = int(round(30.0 * FS_SLOW))
        hop = int(round(15.0 * FS_SLOW))

        for start in range(
            0, len(x_rx) - window_len + 1, hop
        ):
            end = start + window_len
            x_win = x_rx[start:end]

            hr_win, _, f_win, P_win = psd_analysis(
                x_win, HR_BAND
            )

            valid_hr = (
                (f_win >= HR_BAND[0]) &
                (f_win <= HR_BAND[1])
            )

            target_hr = (
                (f_win >= 150.0 / 60.0) &
                (f_win <= 170.0 / 60.0)
            )

            if not np.any(valid_hr) or not np.any(target_hr):
                continue

            target_indices = np.where(target_hr)[0]
            peak_idx = target_indices[
                np.argmax(P_win[target_indices])
            ]

            near_hr = 60.0 * f_win[peak_idx]

            dominant_power = np.max(P_win[valid_hr])
            near_power = P_win[peak_idx]

            relative_db = 10.0 * np.log10(
                (near_power + 1e-30) /
                (dominant_power + 1e-30)
            )

            print(
                f"      RX{rx}, {start / FS_SLOW:.0f}-"
                f"{end / FS_SLOW:.0f} s: "
                f"dominant HR={hr_win:.1f}, "
                f"near-160 peak={near_hr:.1f} bpm, "
                f"relative power={relative_db:.2f} dB"
            )


        hr_rx, _, f_rx, P_rx = psd_analysis(
            x_rx, HR_BAND
        )

        # Get the strongest local peaks in the HR band
        peaks_rx = strongest_hr_peaks(f_rx, P_rx)

        peaks_text = ", ".join(
            f"{value:.2f}" for value in peaks_rx[:8]
        )

        print(
            f"  RX{rx}: HR={hr_rx:.2f} bpm | "
            f"strongest peaks=[{peaks_text}]"
        )

        # Compare the 150-170 bpm region with the strongest HR peak
        valid_hr = (
            (f_rx >= HR_BAND[0]) &
            (f_rx <= HR_BAND[1])
        )

        target_hr = (
            (f_rx >= 150.0 / 60.0) &
            (f_rx <= 170.0 / 60.0)
        )

        if (
            len(f_rx) > 0
            and len(P_rx) > 0
            and np.any(valid_hr)
            and np.any(target_hr)
        ):
            target_indices = np.where(target_hr)[0]

            peak_idx = target_indices[
                np.argmax(P_rx[target_indices])
            ]

            peak_near_160 = 60.0 * f_rx[peak_idx]

            strongest_power = np.max(P_rx[valid_hr])
            target_power = P_rx[peak_idx]

            relative_db = 10.0 * np.log10(
                (target_power + 1e-30) /
                (strongest_power + 1e-30)
            )

            print(
                f"      Peak in 150-170 bpm region: "
                f"{peak_near_160:.2f} bpm | "
                f"relative power: {relative_db:.2f} dB"
            )
        else:
            print(
                "      Could not calculate the 150-170 bpm "
                "peak for this receiver."
            )

# ============================================================
# 5. ESTIMATE HR AND RR FOR EACH CANDIDATE
# ============================================================

rows = []

print("\nCandidates (strongest moving energy first):")
print(
    f"{'#':>2} {'RangeBin':>8} {'Range(m)':>9} "
    f"{'Angle':>7} {'Motion(dB)':>11} "
    f"{'RR_PSD':>9} {'RR_Clear':>10} {'HR_PSD':>9}"
)

for candidate_number, (r, k) in enumerate(cells, start=1):

    # Use neighbouring range bins to estimate the MVDR covariance
    lo = max(int(r) - WIN, 0)
    hi = min(int(r) + WIN + 1, N_BINS_DATA)

    slab = S[:, :, lo:hi]

    X = slab.transpose(1, 0, 2).reshape(N_RX, -1)

    R_cov = (X @ X.conj().T) / X.shape[1]

    loading = (
        DIAG_LOADING
        * np.real(np.trace(R_cov))
        / N_RX
    )

    R_cov = R_cov + loading * np.eye(N_RX)

    try:
        Ria = np.linalg.solve(R_cov, steering[:, k])
    except np.linalg.LinAlgError:
        print(f"Skipping candidate at bin {r}: covariance solve failed.")
        continue

    denominator = steering[:, k].conj() @ Ria

    if abs(denominator) < 1e-20:
        print(f"Skipping candidate at bin {r}: invalid MVDR denominator.")
        continue

    weights = Ria / denominator

    # Beamform the signal from the candidate's centre range bin
    
    # Diagnose whether MVDR suppresses the heartbeat component.
    
    y = weights.conj() @ S[:, :, int(r)].T
    y = remove_static(y)

    # Match the preprocessing in 07_vital_signs.py
    phase = np.unwrap(np.angle(y))

    dphase = detrend(
        np.diff(phase) * FS_SLOW,
        type="linear"
    )

    # HR and RR spectral baselines use the same phase-difference signal
    x_heart = bandpass(dphase, HR_BAND)
    x_resp = bandpass(dphase, RR_BAND)

    hr, _, f_hr, P_hr = psd_analysis(x_heart, HR_BAND)
    rr, rr_clarity, _, _ = psd_analysis(x_resp, RR_BAND)

    hr_peaks = strongest_hr_peaks(f_hr, P_hr)

    hr_peaks_text = "; ".join(
        f"{value:.2f}" for value in hr_peaks
    )

    range_m = float(r) * RANGE_BIN_M
    angle_deg = float(angles[k])
    motion_db = float(P_search[r, k])

    rows.append([
        candidate_number,
        int(r),
        round(range_m, 3),
        round(angle_deg, 2),
        round(motion_db, 2),
        round(float(rr), 2) if np.isfinite(rr) else "",
        round(float(rr_clarity), 2) if np.isfinite(rr_clarity) else "",
        round(float(hr), 2) if np.isfinite(hr) else "",
        hr_peaks_text
    ])

    rr_text = f"{rr:.2f}" if np.isfinite(rr) else "NaN"
    cl_text = (
        f"{rr_clarity:.2f}"
        if np.isfinite(rr_clarity) else "NaN"
    )
    hr_text = f"{hr:.2f}" if np.isfinite(hr) else "NaN"

    print(
        f"{candidate_number:2d} {int(r):8d} {range_m:9.3f} "
        f"{angle_deg:7.1f} {motion_db:11.2f} "
        f"{rr_text:>9} {cl_text:>10} {hr_text:>9}"
    )

    print(
        "   Strongest HR-band peaks (bpm): "
        + (", ".join(f"{v:.2f}" for v in hr_peaks)
           if hr_peaks else "No local peaks found")
    )


# ============================================================
# 6. DISPLAY GROUND-TRUTH CONDITIONS FOR REFERENCE
# ============================================================

print("\nConditions encoded in the recording filename:")

print(
    ", ".join(
        f"{c}: HR={CONDITIONS[c][0]} bpm, "
        f"RR={np.mean(CONDITIONS[c][1]):.1f}/min"
        for c in CONDS
    )
)

print(
    "\nImportant: RR_PSD and HR_PSD are spectral estimates. "
    "RR_PSD is not necessarily the final peak-based or VMD estimate "
    "reported by 07_vital_signs.py."
)


# ============================================================
# 7. PLOT RANGE-ANGLE MAP
# ============================================================

plt.figure(figsize=(10, 6))

plt.pcolormesh(
    angles,
    np.arange(N_BINS_DATA),
    P_db,
    shading="auto",
    vmin=-40,
    vmax=0
)

plt.colorbar(label="Moving-echo power (dB)")

for row in rows:
    plt.plot(
        row[3],
        row[1],
        "wo",
        markersize=13,
        markerfacecolor="none",
        markeredgewidth=1.8
    )

    plt.text(
        row[3],
        row[1] + 3,
        str(row[0]),
        color="white",
        ha="center",
        fontsize=11
    )

plt.xlabel("Angle (degrees)")
plt.ylabel("Range bin")
plt.title("Candidate patient locations")
plt.xlim(-MAX_ANGLE_DEG - 5, MAX_ANGLE_DEG + 5)

plt.tight_layout()
plt.savefig("find_patients.png", dpi=300)

if SHOW_PLOTS:
    plt.show()

plt.close()


# ============================================================
# 8. SAVE RESULTS
# ============================================================

with open("find_patients.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "candidate",
        "range_bin",
        "range_m",
        "angle_deg",
        "motion_dB",
        "RR_PSD_per_min",
        "RR_clarity_dB",
        "HR_PSD_bpm",
        "HR_spectral_peaks_bpm"
    ])

    writer.writerows(rows)

print("\nSaved:")
print("  find_patients.png")
print("  find_patients.csv")
