"""

07_vital_signs.py  -  vital-sign extraction from the beamformed signal of each patient

Input : beamformed.npy   (frames, M)  complex, one column per patient     [script 06]

        targets.npy      (M, 2)       [range bin, angle bin]

Output: vital_results.csv, results_all.csv, vital_signs_target_N.png

THREE METHODS ARE COMPUTED FOR EVERY PATIENT AND PRINTED SIDE BY SIDE

---------------------------------------------------------------------

"vmd"       = Variational Mode Decomposition (vmd_tools.py), NEW:

                HR : the instantaneous-frequency signal (band 0.7-3.3 Hz) is split into K narrow modes;

                     the mode with the most energy that is also spectrally concentrated wins (modes

                     sitting on a harmonic of the breathing rate are penalised); repeated for several K,

                     the median is the answer

                RR : the phase is split into K modes; the lowest-frequency strong mode inside the

                     breathing band is the breathing mode; PSD, autocorrelation and peaks of THAT

                     mode vote on the final value

"paper"     = the method of the SiViS paper (page 6), the baseline you must replicate:

                phase -> unwrap -> instantaneous frequency (difference of the phase)

                  -> band-pass 0.8-3.0 Hz  -> power spectrum peak   = HR

                  -> band-pass 0.1-0.6 Hz  -> peak identification   = RR

"enhanced"  = your additions (kept from your version, cleaned up):

                HR : spectral peaks of 15 s windows, clustered, the most persistent one wins;

                     peaks on harmonics of the breathing rate are ignored (switchable)

                RR : respiration band-pass of the phase itself; PSD, autocorrelation and peaks

                     vote on the final value

METHOD below chooses which of the three goes into HR_est / RR_est (and into 08_evaluate.py).

All are saved (HR_paper, HR_enh, HR_vmd, RR_paper, RR_enh, RR_vmd), so you can report the paper

baseline AND your improvements from the same run.

"""

import os

import csv

import itertools

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert, welch, detrend, correlate

from config import *

from vmd_tools import vmd

# ============================================================

# SETTINGS

# ============================================================

METHOD = "vmd"                      # "vmd", "paper" or "enhanced": which one is written to HR_est / RR_est

# ---- VMD method ----

VMD_K_HR = (4, 5, 6)                # numbers of modes tried for the heart rate (median of the results)

VMD_K_RR = (3, 4)                   # ... and for the breathing rate

VMD_ALPHA_HR = 5000.0               # bandwidth penalty: larger = narrower modes (half-width \\~ FS_SLOW/sqrt(alpha) Hz)

VMD_ALPHA_RR = 3000.0

VMD_PRE_HR = (0.7, 3.3)             # Hz, loose band-pass before the decomposition (the paper band is 0.8-3.0)

VMD_PRE_RR = (0.06, 0.9)

VMD_CONC_HALFWIDTH_HZ = 0.12        # a heart mode is "concentrated" if its power sits within this of its peak

VMD_ALT_MIN_REL = 0.03              # a heart mode on a harmonic of the breathing rate is given up for a mode that is NOT

                                    # on a harmonic if that mode has at least this fraction of its score. (If nothing

                                    # else looks like a heartbeat the harmonic mode is kept: the heart may sit on it.)

                                    # 0.03 was chosen on simulated data (0.01 let noise modes win): check it on yours.

VMD_RR_ENERGY_FRAC = 0.40           # breathing mode = lowest-frequency mode with >= this fraction of the strongest

# HR-specific guard: respiration waveform harmonics can dominate the raw HR-band PSD.
# For VMD candidate selection, reject modes within 5 bpm of breathing harmonics 2x...8x.
# This is a signal-based heuristic, not a hard-coded target HR.
VMD_HARMONIC_MAX_ORDER = 8
VMD_HARMONIC_TOL_BPM = 5.0

HR_BAND = (0.8, 3.0)                # Hz   (paper)

RR_BAND = (0.1, 0.6)                # Hz   (paper)

HR_BPM = (HR_BAND[0] * 60, HR_BAND[1] * 60)

RR_BPM = (RR_BAND[0] * 60, RR_BAND[1] * 60)

FILTER_ORDER = 4                    # Butterworth, forwards and backwards (zero phase)

# Static offset of the beamformed signal. A static echo (bed frame, wall) in the patient's range bin

# adds a constant complex number to the signal; the phase then moves much less than the chest does and

# the heartbeat can nearly vanish. NOT in the paper.

#   "none"   : the paper's processing (angle of the raw beamformed signal)

#   "mean"   : subtract the complex mean (default). In my simulations the heart-rate error with a strong static

#              echo fell from 42 to 6 bpm; with no static echo it also helped (12 to 6).

#   "circfit": subtract the centre of the circle traced by the signal. Better in theory when the chest moves

#              less than one wavelength (the signal draws only a short arc); slightly worse in my simulations.

DC_REMOVAL = "mean"

# "waveform": peaks of the band-passed breathing signal (default, works for sinusoidal breathing)

# "envelope": literal reading of the paper (Hilbert envelope, then peaks); unreliable, for comparison

RR_MODE = "waveform"

# NOTE: counting peaks can double-count ripple when breathing is slow and not sinusoidal. In my

# simulations (RR = 6/min) the spectrum peak was right while the peak count was not. The paper

# method prints the spectrum value next to the peak value so you can see when they disagree.

# ---- enhanced method ----

HR_WINDOW_SEC = 15.0                # window length for the windowed HR

HR_STEP_SEC = 5.0

HR_TOP_CANDIDATES = 5               # spectral peaks kept per window

HR_CLUSTER_TOL = 8.0                # bpm: candidates closer than this belong to one cluster

# Breathing is not a pure sine: its harmonics (2x, 3x, 4x ... the breathing rate) fall inside the

# heart band and can be taller than the heartbeat. With this switch, spectral peaks within

# HR_HARMONIC_TOL bpm of a multiple of the (enhanced) breathing rate are ignored by the enhanced HR.

# CAUTION: if the real heart rate sits on a harmonic (e.g. HR 120 with RR 30), it is ignored too.

HR_MASK_HARMONICS = False

HR_HARMONIC_TOL = 1.5               # bpm

HR_HARMONIC_ORDERS = 4              # only harmonics 2 ... this order are masked (higher ones are weak, and

                                    # masking a dense comb would also delete the true heart rate)

RR_ACF_MIN_QUALITY = 0.50           # autocorrelation peak height needed to be trusted

RR_AGREE_TOL = 4.0                  # breaths/min: general agreement tolerance for PSD/ACF/peaks fusion

RR_PEAK_PSD_TOL = 3.0               # if waveform peak-count and its own PSD disagree more, prefer PSD

VMD_RR_PSD_TOL = 2.0                # maximum allowed difference between final VMD RR and input RR-band PSD

MIN_RR_PEAKS = 3

PROMINENCES = (0.30, 0.20, 0.15, 0.10, 0.07)     # peak prominence tried in turn (z-score units)

# "angle_order": target 1 (left) = first condition in the file name, 2 = second ...

# "best_match" : DEBUGGING ONLY, it looks at the truth. Never use it for reported results.

ASSIGN_MODE = "angle_order"

# ============================================================

# 1. LOAD

# ============================================================

B = np.load("beamformed.npy")

targets = np.load("targets.npy")

if B.ndim != 2:

    raise ValueError(f"beamformed.npy must be [frames, targets], got {B.shape}")

N_FRAMES, M = B.shape

print("Beamformed shape:", B.shape)

print(f"Slow-time rate: {FS_SLOW:.1f} Hz, recording length: {N_FRAMES / FS_SLOW:.1f} s")

print(f"METHOD written to HR_est / RR_est: {METHOD}      static offset removal: {DC_REMOVAL}")

if N_FRAMES / FS_SLOW < 20:

    print("NOTE: recording shorter than 20 s: heart-rate frequency resolution is coarse.")

# ============================================================

# 2. BASIC FUNCTIONS

# ============================================================

def fill_nonfinite_complex(y):

    """Replace NaN/Inf samples of a complex signal by interpolation."""

    y = np.asarray(y, dtype=np.complex128)

    good = np.isfinite(y.real) & np.isfinite(y.imag)

    if good.all():

        return y

    if good.sum() < 2:

        return np.nan_to_num(y)

    idx = np.arange(len(y))

    return np.interp(idx, idx[good], y.real[good]) + 1j * np.interp(idx, idx[good], y.imag[good])

def bandpass(x, band, fs, order=FILTER_ORDER):

    """Zero-phase Butterworth band-pass."""

    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")

    return sosfiltfilt(sos, np.asarray(x, dtype=float))

def remove_static(y, mode):

    """Subtract the static (constant) complex offset of the beamformed signal."""

    y = np.asarray(y, dtype=np.complex128)

    if mode == "none":

        return y

    mean = y.mean()

    if mode == "mean":

        return y - mean

    x, z = y.real, y.imag

    A = np.column_stack([2 * x, 2 * z, np.ones_like(x)])

    sol = np.linalg.lstsq(A, x ** 2 + z ** 2, rcond=None)[0]

    centre = sol[0] + 1j * sol[1]

    radius2 = sol[2] + sol[0] ** 2 + sol[1] ** 2

    spread = np.abs(y - mean).max()

    if not np.isfinite(radius2) or radius2 <= 0 or np.abs(centre - mean) > spread:   # unreliable fit

        return y - mean

    return y - centre

def zscore(x):

    x = np.asarray(x, dtype=float)

    x = x - x.mean()

    s = x.std()

    return x / s if s > 1e-12 else np.zeros_like(x)

def psd_peak(x, fs, band):

    """Power spectral density (Welch, Hann, zero-padded) and its tallest peak inside the band.

    Returns (peak frequency in Hz, frequency axis, PSD). The peak is refined by parabolic interpolation."""

    x = np.asarray(x, dtype=float)

    n = len(x)

    if n < 16:

        return np.nan, np.array([]), np.array([])

    nperseg = min(2048, n)

    nfft = max(8192, 2 ** int(np.ceil(np.log2(nperseg))))

    f, P = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,

                 nfft=nfft, detrend="linear", scaling="density")

    idx = np.where((f >= band[0]) & (f <= band[1]))[0]

    if len(idx) == 0:

        return np.nan, f, P

    k = idx[np.argmax(P[idx])]

    f_peak = f[k]

    if 0 < k < len(P) - 1:

        den = P[k - 1] - 2 * P[k] + P[k + 1]

        if abs(den) > 1e-20:

            f_peak += 0.5 * (P[k - 1] - P[k + 1]) / den * (f[1] - f[0])

    return float(f_peak), f, P

def peak_rate(x, fs, bpm_band, prominences=PROMINENCES):

    """Rate from waveform peaks ("peak identification"). Both polarities and several prominence

    values are tried; the best-scoring valid result wins.

    Returns dict: rate (bpm, NaN if none), peaks (sample indices), polarity."""

    z = zscore(x)

    # two peaks cannot be closer than 80% of the shortest physiological period

    distance = max(1, int(0.8 * 60.0 / bpm_band[1] * fs))

    for prom in prominences:

        best = None

        for polarity, sig in (("positive", z), ("negative", -z)):

            pk, props = find_peaks(sig, distance=distance, prominence=prom)

            if len(pk) < MIN_RR_PEAKS:

                continue

            gaps = np.diff(pk)

            rate = 60.0 * fs / np.median(gaps)

            if not (bpm_band[0] <= rate <= bpm_band[1]):

                continue

            cv = np.std(gaps) / (np.mean(gaps) + 1e-12)

            score = len(pk) + 2.0 * np.mean(props["prominences"]) - 3.0 * cv

            if best is None or score > best["score"]:

                best = {"rate": float(rate), "peaks": pk, "polarity": polarity, "score": float(score)}

        if best is not None:

            return best

    return {"rate": np.nan, "peaks": np.array([], dtype=int), "polarity": "none", "score": 0.0}

def acf_rate(x, fs, bpm_band):

    """Rate from the autocorrelation. Returns (rate bpm, quality = height of the peak, lags, acf)."""

    x = detrend(np.asarray(x, dtype=float), type="linear")

    if x.std() < 1e-12:

        return np.nan, 0.0, np.array([]), np.array([])

    ac = correlate(x, x, mode="full", method="fft")[len(x) - 1:]

    ac = ac / ac[0]

    lo = max(1, int(np.ceil(fs * 60.0 / bpm_band[1])))

    hi = min(len(ac) - 1, int(np.floor(fs * 60.0 / bpm_band[0])))

    if hi <= lo:

        return np.nan, 0.0, np.arange(len(ac)), ac

    seg = ac[lo:hi + 1]

    pk, _ = find_peaks(seg, prominence=0.01)

    k = pk[np.argmax(seg[pk])] if len(pk) else int(np.argmax(seg))

    return float(60.0 * fs / (lo + k)), float(seg[k]), np.arange(len(ac)), ac

# ============================================================

# 3. ENHANCED HEART RATE: windows + clustering + persistence

# ============================================================

def spectral_candidates(x, fs, band, top_n, rr_bpm=np.nan):

    """Strongest local spectral peaks in the band: list of dicts (bpm, power relative to the tallest).

    If rr_bpm is given and HR_MASK_HARMONICS is on, peaks on harmonics of rr_bpm are dropped."""

    _, f, P = psd_peak(x, fs, band)

    if len(f) == 0:

        return []

    m = (f >= band[0]) & (f <= band[1])

    fb, Pb = f[m], P[m]

    if len(Pb) == 0 or Pb.max() <= 0:

        return []

    pk, _ = find_peaks(Pb, prominence=0.01 * Pb.max())

    if len(pk) == 0:

        pk = np.array([int(np.argmax(Pb))])

    cands = [{"bpm": float(fb[k] * 60), "power": float(Pb[k] / Pb.max())} for k in pk]

    if HR_MASK_HARMONICS and np.isfinite(rr_bpm) and rr_bpm > 0:

        kept = [c for c in cands

                if min(abs(c["bpm"] - h * rr_bpm) for h in range(2, HR_HARMONIC_ORDERS + 1)) > HR_HARMONIC_TOL]

        cands = kept if kept else cands          # never remove everything

    cands.sort(key=lambda d: d["power"], reverse=True)

    return cands[:top_n]

def windowed_hr(x, fs, band, rr_bpm=np.nan):

    """HR from overlapping windows. Candidates of all windows are clustered by frequency and the

    cluster that persists in the most windows (with decent power) wins.

    CAUTION: a breathing harmonic is also persistent. Check the printed clusters.

    Returns (hr bpm, strongest candidate of every window, clusters sorted by score)."""

    win, step = int(HR_WINDOW_SEC * fs), int(HR_STEP_SEC * fs)

    if len(x) < win:                                  # recording shorter than one window: whole record

        c = spectral_candidates(x, fs, band, 1, rr_bpm)

        return (c[0]["bpm"] if c else np.nan), [], []

    members_all, per_window = [], []

    for w, start in enumerate(range(0, len(x) - win + 1, step), start=1):

        cands = spectral_candidates(x[start:start + win], fs, band, HR_TOP_CANDIDATES, rr_bpm)

        if cands:

            per_window.append(cands[0]["bpm"])

            members_all += [dict(c, window=w) for c in cands]

    if not members_all:

        return np.nan, per_window, []

    members_all.sort(key=lambda d: d["bpm"])

    clusters = []

    for c in members_all:

        for cl in clusters:

            centre = np.average([m["bpm"] for m in cl], weights=[max(m["power"], 0.01) for m in cl])

            if abs(c["bpm"] - centre) <= HR_CLUSTER_TOL:

                cl.append(c)

                break

        else:

            clusters.append([c])

    n_win = max(1, len(per_window))

    scored = []

    for cl in clusters:

        w8 = np.array([max(m["power"], 0.01) for m in cl])

        centre = float(np.average([m["bpm"] for m in cl], weights=w8))

        n_unique = len({m["window"] for m in cl})

        persistence = n_unique / n_win

        mean_power = float(np.mean([m["power"] for m in cl]))

        score = 4.0 * persistence + 1.5 * mean_power + 0.10 * min(n_unique, 10)

        scored.append({"center": centre, "persistence": persistence, "mean_power": mean_power, "score": score})

    scored.sort(key=lambda d: d["score"], reverse=True)

    return scored[0]["center"], per_window, scored

# ============================================================

# 4. ENHANCED RESPIRATION RATE: PSD, autocorrelation and peaks vote

# ============================================================

def fuse_rr(rr_psd, rr_acf, acf_quality, rr_peak, n_peaks):

    """Combine the three estimates. Usable: PSD always; ACF only if its quality is high enough;

    peaks only if enough breaths were found. If at least two usable estimates agree (within

    RR_AGREE_TOL) their mean is returned; otherwise the PSD value, then the others."""

    est = []

    if np.isfinite(rr_psd):

        est.append(("PSD", rr_psd))

    if np.isfinite(rr_acf) and acf_quality >= RR_ACF_MIN_QUALITY:

        est.append(("ACF", rr_acf))

    if np.isfinite(rr_peak) and n_peaks >= MIN_RR_PEAKS:

        est.append(("PEAK", rr_peak))

    if not est:

        return np.nan, "NONE"

    best = None

    for (n1, v1), (n2, v2) in itertools.combinations(est, 2):

        d = abs(v1 - v2)

        if d <= RR_AGREE_TOL and (best is None or d < best[0]):

            best = (d, f"{n1}+{n2}", 0.5 * (v1 + v2))

    if best is not None:

        return best[2], best[1]

    return est[0][1], est[0][0]

# ============================================================

# 4b. VMD: BREATHING MODE AND HEART MODE

# ============================================================

def vmd_breathing(phase_d, fs):

    """Estimate RR with VMD, but reject a VMD result that contradicts the input RR spectrum.

    A low-frequency VMD mode can represent a subharmonic or movement artifact. The input

    RR-band PSD is therefore a reference. If the median VMD estimate is farther than

    VMD_RR_PSD_TOL from that reference, return the input PSD estimate and flag the fallback.

    This does not use the programmed ground truth to choose a rate.

    """

    pre = bandpass(phase_d, VMD_PRE_RR, fs)

    rr_input = bandpass(phase_d, RR_BAND, fs)

    ref_f, _, _ = psd_peak(rr_input, fs, RR_BAND)

    ref_bpm = 60.0 * ref_f if np.isfinite(ref_f) else np.nan

    runs = []

    for K in VMD_K_RR:

        modes, cf = vmd(

            pre, fs, K, alpha=VMD_ALPHA_RR,

            f_init=np.linspace(RR_BAND[0], VMD_PRE_RR[1], K)

        )

        candidates = []

        for k in range(K):

            if not (RR_BAND[0] <= cf[k] <= RR_BAND[1]):

                continue

            f_pk, f, P = psd_peak(modes[k], fs, RR_BAND)

            if not np.isfinite(f_pk) or len(P) == 0:

                continue

            in_band = (f >= RR_BAND[0]) & (f <= RR_BAND[1])

            near = np.abs(f - f_pk) <= max(0.04, VMD_CONC_HALFWIDTH_HZ / 2)

            concentration = P[near & in_band].sum() / (P[in_band].sum() + 1e-30)

            rate = 60.0 * f_pk

            energy = float(np.var(modes[k]))

            agreement = (np.exp(-0.5 * ((rate - ref_bpm) / VMD_RR_PSD_TOL) ** 2)

                         if np.isfinite(ref_bpm) else 1.0)

            score = energy * concentration * (0.25 + 0.75 * agreement)

            candidates.append({"rate": rate, "mode": modes[k], "score": score,

                               "agreement": abs(rate - ref_bpm) if np.isfinite(ref_bpm) else np.nan})

        if not candidates:

            continue

        near_ref = [c for c in candidates if np.isfinite(c["agreement"]) and c["agreement"] <= VMD_RR_PSD_TOL]

        pool = near_ref if near_ref else candidates

        chosen = max(pool, key=lambda c: c["score"])

        runs.append((K, chosen["rate"], chosen["mode"], chosen["score"]))

    if not runs:

        return {"rr": ref_bpm, "mode": None, "runs": [], "reference": ref_bpm,

                "fallback": np.isfinite(ref_bpm)}

    vals = np.array([r[1] for r in runs])

    med = float(np.median(vals))

    pick = runs[int(np.argmin(np.abs(vals - med)))]

    fallback = bool(np.isfinite(ref_bpm) and abs(med - ref_bpm) > VMD_RR_PSD_TOL)

    rr_out = float(ref_bpm) if fallback else med

    return {"rr": rr_out, "mode": pick[2], "runs": [(r[0], r[1]) for r in runs],

            "reference": ref_bpm, "fallback": fallback, "vmd_median": med}

def vmd_heart(dphase, fs, rr_bpm=np.nan):

    """Estimate HR from VMD modes with a consistency check against the input HR-band PSD.

    The old implementation selected the highest energy/concentration mode independently for each K

    and took the median, even if the selected frequencies disagreed substantially. The revised

    version preferentially chooses a valid VMD mode near the input's dominant HR-band frequency.

    This cannot recover a heartbeat absent from the beamformed signal; it avoids switching to an

    unsupported VMD mode when a mode consistent with the input peak is available.

    """

    pre = bandpass(dphase, VMD_PRE_HR, fs)

    ref_f, _, _ = psd_peak(pre, fs, HR_BAND)

    ref_bpm = 60.0 * ref_f if np.isfinite(ref_f) else np.nan

    runs = []

    for K in VMD_K_HR:

        modes, cf = vmd(

            pre, fs, K, alpha=VMD_ALPHA_HR,

            f_init=np.linspace(VMD_PRE_HR[0], VMD_PRE_HR[1], K)

        )

        candidates = []

        for k in range(K):

            if not (HR_BAND[0] <= cf[k] <= HR_BAND[1]):

                continue

            f_pk, f, P = psd_peak(modes[k], fs, HR_BAND)

            if not np.isfinite(f_pk) or len(P) == 0:

                continue

            in_band = (f >= HR_BAND[0]) & (f <= HR_BAND[1])

            near = np.abs(f - f_pk) <= VMD_CONC_HALFWIDTH_HZ

            concentration = P[near & in_band].sum() / (P[in_band].sum() + 1e-30)

            bpm = 60.0 * f_pk

            score = float(np.var(modes[k]) * concentration)

            on_harmonic = (np.isfinite(rr_bpm) and rr_bpm > 0 and
                           min(abs(bpm - h * rr_bpm)
                               for h in range(2, VMD_HARMONIC_MAX_ORDER + 1)) <= VMD_HARMONIC_TOL_BPM)

            # Retain harmonic penalty, but include a soft consistency preference for the input PSD.

            ref_delta = abs(bpm - ref_bpm) if np.isfinite(ref_bpm) else np.nan

            agreement = (np.exp(-0.5 * (ref_delta / 10.0) ** 2)

                         if np.isfinite(ref_delta) else 1.0)

            score *= (0.25 + 0.75 * agreement)

            candidates.append({"K": K, "bpm": bpm, "score": score, "mode": modes[k],

                               "on_harmonic": on_harmonic, "ref_delta": ref_delta})

        if not candidates:

            continue

        # TEMPORARY DIAGNOSTIC: inspect every valid VMD heart-mode candidate.
        # This reports the decomposition itself; it does not use ground truth.
        print(f"    [VMD DEBUG] K={K}, input HR PSD={ref_bpm:.2f} bpm")
        for c in sorted(candidates, key=lambda item: item["score"], reverse=True):
            ref_delta_text = (
                f"{c['ref_delta']:.2f}"
                if np.isfinite(c["ref_delta"])
                else "N/A"
            )
            print(
                f"        Mode HR={c['bpm']:.2f} bpm | "
                f"score={c['score']:.5g} | "
                f"breathing harmonic={c['on_harmonic']} | "
                f"distance from input peak={ref_delta_text} bpm"
            )

        non_harmonic = [

        c for c in candidates

        if not c["on_harmonic"]

        ]

        pool = non_harmonic if non_harmonic else candidates

        # Do not force the selected VMD mode to remain within 10 bpm

        # of the dominant input PSD peak. Retain the existing soft

        # consistency score, but allow other valid modes to compete.

        best = max(pool, key=lambda c: c["score"])

        runs.append((K, best["bpm"], best["score"], best["mode"]))

    if not runs:

        return {"hr": np.nan, "mode": None, "runs": [], "reference": ref_bpm}

    vals = np.array([r[1] for r in runs])

    med = float(np.median(vals))

    pick = runs[int(np.argmin(np.abs(vals - med)))]

    return {"hr": med, "mode": pick[3], "runs": [(r[0], r[1], r[2]) for r in runs],

            "reference": ref_bpm}

# 5. PROCESS EVERY PATIENT

# ============================================================

hr_paper = np.full(M, np.nan)

hr_enh = np.full(M, np.nan)

hr_vmd = np.full(M, np.nan)

rr_paper = np.full(M, np.nan)

rr_enh = np.full(M, np.nan)

rr_vmd = np.full(M, np.nan)

for m in range(M):

    print("\n" + "=" * 65)

    print(f"TARGET {m + 1}")

    print("=" * 65)

    # ---- complex signal -> phase -> instantaneous frequency ----

    y = remove_static(fill_nonfinite_complex(B[:, m]), DC_REMOVAL)

    phase = np.unwrap(np.angle(y))

    phase_d = detrend(phase, type="linear")                      # chest motion (rad)

    dphase = detrend(np.diff(phase) * FS_SLOW, type="linear")    # instantaneous frequency (rad/s)

    # ---- the two band-pass filters ----

    x_heart = bandpass(dphase, HR_BAND, FS_SLOW)                 # HR: from the paper

    x_resp_paper = bandpass(dphase, RR_BAND, FS_SLOW)            # RR (paper): same dphase signal

    x_resp_enh = bandpass(dphase, RR_BAND, FS_SLOW)             # RR enhancement uses paper phase-difference signal

    # ================= PAPER METHOD =================

    f_hr, hr_f, hr_P = psd_peak(x_heart, FS_SLOW, HR_BAND)

    hr_paper[m] = 60 * f_hr if np.isfinite(f_hr) else np.nan

    sig_paper = np.abs(hilbert(x_resp_paper)) if RR_MODE == "envelope" else x_resp_paper

    info_p = peak_rate(sig_paper, FS_SLOW, RR_BPM)

    f_chk, _, _ = psd_peak(sig_paper, FS_SLOW, RR_BAND)           # spectrum of the same signal (cross-check)

    rr_paper_psd = 60 * f_chk if np.isfinite(f_chk) else np.nan

    rr_paper_peak = info_p["rate"]

    if (np.isfinite(rr_paper_peak) and np.isfinite(rr_paper_psd)

            and abs(rr_paper_peak - rr_paper_psd) > RR_PEAK_PSD_TOL):

        # A peak-count estimate can double-count ripple; use the same signal's spectral estimate

        # when the disagreement is too large.

        rr_paper[m] = rr_paper_psd

        rr_paper_source = "PSD fallback (peak/PSD disagreement)"

    elif np.isfinite(rr_paper_peak):

        rr_paper[m] = rr_paper_peak

        rr_paper_source = "waveform peaks"

    else:

        rr_paper[m] = rr_paper_psd

        rr_paper_source = "PSD fallback (no valid peaks)"

    # ================= ENHANCED METHOD =================

    sig_enh = np.abs(hilbert(x_resp_enh)) if RR_MODE == "envelope" else x_resp_enh

    f_rr, rr_f, rr_P = psd_peak(sig_enh, FS_SLOW, RR_BAND)

    rr_psd = 60 * f_rr if np.isfinite(f_rr) else np.nan

    rr_acf, acf_q, rr_lags, rr_ac = acf_rate(sig_enh, FS_SLOW, RR_BPM)

    info_e = peak_rate(sig_enh, FS_SLOW, RR_BPM)

    rr_enh[m], rr_how = fuse_rr(rr_psd, rr_acf, acf_q, info_e["rate"], len(info_e["peaks"]))

    # heart rate (after the breathing rate, because the harmonic mask needs it)

    hr_enh[m], hr_win, hr_clusters = windowed_hr(x_heart, FS_SLOW, HR_BAND, rr_enh[m])

    # ================= VMD METHOD =================

    vb = vmd_breathing(dphase, FS_SLOW)

    rr_mode = vb["mode"]

    if rr_mode is not None:

        v_acf, v_acf_q, _, _ = acf_rate(rr_mode, FS_SLOW, RR_BPM)

        v_pk = peak_rate(rr_mode, FS_SLOW, RR_BPM)

        if vb.get("fallback", False):

            # The VMD modes did not preserve the input's dominant RR peak. Use the input PSD

            # estimate rather than letting inconsistent mode peaks override it.

            rr_vmd[m] = vb["rr"]

            rr_vmd_how = "input PSD fallback (VMD mismatch)"

            print(f"    VMD RR mismatch: mode median {vb.get('vmd_median', np.nan):.2f}, "

                  f"input RR PSD {vb['reference']:.2f}; using input PSD.")

        else:

            rr_vmd[m], rr_vmd_how = fuse_rr(

                vb["rr"], v_acf, v_acf_q, v_pk["rate"], len(v_pk["peaks"])

            )

            # Keep the FINAL fused VMD estimate consistent with the dominant RR-band PSD.

            # This is a signal-quality guard, not ground-truth-based selection.

            vmd_reference = vb.get("reference", np.nan)

            if (np.isfinite(vmd_reference) and np.isfinite(rr_vmd[m])

                    and abs(rr_vmd[m] - vmd_reference) > VMD_RR_PSD_TOL):

                print(

                    f"    Fused VMD RR {rr_vmd[m]:.2f} differs from input PSD "

                    f"{vmd_reference:.2f} by more than {VMD_RR_PSD_TOL:.1f}; "

                    "using input PSD estimate."

                )

                rr_vmd[m] = float(vmd_reference)

                rr_vmd_how = "input PSD fallback (fused RR mismatch)"

    else:

        v_acf, v_acf_q, v_pk = np.nan, 0.0, {"rate": np.nan, "peaks": []}

        rr_vmd[m] = vb["rr"] if np.isfinite(vb.get("rr", np.nan)) else rr_paper[m]

        rr_vmd_how = "input PSD fallback (no VMD mode)"

    vh = vmd_heart(dphase, FS_SLOW, rr_vmd[m])

    hr_vmd[m] = vh["hr"] if np.isfinite(vh["hr"]) else hr_paper[m]

    # ---- print ----

    print(f"\nHR  VMD                            : {hr_vmd[m]:7.2f} bpm")

    for K, bpm, sc in vh["runs"]:

        print(f"    K = {K}: heart mode {bpm:7.2f} bpm  (score {sc:.3g})")

    print(f"RR  VMD [{rr_vmd_how}]".ljust(36) + f": {rr_vmd[m]:7.2f} breaths/min")

    print("    breathing mode per K           : " + ", ".join(f"K={K}: {v:.2f}" for K, v in vb["runs"])

          + f"   (ACF {v_acf:.2f}, peaks {v_pk['rate']:.2f})")

    print(f"\nHR  paper    (PSD peak)            : {hr_paper[m]:7.2f} bpm")

    print(f"HR  enhanced (windowed clusters)   : {hr_enh[m]:7.2f} bpm")

    if hr_win:

        print("    strongest peak per window      : " + ", ".join(f"{v:.0f}" for v in hr_win))

    for i, c in enumerate(hr_clusters[:5], start=1):

        print(f"    cluster {i}: {c['center']:7.2f} bpm  persistence {c['persistence']:.2f}  "

              f"power {c['mean_power']:.2f}  score {c['score']:.2f}")

    cand = spectral_candidates(x_heart, FS_SLOW, HR_BAND, 5)             # unmasked, for inspection

    print("    top peaks, whole recording     : " + ", ".join(f"{c['bpm']:.1f} ({c['power']:.2f})" for c in cand))

    print(f"\nRR  paper [{rr_paper_source}] ({info_p['polarity']} peaks) : {rr_paper[m]:7.2f} breaths/min"

          f"   (spectrum of the same signal: {rr_paper_psd:.2f})")

    if np.isfinite(rr_paper[m]) and np.isfinite(rr_paper_psd) and abs(rr_paper[m] - rr_paper_psd) > RR_AGREE_TOL:

        print("    WARNING: peak count and spectrum disagree (ripple from breathing harmonics is a common cause).")

    print(f"RR  enhanced [{rr_how}]".ljust(36) + f": {rr_enh[m]:7.2f} breaths/min")

    print(f"    PSD {rr_psd:.2f}   ACF {rr_acf:.2f} (quality {acf_q:.2f})   "

          f"peaks {info_e['rate']:.2f} ({len(info_e['peaks'])} found)")

    # Transparent diagnostic: large disagreement means HR is ambiguous in this recording.

    hr_candidates_now = np.array([hr_paper[m], hr_enh[m], hr_vmd[m]], dtype=float)

    hr_candidates_now = hr_candidates_now[np.isfinite(hr_candidates_now)]

    if len(hr_candidates_now) >= 2 and (np.max(hr_candidates_now) - np.min(hr_candidates_now)) > 20.0:

        print("WARNING: HR methods disagree by >20 bpm; treat HR as low-confidence and inspect the heart-band spectrum.")

    # ---- plots ----

    t = np.arange(len(sig_paper)) / FS_SLOW

    n_show = min(len(t), int(30 * FS_SLOW))

    pk = info_p["peaks"][info_p["peaks"] < n_show]

    fig, ax = plt.subplots(5, 1, figsize=(11, 15))

    ax[0].plot(np.arange(len(phase_d))[:n_show] / FS_SLOW, phase_d[:n_show])

    ax[0].set_title(f"Target {m + 1}: detrended unwrapped phase (chest motion)")

    ax[0].set_ylabel("rad")

    ax[1].plot(t[:n_show], sig_paper[:n_show])

    ax[1].plot(t[pk], sig_paper[pk], "rv")

    ax[1].set_title(f"Breathing band {RR_BAND[0]}-{RR_BAND[1]} Hz (paper signal)  ->  RR paper = {rr_paper[m]:.1f}/min")

    ax[1].set_xlabel("Time (s)")

    if len(rr_P):

        ax[2].plot(rr_f * 60, rr_P / rr_P.max())

    for val, lab, ls in ((rr_psd, "PSD", "--"), (rr_acf, "ACF", ":"), (info_e["rate"], "peaks", "-."),

                         (rr_enh[m], "enhanced", "-")):

        if np.isfinite(val):

            ax[2].axvline(val, linestyle=ls, label=f"{lab} {val:.1f}")

    if rr_mode is not None:

        _, f_rm, P_rm = psd_peak(rr_mode, FS_SLOW, RR_BAND)

        if len(P_rm):

            ax[2].plot(f_rm * 60, P_rm / P_rm.max(), color="m", alpha=0.7, label="VMD breathing mode")

    if np.isfinite(rr_vmd[m]):

        ax[2].axvline(rr_vmd[m], linestyle="-", color="m", label=f"VMD {rr_vmd[m]:.1f}")

    ax[2].set_xlim(*RR_BPM)

    ax[2].set_title("Breathing spectrum (enhanced signal and VMD mode)")

    ax[2].set_xlabel("breaths/min")

    ax[2].legend()

    if len(rr_ac):

        lag_s = rr_lags[1:] / FS_SLOW

        bpm = 60.0 / lag_s

        ok = (bpm >= RR_BPM[0]) & (bpm <= RR_BPM[1])

        ax[3].plot(bpm[ok], rr_ac[1:][ok])

    if np.isfinite(rr_acf):

        ax[3].axvline(rr_acf, linestyle="--")

    ax[3].set_xlim(*RR_BPM)

    ax[3].set_title(f"Breathing autocorrelation -> {rr_acf:.1f}/min (quality {acf_q:.2f})")

    ax[3].set_xlabel("breaths/min")

    if len(hr_P):

        ax[4].plot(hr_f * 60, hr_P / hr_P.max())

    if np.isfinite(hr_paper[m]):

        ax[4].axvline(hr_paper[m], linestyle="--", color="r", label=f"paper {hr_paper[m]:.1f}")

    if np.isfinite(hr_enh[m]):

        ax[4].axvline(hr_enh[m], linestyle="-.", color="g", label=f"enhanced {hr_enh[m]:.1f}")

    if vh["mode"] is not None:

        _, f_vm, P_vm = psd_peak(vh["mode"], FS_SLOW, HR_BAND)

        if len(P_vm):

            ax[4].plot(f_vm * 60, P_vm / P_vm.max(), color="m", alpha=0.7, label="VMD heart mode")

    if np.isfinite(hr_vmd[m]):

        ax[4].axvline(hr_vmd[m], linestyle="-", color="m", label=f"VMD {hr_vmd[m]:.1f}")

    ax[4].set_xlim(HR_BPM[0] - 5, HR_BPM[1] + 5)

    ax[4].set_title("Heart-band spectrum")

    ax[4].set_xlabel("beats/min")

    ax[4].legend()

    for a in ax:

        a.grid(True)

    plt.tight_layout()

    plt.savefig(f"vital_signs_target_{m + 1}.png", dpi=200, bbox_inches="tight")

    if SHOW_PLOTS:

        plt.show()

    plt.close(fig)

# ============================================================

# 6. GROUND TRUTH AND PAIRING WITH THE CONDITIONS IN THE FILE NAME

# ============================================================

def truth_of(cond):

    hr, rr = CONDITIONS[cond]

    return float(hr), float(np.mean(rr))          # condition L has 3 RR values: mean only

hr_sel = {"paper": hr_paper, "enhanced": hr_enh, "vmd": hr_vmd}[METHOD]

rr_sel = {"paper": rr_paper, "enhanced": rr_enh, "vmd": rr_vmd}[METHOD]

conds = CONDS[:M]

if len(conds) < M:

    print("\nWARNING: fewer conditions in the file name than detected targets.")

assign = list(range(len(conds)))

if ASSIGN_MODE == "best_match" and len(conds) == M:

    best = None

    for perm in itertools.permutations(range(M)):

        err = 0.0

        for i, p in enumerate(perm):

            ht, rt = truth_of(conds[p])

            err += abs(hr_sel[i] - ht) / 50 + abs(np.nan_to_num(rr_sel[i], nan=50) - rt) / 10

        if best is None or err < best[0]:

            best = (err, perm)

    assign = list(best[1])

    print("\nWARNING: ASSIGN_MODE = best_match uses the truth: DEBUGGING ONLY.")

# ---- PAIRING CHECK (diagnosis only; uses the truth, never used for the reported results) ----

if len(conds) == M and 1 < M <= 4:

    errs = []

    for perm in itertools.permutations(range(M)):

        e = 0.0

        for i, p in enumerate(perm):

            ht, rt = truth_of(conds[p])

            e += abs(hr_sel[i] - ht) / 50 + abs(np.nan_to_num(rr_sel[i], nan=50) - rt) / 10

        errs.append((e, perm))

    errs.sort()

    cur = tuple(assign)

    print("\nPAIRING CHECK (uses the truth; diagnosis only, not used for scoring)")

    for e, perm in sorted(errs, key=lambda x: (x[1] != cur, x[0]))[:3]:

        label = "  <- current pairing" if perm == cur else ""

        print("   " + ", ".join(f"target {i + 1}->{conds[p]}" for i, p in enumerate(perm)) + f"   error {e:.2f}{label}")

    if errs[0][1] != cur and errs[0][0] < 0.5 * dict((p, e) for e, p in errs)[cur]:

        print("   NOTE: another pairing fits much better. The detected order may not match SimMan A, B (see TARGET_ORDER in config.py).")

    elif errs[0][0] > 1.0:

        print("   NOTE: no pairing fits well: the estimates themselves are off (wrong target position or leakage), not the order.")

def rnd(v, k=2):

    return round(float(v), k) if np.isfinite(v) else np.nan

rows = []

for i in range(M):

    cond = conds[assign[i]] if i < len(conds) else ""

    hr_t, rr_t = truth_of(cond) if cond else (np.nan, np.nan)

    sin_t = np.clip(2.0 * (targets[i, 1] - N_THETA // 2) / N_THETA, -1.0, 1.0)

    rows.append({

        "recording": REC_NAME, "target": i + 1, "angle_deg": round(float(np.degrees(np.arcsin(sin_t))), 1),

        "cond": cond,

        "HR_est": rnd(hr_sel[i]), "HR_true": hr_t, "RR_est": rnd(rr_sel[i]), "RR_true": rr_t,

        "HR_paper": rnd(hr_paper[i]), "HR_enh": rnd(hr_enh[i]), "HR_vmd": rnd(hr_vmd[i]), "RR_vmd": rnd(rr_vmd[i]),

        "RR_paper": rnd(rr_paper[i]), "RR_enh": rnd(rr_enh[i]),

    })

header = ["recording", "target", "angle_deg", "cond", "HR_est", "HR_true", "RR_est", "RR_true",

          "HR_paper", "HR_enh", "HR_vmd", "RR_paper", "RR_enh", "RR_vmd"]

# ============================================================

# 7. PRINT AND SAVE

# ============================================================

print("\n" + "=" * 65)

print(f"RESULTS OF THIS RECORDING   (HR_est / RR_est = {METHOD} method)")

print("=" * 65)

print("  ".join(f"{h:>9}" for h in header[1:]))

for r in rows:

    print("  ".join(f"{str(r[h]):>9}" for h in header[1:]))

with open("vital_results.csv", "w", newline="") as f:

    w = csv.DictWriter(f, fieldnames=header)

    w.writeheader()

    w.writerows(rows)

# results_all.csv: keep the rows of other recordings, replace the rows of this one.

# (old files with fewer columns are upgraded; missing values stay empty)

old = []

if os.path.exists("results_all.csv"):

    with open("results_all.csv", newline="") as f:

        old = [r for r in csv.DictReader(f) if r.get("recording") != REC_NAME]

with open("results_all.csv", "w", newline="") as f:

    w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", restval="")

    w.writeheader()

    w.writerows(old)

    w.writerows(rows)

print("\nSaved:")

print("  vital_results.csv")

print("  results_all.csv  (this recording added or replaced)")
