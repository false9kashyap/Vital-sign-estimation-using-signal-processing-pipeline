"""
config.py  -  SHARED SETTINGS FOR ALL SiViS SCRIPTS (edit only here)

Every numbered script (01_..., 02_..., ...) imports this file, so a setting such as the
ADC rate is written in ONE place only.
"""

import os
import re
import numpy as np


# ============================================================
# 1. RECORDING
# ============================================================

CSV_FILE = os.environ.get(
    "SIVIS_CSV",
    r"108_TopFront_2m_45deg_Man2_CondI_CondE_ADC256_Chirp128_SR1000_RE60_FR40_Gain30_FS30_IWR_B.csv"
)

REC_NAME = os.path.splitext(os.path.basename(CSV_FILE))[0]

# Show plot windows? (run_all.py switches this off)
SHOW_PLOTS = os.environ.get("SIVIS_NO_SHOW") is None


# ============================================================
# 2. RADAR CUBE SHAPE
# ============================================================

def parse_name(name):
    """Read radar settings, number of mannequins and conditions from the file name."""
    base = os.path.splitext(os.path.basename(name))[0]

    m = re.search(r"(?:SimMan|Man)(\d+)", base)
    n_man = int(m.group(1)) if m else 2

    conds = re.findall(r"Cond([A-L])", base)

    m = re.search(r"(?:FR|FP)(\d+)", base)
    frame_ms = float(m.group(1)) if m else 40.0

    m = re.search(r"Chirp(\d+)", base)
    chirps = int(m.group(1)) if m else None

    m = re.search(r"ADC(\d+)", base)
    adc = int(m.group(1)) if m else None

    m = re.search(r"_SR(\d+)", base)
    sr_hz = float(m.group(1)) * 1e3 if m else None          # SR is in kS/s (paper, Table 1)

    m = re.search(r"_FS(\d+)", base)
    slope = float(m.group(1)) * 1e12 if m else None         # FS is in MHz/us

    return {"n_man": n_man, "conds": conds, "frame_ms": frame_ms,
            "chirps": chirps, "adc": adc, "sr_hz": sr_hz, "slope": slope}


_meta = parse_name(CSV_FILE)

N_RX = 4

# Chirps per frame and ADC samples come from the file name (Chirp128, ADC256);
# the environment variable overrides are for testing.
N_CHIRPS = int(os.environ.get("SIVIS_N_CHIRPS", _meta["chirps"] or 128))
N_ADC = int(os.environ.get("SIVIS_N_ADC", _meta["adc"] or 256))

# TDM-MIMO: number of transmit antennas whose chirps are interleaved inside a frame.
# 1 = all chirps of a frame come from the same TX (your CSV has 128 chirps per frame,
# which is not divisible by 3, so 1 is the most likely value).
# Script 03 prints a check of this; change it only if that check says 2 or 3.
N_TX = int(os.environ.get("SIVIS_N_TX", 1))
TX_INDEX = 0                                               # which TX to use (0 = first)


# ============================================================
# 3. RADAR NUMBERS  (from the file name)
# ============================================================

# SR1000 -> 1000 kS/s = 1 MHz        (taken from the file name; default 1 MHz)
FS = _meta["sr_hz"] or 1e6                  # ADC sampling rate, Hz

# FS30 -> frequency slope = 30 MHz/us  (taken from the file name; default 30)
SLOPE = _meta["slope"] or 30e12             # Hz/s

C = 299792458                   # speed of light, m/s

CENTER_FREQ = 78e9              # Hz (77-81 GHz band)
WAVELENGTH = C / CENTER_FREQ    # about 3.8 mm

# Distance of ONE range bin (metres per FFT bin).
# For the complex range FFT: bin spacing = c * FS / (2 * SLOPE * N_ADC).
# NOTE: with FS = 1 MHz and 30 MHz/us the ADC window is 256 us long, i.e. a 7.68 GHz sweep,
# which is more than the chip can sweep (about 4 GHz). So the file-name numbers cannot all
# be literal. Your first chirp puts the strongest echo at 2.13 m with these numbers (the file
# name says 2 m), so they work as a scale, but treat distances in metres as APPROXIMATE
# and confirm with the motion-energy plot made by script 03.
RANGE_BIN_M = C * FS / (2 * SLOPE * N_ADC)

N_BINS = N_ADC // 2             # keep the first half of the range FFT (positive beat frequencies)


# ============================================================
# 4. FRAME RATE AND PATIENTS  (parsed from the file name)
# ============================================================

M_TARGETS = _meta["n_man"]                   # number of mannequins = patients to find
CONDS = _meta["conds"]                       # e.g. ['I', 'E']
FRAME_PERIOD = _meta["frame_ms"] * 1e-3      # seconds between frames
FS_SLOW = 1.0 / FRAME_PERIOD                 # slow-time sampling rate (25 Hz for 40 ms)


# ============================================================
# 5. ANTENNA GEOMETRY  (used for the angle processing)
# ============================================================

N_THETA = 128                                # number of angle bins (paper)

# Positions of the 4 RX antennas along the azimuth axis, in WAVELENGTHS.
# Default: 4 antennas in a line, half a wavelength apart.
# The IWR6843AOP antennas are arranged in 2-D: CHECK the TI datasheet and edit this list
# (for example, use only antennas that lie on one horizontal line).
RX_POS = np.arange(N_RX) * 0.5


# ============================================================
# 6. GROUND TRUTH (paper, scenarios A-L):  HR (bpm), RR (breaths/min)
# ============================================================

CONDITIONS = {
    "A": (80, 14),   "B": (60, 12),   "C": (110, 20),  "D": (80, 0),
    "E": (90, 22),   "F": (130, 33),  "G": (120, 30),  "H": (50, 6),
    "I": (160, 16),  "J": (104, 18),  "K": (125, 30),  "L": (110, (10, 6, 14)),
}


# ============================================================
# 7. OPTIONS
# ============================================================

# Remove static echoes (time average) before the range-angle map. NOT in the paper, but ON by default.
# Why: bed frames, walls or the ceiling right in front of the radar give a strong STATIC echo at
# boresight (0 degrees). On the plain map it outshines a patient who is off-axis, so the detector
# picks "0 degrees" instead of the patient, the beam points at the wrong place, and HR/RR are wrong.
# Static echoes are removed by subtracting the time average; chests, which move, remain.
# Set the environment variable SIVIS_REMOVE_STATIC=0 (or change "1" below) to get the paper's plain map.
REMOVE_STATIC_FOR_MAP = os.environ.get("SIVIS_REMOVE_STATIC", "1") == "1"


# In which order are the detected patients listed (and therefore paired with the conditions of the
# file name: first condition = SimMan A, second = SimMan B)?
#   "angle": left to right by angle (patients side by side)
#   "range": near to far by range   (patients one behind the other, seen from the front)
# The file name only tells which condition belongs to SimMan A and which to SimMan B (paper, Table 1);
# which mannequin is the left / nearer one depends on the set-up (paper, Fig. 1). Check it with
# 09_find_patients.py, which lists the breathing and heart rate found at every candidate position.
TARGET_ORDER = os.environ.get("SIVIS_TARGET_ORDER", "angle")


# ============================================================
# 8. MUSIC (super-resolution angle estimation)   -- used by 04 and 09
# ============================================================

# With only 4 RX antennas the ordinary (Bartlett / FFT) beam is about 25 degrees wide, so two
# mannequins a few degrees apart merge into ONE blob and the second target is then placed by the
# separation rule, not by the data. MUSIC splits the covariance matrix of the MOVING echoes into a
# signal and a noise subspace and finds angles from the noise subspace: it separates sources that
# are much closer than one beamwidth.
#
# The detection map is   Bartlett power  x  (normalised MUSIC pseudo-spectrum) ** MUSIC_EXPONENT
# (Bartlett gives a calibrated power per cell, MUSIC gives the sharp angle position.)
USE_MUSIC = os.environ.get("SIVIS_USE_MUSIC", "1") == "1"

MUSIC_WIN = 1               # covariance of range bin r is averaged over bins r-WIN ... r+WIN
MUSIC_EIG_RATIO = 4.0       # an eigenvalue counts as a source if it is > this x the smallest eigenvalue
MUSIC_EXPONENT = 1.0        # >1 = sharper peaks, <1 = softer
MUSIC_FB = True             # forward-backward averaging (only used if RX_POS is a uniform line)
