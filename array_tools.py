"""
array_tools.py  -  angle processing shared by 04_range_angle_map.py and 09_find_patients.py

    steering(sin_t)         steering vectors of the RX array
    compute_maps(S)         Bartlett map, MUSIC pseudo-spectrum, and the combined detection map

All maps are (N_BINS, N_THETA); angle bin k <-> sin(theta) = 2 (k - N_THETA/2) / N_THETA.
"""

import numpy as np
from config import *


def sin_axis():
    """sin(theta) of every angle bin (straight ahead = bin N_THETA/2)."""
    return 2.0 * (np.arange(N_THETA) - N_THETA // 2) / N_THETA


def steering(sin_t):
    """a[n, k] = exp(j 2 pi x_n sin_t[k]),  x_n = RX_POS (wavelengths).  Returns [N_RX, len(sin_t)]."""
    sin_t = np.atleast_1d(np.asarray(sin_t, dtype=float))
    return np.exp(1j * 2 * np.pi * RX_POS[:, None] * sin_t[None, :])


def uniform_line():
    d = np.diff(RX_POS)
    return bool(np.allclose(d, d[0]))


def covariances(S, chunk=200):
    """Covariance matrix of the RX antennas for every range bin.  S: [frames, RX, bins] -> [bins, RX, RX]."""
    n_frames, _, n_bins = S.shape
    R = np.zeros((n_bins, N_RX, N_RX), dtype=np.complex128)
    for start in range(0, n_frames, chunk):
        s = S[start:start + chunk].astype(np.complex128)
        R += np.einsum("fnb,fmb->bnm", s, s.conj())
    return R / n_frames


def slab_average(R, win):
    """Average the covariance of range bin r over the bins r-win ... r+win."""
    if win <= 0:
        return R
    Rp = np.pad(R, ((win, win), (0, 0), (0, 0)), mode="edge")
    n = R.shape[0]
    return sum(Rp[win + d: win + d + n] for d in range(-win, win + 1)) / (2 * win + 1)


def forward_backward(R):
    """Forward-backward averaging (uniform linear array): decorrelates coherent arrivals."""
    return 0.5 * (R + R[:, ::-1, ::-1].conj())


def music_pseudospectrum(R, a, max_src, ratio):
    """MUSIC pseudo-spectrum for every range bin.

    R : [bins, RX, RX]   a : [RX, angles]
    The number of sources in every bin is estimated from the eigenvalues (those larger than
    `ratio` x the smallest one), limited to 1 ... max_src.
    Returns P [bins, angles] normalised to a maximum of 1 in every bin, and n_src [bins].
    """
    trace = np.trace(R, axis1=1, axis2=2).real
    Rl = R + (1e-9 * trace + 1e-30)[:, None, None] * np.eye(N_RX)

    w, v = np.linalg.eigh(Rl)                                   # eigenvalues ascending

    n_src = (w > ratio * np.maximum(w[:, :1], 1e-30)).sum(axis=1)
    n_src = np.clip(n_src, 1, max_src)

    proj = np.einsum("nt,bni->bit", a.conj(), v)                # a^H v_i   [bins, RX, angles]

    noise = np.arange(N_RX)[None, :] < (N_RX - n_src)[:, None]  # which eigenvectors span the noise subspace

    den = (np.abs(proj) ** 2 * noise[:, :, None]).sum(axis=1)   # a^H En En^H a

    P = 1.0 / (den + 1e-3)

    P /= P.max(axis=1, keepdims=True)

    return P, n_src


def compute_maps(S):
    """S: [frames, RX, bins] slow-time data (static echoes NOT yet removed).

    Returns  combined, bartlett, music, n_src   (first three are [bins, angles], linear power / unitless)
    """
    a = steering(sin_axis())

    S_move = S - S.mean(axis=0, keepdims=True)          # moving echoes only: MUSIC is about the chests

    R_move = covariances(S_move)

    R_bart = R_move if REMOVE_STATIC_FOR_MAP else covariances(S)

    bartlett = np.real(np.einsum("nt,bnm,mt->bt", a.conj(), R_bart, a))

    Rm = slab_average(R_move, MUSIC_WIN)

    if MUSIC_FB and uniform_line():
        Rm = forward_backward(Rm)

    max_src = max(1, min(M_TARGETS, N_RX - 1))

    music, n_src = music_pseudospectrum(Rm, a, max_src, MUSIC_EIG_RATIO)

    combined = bartlett * music ** MUSIC_EXPONENT if USE_MUSIC else bartlett

    return combined, bartlett, music, n_src
