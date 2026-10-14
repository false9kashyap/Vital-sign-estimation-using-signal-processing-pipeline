"""
vmd_tools.py  -  Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014), 1-D real signals

    modes, centres = vmd(x, fs, K, alpha=...)

Splits x into K narrow-band modes u_k, each concentrated around its own centre frequency w_k:

    minimise   sum_k  || d/dt [ (delta(t) + j/(pi t)) * u_k(t) ] exp(-j w_k t) ||^2
    subject to sum_k u_k = x

The solution is found by alternating updates in the frequency domain (ADMM):

    U_k(w) = ( F(w) - sum_{i!=k} U_i(w) + L(w)/2 ) / ( 1 + alpha (w - w_k)^2 )
    w_k    = sum w |U_k(w)|^2 / sum |U_k(w)|^2            (centre of gravity of the spectrum)

alpha  = bandwidth constraint: large alpha -> narrow modes. (w in cycles/sample, so for a given alpha
         the half-power half-width of a mode is about fs / sqrt(alpha) Hz.)
tau    = dual-ascent step; 0 = no strict reconstruction, tolerant to noise (recommended for radar).
f_init = starting centre frequencies in Hz (K values). Choosing them inside the band of interest
         makes the result stable.

Returns the modes sorted by centre frequency (low to high), shape (K, len(x)), and the centres in Hz.
"""

import numpy as np


def vmd(x, fs, K, alpha=2000.0, tau=0.0, tol=1e-6, max_iter=300, f_init=None):

    x = np.asarray(x, dtype=float)

    T = len(x)
    half = T // 2

    # mirror the signal at both ends: reduces edge artefacts
    xm = np.concatenate([x[:half][::-1], x, x[T - half:][::-1]])

    L = len(xm)

    f = np.fft.fftfreq(L)                         # cycles/sample
    pos = f > 0                                   # analytic signal: positive frequencies only

    F = np.fft.fft(xm) * pos

    if f_init is None:
        omega = (0.5 / K) * np.arange(K) + 0.25 / K
    else:
        omega = np.asarray(f_init, dtype=float) / fs

    omega = np.array(omega, dtype=float)

    u = np.zeros((K, L), dtype=np.complex128)
    lam = np.zeros(L, dtype=np.complex128)

    fp = f[pos]

    for _ in range(max_iter):

        u_old = u.copy()

        for k in range(K):

            others = u.sum(axis=0) - u[k]

            u[k] = (F - others + lam / 2.0) / (1.0 + alpha * (f - omega[k]) ** 2)

            p = np.abs(u[k][pos]) ** 2

            omega[k] = (fp * p).sum() / (p.sum() + 1e-30)

        if tau > 0:
            lam = lam + tau * (u.sum(axis=0) - F)

        change = np.sum(np.abs(u - u_old) ** 2) / (np.sum(np.abs(u_old) ** 2) + 1e-30)

        if change < tol:
            break

    modes = np.array([2.0 * np.real(np.fft.ifft(u[k]))[half:half + T] for k in range(K)])

    order = np.argsort(omega)

    return modes[order], omega[order] * fs
