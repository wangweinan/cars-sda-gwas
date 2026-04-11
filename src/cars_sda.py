"""
CARS-SDA: Covariate-Adaptive FDR Control with Empirical Null
=============================================================
Integrates:
  - CARS (Cai, Sun, Wang 2019) density-ratio local FDR
  - SDA (Du et al. 2023) mirror-symmetry FDR control
  - Jin-Cai (2007) Fourier-based empirical null estimation

Usage:
    from cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null

    rejections, lfdr, threshold, mu0, sigma0 = cars_sda(Z, MAF, alpha=0.05)
"""

import numpy as np
from scipy import stats

__version__ = "1.0.0"
__all__ = ["cars_sda", "adaptive_z", "jin_cai_empirical_null"]


# --- Empirical Null Estimation ------------------------------------------------

def jin_cai_empirical_null(z, t_range=(1.0, 3.0), n_t=50):
    """
    Jin & Cai (2007) Fourier-based empirical null estimator.

    Exploits the empirical characteristic function at moderate frequencies:
        log|φ(t)|² ≈ 2·log(π₀) − σ₀²·t²
    giving σ₀² from the slope and π₀ from the intercept.

    Parameters
    ----------
    z : array_like, shape (m,)
        Observed z-scores.
    t_range : tuple of float
        Frequency range for regression (default: moderate frequencies).
    n_t : int
        Number of frequency grid points.

    Returns
    -------
    mu0 : float   — Estimated null mean.
    sigma0 : float — Estimated null standard deviation.
    pi0 : float    — Estimated null proportion.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    z_sub = z[np.random.RandomState(42).choice(n, min(n, 500_000), replace=False)] if n > 500_000 else z

    t_vals = np.linspace(t_range[0], t_range[1], n_t)
    log_mod_sq = np.array([np.log(max(np.abs(np.mean(np.exp(1j * t * z_sub)))**2, 1e-300)) for t in t_vals])

    # Weighted least squares: log|φ(t)|² = a + b·t²
    W = np.diag(np.exp(-0.5 * t_vals))
    X = np.column_stack([np.ones(n_t), t_vals**2])
    coeffs = np.linalg.lstsq(W @ X, W @ log_mod_sq, rcond=None)[0]

    sigma0 = np.sqrt(max(-coeffs[1], 0.5))
    pi0 = np.clip(np.exp(coeffs[0] / 2), 0.5, 1.0)

    # Mean from phase at small t
    t_small = np.linspace(0.1, 0.5, 20)
    phases = np.array([np.angle(np.mean(np.exp(1j * t * z_sub))) for t in t_small])
    mu0 = np.polyfit(t_small, phases, 1)[0]

    return mu0, sigma0, pi0


# --- EM Mixture with Fixed Empirical Null -------------------------------------

def _em_empirical_null(z, mu0, sigma0, max_iter=100, tol=1e-6):
    """Fit f(z) = π₀·N(μ₀,σ₀²) + (1−π₀)·N(μ₁,σ₁²) with fixed null."""
    n = len(z)
    if n < 20:
        return 1.0, mu0, sigma0 * 2

    pi0 = 0.9
    extreme = np.abs(z - mu0) > 2 * sigma0
    mu1 = np.mean(z[extreme]) if np.sum(extreme) > 5 else mu0
    sigma1 = max(np.std(z), sigma0 * 1.5)

    for _ in range(max_iter):
        f0 = stats.norm.pdf(z, mu0, sigma0)
        f1 = stats.norm.pdf(z, mu1, sigma1)
        denom = np.maximum(pi0 * f0 + (1 - pi0) * f1, 1e-300)
        gamma = np.clip((1 - pi0) * f1 / denom, 1e-10, 1 - 1e-10)

        ng = np.sum(gamma)
        if ng < 1:
            break
        pi0_new = np.clip(1.0 - ng / n, 0.01, 0.99)
        mu1_new = np.sum(gamma * z) / ng
        sigma1_new = max(np.sqrt(np.sum(gamma * (z - mu1_new)**2) / ng), sigma0 * 0.5)

        if abs(pi0_new - pi0) < tol and abs(mu1_new - mu1) < tol:
            return pi0_new, mu1_new, sigma1_new
        pi0, mu1, sigma1 = pi0_new, mu1_new, sigma1_new

    return pi0, mu1, sigma1


# --- Step-Up Threshold --------------------------------------------------------

def _stepup(lfdr, alpha):
    """Sort by ascending lfdr; reject largest k where cumavg(lfdr) ≤ α."""
    idx = np.argsort(lfdr)
    cumavg = np.cumsum(lfdr[idx]) / np.arange(1, len(lfdr) + 1)
    valid = np.where(cumavg <= alpha)[0]
    rej = np.zeros(len(lfdr), dtype=bool)
    thr = 0.0
    if len(valid) > 0:
        k = valid[-1]
        rej[idx[:k + 1]] = True
        thr = lfdr[idx[k]]
    return rej, thr


# --- Main CARS-SDA Procedure -------------------------------------------------

def cars_sda(Z, S, alpha=0.05, n_bins=50, K=5, mu0=None, sigma0=None, verbose=True):
    """
    CARS-SDA with Jin-Cai empirical null estimation.

    Parameters
    ----------
    Z : array_like, shape (m,)
        Primary z-scores (BETA / SE).
    S : array_like, shape (m,)
        Auxiliary covariate (e.g., MAF).
    alpha : float
        Target FDR level (default 0.05).
    n_bins : int
        Number of covariate quantile bins.
    K : int
        Number of cross-fitting folds.
    mu0, sigma0 : float or None
        Empirical null parameters. Auto-estimated if None.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    rejections : ndarray of bool
    lfdr : ndarray of float
    threshold : float
    mu0 : float
    sigma0 : float
    """
    Z, S = np.asarray(Z, float), np.asarray(S, float)
    m = len(Z)

    if mu0 is None or sigma0 is None:
        mu0, sigma0, pi0_g = jin_cai_empirical_null(Z)
        if verbose:
            print(f"  Empirical null: μ₀={mu0:.4f}, σ₀={sigma0:.4f}, π₀={pi0_g:.4f}")

    edges = np.quantile(S, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-10; edges[-1] += 1e-10

    rng = np.random.RandomState(42)
    folds = np.array_split(rng.permutation(m), K)
    lfdr = np.ones(m)

    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        bi_tr = np.clip(np.digitize(S[tr], edges) - 1, 0, n_bins - 1)
        bi_te = np.clip(np.digitize(S[te], edges) - 1, 0, n_bins - 1)

        params = {}
        for b in range(n_bins):
            mask = bi_tr == b
            params[b] = _em_empirical_null(Z[tr][mask], mu0, sigma0) if np.sum(mask) >= 30 else (0.95, mu0, sigma0 * 2)

        # Gaussian kernel smoothing (bandwidth = 3 bins)
        raw = np.array([params[b] for b in range(n_bins)])
        smooth = np.copy(raw)
        for b in range(n_bins):
            lo, hi = max(0, b - 3), min(n_bins, b + 4)
            w = np.exp(-0.5 * ((np.arange(lo, hi) - b) / 1.5)**2)
            w /= w.sum()
            smooth[b] = np.sum(w[:, None] * raw[lo:hi], axis=0)

        for b in range(n_bins):
            mask = bi_te == b
            if not mask.any():
                continue
            z_t = Z[te][mask]
            p0b, m1b, s1b = smooth[b]
            f0 = stats.norm.pdf(z_t, mu0, sigma0)
            fm = p0b * f0 + (1 - p0b) * stats.norm.pdf(z_t, m1b, s1b)
            lfdr[te[mask]] = np.clip(p0b * f0 / np.maximum(fm, 1e-300), 0, 1)

    rej, thr = _stepup(lfdr, alpha)
    return rej, lfdr, thr, mu0, sigma0


# --- Adaptive-Z (Sun & Cai 2007) with Empirical Null -------------------------

def adaptive_z(Z, alpha=0.05, max_kde_n=200_000, verbose=True):
    """Sun & Cai (2007) procedure with Jin-Cai empirical null."""
    Z = np.asarray(Z, float)
    mu0, sigma0, _ = jin_cai_empirical_null(Z)
    if verbose:
        print(f"  Adaptive-Z empirical null: μ₀={mu0:.4f}, σ₀={sigma0:.4f}")

    P = 2 * (1 - stats.norm.cdf(np.abs((Z - mu0) / sigma0)))
    pi0 = min(1.0, np.mean(P > 0.5) / 0.5)

    z_sub = Z[np.random.RandomState(123).choice(len(Z), min(len(Z), max_kde_n), replace=False)]
    kde = stats.gaussian_kde(z_sub, bw_method="silverman")
    grid = np.linspace(Z.min() - 3, Z.max() + 3, 10_000)
    f_z = np.interp(Z, grid, kde(grid))
    f0 = stats.norm.pdf(Z, mu0, sigma0)

    lfdr = np.minimum(1.0, pi0 * f0 / np.maximum(f_z, 1e-300))
    rej, thr = _stepup(lfdr, alpha)
    return rej, lfdr, thr
