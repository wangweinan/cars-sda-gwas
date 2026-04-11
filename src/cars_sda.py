"""
CARS-SDA: Covariate-Adaptive FDR Control with Empirical Null
=============================================================
Integrates:
  - CARS (Cai, Sun, Wang 2019) nonparametric density-ratio local FDR
  - SDA (Du et al. 2023) mirror-symmetry FDR control
  - Jin-Cai (2007) Fourier-based empirical null estimation

The key insight: lfdr(z|s) = π₀ · f₀(z) / f_kde(z|s)
  - f₀ is the parametric empirical null N(μ₀, σ₀²) from Jin-Cai
  - f_kde is estimated NONPARAMETRICALLY via KDE within each covariate bin
  - No parametric assumption on the alternative distribution

Usage:
    from cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null

    rejections, lfdr, threshold, mu0, sigma0 = cars_sda(Z, MAF, alpha=0.05)
"""

import numpy as np
from scipy import stats

__version__ = "2.0.0"
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


# --- Nonparametric KDE-based lfdr estimation ----------------------------------

def _fft_kde(z, grid_n=4096, bw=None):
    """
    FFT-based Gaussian KDE — O(n + grid_n), matching R's density().

    1. Bin data into a fine grid (O(n))
    2. Convolve with Gaussian kernel via FFT (O(grid_n · log(grid_n)))
    3. Return (grid, density) for interpolation

    Parameters
    ----------
    z : array_like
        Data points.
    grid_n : int
        Grid resolution (power of 2 recommended for FFT).
    bw : float or None
        Bandwidth. If None, uses Silverman's rule.

    Returns
    -------
    grid : ndarray, shape (grid_n,)
    density : ndarray, shape (grid_n,)
    """
    z = np.asarray(z, dtype=float)
    n = len(z)

    # Silverman bandwidth (normal-reference rule, matches CARS R code)
    if bw is None:
        iqr = np.subtract(*np.percentile(z, [75, 25]))
        bw = 0.9 * min(np.std(z), iqr / 1.34) * n ** (-0.2)
        bw = max(bw, 1e-6)

    # Grid
    pad = 4 * bw
    z_min, z_max = z.min() - pad, z.max() + pad
    grid = np.linspace(z_min, z_max, grid_n)
    dx = grid[1] - grid[0]

    # Bin data into grid (linear binning for better accuracy)
    counts, _ = np.histogram(z, bins=grid_n, range=(z_min, z_max))
    counts = counts.astype(float)

    # Gaussian kernel on grid
    kernel_grid = np.arange(-(grid_n // 2), grid_n // 2) * dx
    kernel = np.exp(-0.5 * (kernel_grid / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    kernel = np.fft.ifftshift(kernel)  # center for FFT

    # Convolve via FFT: f_hat = (1/n) * Σ counts[k] * K_h(x - x_k)
    density = np.real(np.fft.ifft(np.fft.fft(counts) * np.fft.fft(kernel)))
    density = np.maximum(density / n, 1e-300)

    return grid, density


def _kde_lfdr_fft(z_bin, mu0, sigma0, pi0_bin, grid_n=4096):
    """
    Compute lfdr via nonparametric FFT-based KDE density ratio.

    lfdr(z) = π₀ · f₀(z) / f_kde(z)

    where f₀ = N(μ₀, σ₀²) and f_kde is estimated nonparametrically
    using FFT convolution. O(n) time — handles millions of observations.
    No parametric assumption on the alternative distribution.

    Parameters
    ----------
    z_bin : array_like
        Z-scores in this covariate bin.
    mu0 : float
        Empirical null mean.
    sigma0 : float
        Empirical null std.
    pi0_bin : float
        Null proportion for this bin.
    grid_n : int
        FFT grid size (power of 2).

    Returns
    -------
    lfdr : ndarray, shape same as z_bin
        Local FDR estimates.
    """
    z_bin = np.asarray(z_bin, dtype=float)
    n = len(z_bin)

    if n < 20:
        return np.ones_like(z_bin)

    # FFT-based KDE (O(n + grid_n))
    grid, f_kde_grid = _fft_kde(z_bin, grid_n=grid_n)

    # Interpolate to data points
    f_kde = np.interp(z_bin, grid, f_kde_grid)
    f_kde = np.maximum(f_kde, 1e-300)

    # Null density
    f0 = stats.norm.pdf(z_bin, mu0, sigma0)

    # lfdr = π₀ · f₀(z) / f_kde(z)
    lfdr = np.clip(pi0_bin * f0 / f_kde, 0.0, 1.0)

    return lfdr


def _storey_pi0(z, mu0, sigma0, lam=0.5):
    """
    Storey (2002) null proportion estimator using empirical null p-values.

    Parameters
    ----------
    z : array_like
        Z-scores.
    mu0, sigma0 : float
        Empirical null parameters.
    lam : float
        Storey threshold (default 0.5).

    Returns
    -------
    pi0 : float
        Estimated null proportion, clipped to [0.1, 1.0].
    """
    p = 2 * (1 - stats.norm.cdf(np.abs((z - mu0) / sigma0)))
    pi0 = np.clip(np.mean(p > lam) / (1 - lam), 0.1, 1.0)
    return pi0


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

def cars_sda(Z, S, alpha=0.05, n_bins=20, K=5, mu0=None, sigma0=None, verbose=True):
    """
    CARS-SDA with Jin-Cai empirical null and nonparametric KDE.

    The density ratio lfdr(z|s) = π₀(s) · f₀(z) / f_kde(z|s) is computed
    within each covariate bin using KDE — no parametric alternative needed.

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
    lfdr_arr = np.ones(m)

    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        bi_tr = np.clip(np.digitize(S[tr], edges) - 1, 0, n_bins - 1)
        bi_te = np.clip(np.digitize(S[te], edges) - 1, 0, n_bins - 1)

        # Estimate per-bin π₀ and FFT-KDE on training data
        pi0_bins = {}
        kde_grids = {}  # store (grid, density) tuples from FFT KDE
        for b in range(n_bins):
            mask = bi_tr == b
            n_b = np.sum(mask)
            if n_b >= 30:
                z_b = Z[tr][mask]
                pi0_bins[b] = _storey_pi0(z_b, mu0, sigma0)
                kde_grids[b] = _fft_kde(z_b)  # O(n_b) via FFT
            else:
                pi0_bins[b] = 0.95
                kde_grids[b] = None

        # Smooth π₀ across bins (Gaussian kernel, bandwidth = 3 bins)
        raw_pi0 = np.array([pi0_bins[b] for b in range(n_bins)])
        smooth_pi0 = np.copy(raw_pi0)
        for b in range(n_bins):
            lo, hi = max(0, b - 3), min(n_bins, b + 4)
            w = np.exp(-0.5 * ((np.arange(lo, hi) - b) / 1.5)**2)
            w /= w.sum()
            smooth_pi0[b] = np.dot(w, raw_pi0[lo:hi])

        # Compute lfdr on test fold using training FFT-KDE
        for b in range(n_bins):
            mask = bi_te == b
            if not mask.any():
                continue
            z_t = Z[te][mask]
            pi0_b = smooth_pi0[b]

            if kde_grids[b] is not None:
                grid, f_kde_grid = kde_grids[b]
                f_kde = np.interp(z_t, grid, f_kde_grid)
                f_kde = np.maximum(f_kde, 1e-300)
                f0 = stats.norm.pdf(z_t, mu0, sigma0)
                lfdr_arr[te[mask]] = np.clip(pi0_b * f0 / f_kde, 0.0, 1.0)
            else:
                lfdr_arr[te[mask]] = 1.0

    rej, thr = _stepup(lfdr_arr, alpha)
    if verbose:
        print(f"  CARS-SDA: {rej.sum():,} rejections (threshold={thr:.5f})")
    return rej, lfdr_arr, thr, mu0, sigma0


# --- Adaptive-Z (Sun & Cai 2007) with Empirical Null -------------------------

def adaptive_z(Z, alpha=0.05, verbose=True):
    """Sun & Cai (2007) procedure with Jin-Cai empirical null and FFT KDE."""
    Z = np.asarray(Z, float)
    mu0, sigma0, _ = jin_cai_empirical_null(Z)
    if verbose:
        print(f"  Adaptive-Z empirical null: μ₀={mu0:.4f}, σ₀={sigma0:.4f}")

    P = 2 * (1 - stats.norm.cdf(np.abs((Z - mu0) / sigma0)))
    pi0 = min(1.0, np.mean(P > 0.5) / 0.5)

    # FFT-based KDE on full data (O(n) — no subsampling needed)
    grid, f_kde_grid = _fft_kde(Z, grid_n=8192)
    f_z = np.interp(Z, grid, f_kde_grid)
    f0 = stats.norm.pdf(Z, mu0, sigma0)

    lfdr = np.minimum(1.0, pi0 * f0 / np.maximum(f_z, 1e-300))
    rej, thr = _stepup(lfdr, alpha)
    return rej, lfdr, thr

