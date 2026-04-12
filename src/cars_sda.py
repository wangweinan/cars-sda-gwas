"""
CARS-SDA: Covariate-Adaptive FDR Control with Empirical Null
=============================================================
Integrates:
  - CARS (Cai, Sun, Wang 2019) bivariate density-ratio statistic
  - Jin-Cai (2007) Fourier-based empirical null estimation
  - Barber-Candès step-up for FDR control

NOTE: Despite the name, we do NOT currently implement the full
SDA step-down dependency adjustment (Du et al. 2023). Our step-up
procedure treats observations as independent. For LD-correlated
GWAS data, FDR control relies on the PRDS condition being satisfied
(plausible under positive LD, but not formally proven).
The "SDA" name is retained for continuity but should be understood
as aspirational — proper LD-aware step-down is a future extension.

The CARS statistic is the bivariate analogue of lfdr:

              f₀(Z) · (|T_τ|/m) · f*(S | T_τ)
  CARS(i) = ─────────────────────────────────────
                correction · f(Z, S)

Where:
  - f₀ = N(μ₀, σ₀²) from Jin-Cai empirical null
  - T_τ = {i : lfdr_marginal ≥ τ} (null-like screening set)
  - f*(S | T_τ) = covariate density among null-like obs
  - f(Z, S) = bivariate joint density (2D FFT-KDE)
  - correction = P(lfdr ≥ τ | H₀) via Monte Carlo

Usage:
    from cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null

    rejections, cars_stat, threshold, mu0, sigma0 = cars_sda(Z, MAF, alpha=0.05)
"""

import numpy as np
from scipy import stats, interpolate

__version__ = "3.0.0"
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
    rng = np.random.RandomState(42)
    z_sub = z[rng.choice(n, min(n, 500_000), replace=False)] if n > 500_000 else z

    t_vals = np.linspace(t_range[0], t_range[1], n_t)
    log_mod_sq = np.array([
        np.log(max(np.abs(np.mean(np.exp(1j * t * z_sub)))**2, 1e-300))
        for t in t_vals
    ])

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


# --- FFT-based KDE (1D and 2D) -----------------------------------------------

def _silverman_bw(x):
    """Silverman's rule-of-thumb bandwidth (normal-reference)."""
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(np.std(x), iqr / 1.34) if iqr > 0 else np.std(x)
    return max(0.9 * sigma * n ** (-0.2), 1e-8)


def _fft_kde(z, grid_n=4096, bw=None):
    """
    1D FFT-based Gaussian KDE — O(n + grid_n), matching R's density().

    Parameters
    ----------
    z : array_like — Data points.
    grid_n : int   — Grid resolution (power of 2).
    bw : float     — Bandwidth (Silverman if None).

    Returns
    -------
    grid : ndarray, shape (grid_n,)
    density : ndarray, shape (grid_n,)
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    if bw is None:
        bw = _silverman_bw(z)

    pad = 4 * bw
    z_min, z_max = z.min() - pad, z.max() + pad
    grid = np.linspace(z_min, z_max, grid_n)
    dx = grid[1] - grid[0]

    counts, _ = np.histogram(z, bins=grid_n, range=(z_min, z_max))
    counts = counts.astype(float)

    kernel_grid = np.arange(-(grid_n // 2), grid_n // 2) * dx
    kernel = np.exp(-0.5 * (kernel_grid / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    kernel = np.fft.ifftshift(kernel)

    density = np.real(np.fft.ifft(np.fft.fft(counts) * np.fft.fft(kernel)))
    density = np.maximum(density / n, 1e-300)

    return grid, density


def _fft_kde_2d(z, s, grid_nz=512, grid_ns=512, bw_z=None, bw_s=None):
    """
    2D FFT-based Gaussian KDE using product kernel.

    K_h(z, s) = K_{h1}(z) · K_{h2}(s)

    Exploits separability for O(n + G² log G) computation.

    Parameters
    ----------
    z, s : array_like, shape (n,)
        Primary and auxiliary variables.
    grid_nz, grid_ns : int
        Grid dimensions.
    bw_z, bw_s : float
        Bandwidths (Silverman if None).

    Returns
    -------
    grid_z : ndarray, shape (grid_nz,)
    grid_s : ndarray, shape (grid_ns,)
    density_2d : ndarray, shape (grid_nz, grid_ns)
    """
    z = np.asarray(z, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(z)

    if bw_z is None:
        bw_z = _silverman_bw(z)
    if bw_s is None:
        bw_s = _silverman_bw(s)

    # Grids
    pad_z, pad_s = 4 * bw_z, 4 * bw_s
    z_min, z_max = z.min() - pad_z, z.max() + pad_z
    s_min, s_max = s.min() - pad_s, s.max() + pad_s
    grid_z = np.linspace(z_min, z_max, grid_nz)
    grid_s = np.linspace(s_min, s_max, grid_ns)
    dz = grid_z[1] - grid_z[0]
    ds = grid_s[1] - grid_s[0]

    # 2D histogram
    counts, _, _ = np.histogram2d(
        z, s, bins=[grid_nz, grid_ns],
        range=[[z_min, z_max], [s_min, s_max]]
    )
    counts = counts.astype(float)

    # Separable product kernel
    kz = np.arange(-(grid_nz // 2), grid_nz // 2) * dz
    ks = np.arange(-(grid_ns // 2), grid_ns // 2) * ds
    kernel_z = np.exp(-0.5 * (kz / bw_z)**2) / (bw_z * np.sqrt(2 * np.pi))
    kernel_s = np.exp(-0.5 * (ks / bw_s)**2) / (bw_s * np.sqrt(2 * np.pi))
    kernel_z = np.fft.ifftshift(kernel_z)
    kernel_s = np.fft.ifftshift(kernel_s)

    # 2D kernel = outer product (separable)
    kernel_2d = np.outer(kernel_z, kernel_s)

    # 2D FFT convolution
    density = np.real(np.fft.ifft2(np.fft.fft2(counts) * np.fft.fft2(kernel_2d)))
    density = np.maximum(density / n, 1e-300)

    return grid_z, grid_s, density


def _interp_1d(x, grid, density):
    """Interpolate 1D KDE to data points, clamped to grid range."""
    return np.maximum(np.interp(x, grid, density), 1e-300)


def _interp_2d(z, s, grid_z, grid_s, density_2d):
    """Interpolate 2D KDE to data points via RegularGridInterpolator."""
    # Grid centers (histogram bin centers, not edges)
    interp_fn = interpolate.RegularGridInterpolator(
        (grid_z, grid_s), density_2d,
        method='linear', bounds_error=False, fill_value=1e-300
    )
    points = np.column_stack([z, s])
    return np.maximum(interp_fn(points), 1e-300)


# --- Storey π₀ estimation -----------------------------------------------------

def _storey_pi0(z, mu0, sigma0, lam=0.5):
    """Storey (2002) null proportion estimator using empirical-null p-values."""
    p = 2 * (1 - stats.norm.cdf(np.abs((z - mu0) / sigma0)))
    return np.clip(np.mean(p > lam) / (1 - lam), 0.1, 1.0)


# --- Step-Up Threshold --------------------------------------------------------

def _stepup(statistic, alpha):
    """Sort by ascending statistic; reject largest k where cumavg ≤ α."""
    idx = np.argsort(statistic)
    cumavg = np.cumsum(statistic[idx]) / np.arange(1, len(statistic) + 1)
    valid = np.where(cumavg <= alpha)[0]
    rej = np.zeros(len(statistic), dtype=bool)
    thr = 0.0
    if len(valid) > 0:
        k = valid[-1]
        rej[idx[:k + 1]] = True
        thr = statistic[idx[k]]
    return rej, thr


# --- Main CARS-SDA Procedure -------------------------------------------------

def cars_sda(Z, S, alpha=0.05, tau=0.9, mu0=None, sigma0=None, verbose=True):
    """
    CARS-SDA: proper bivariate density-ratio statistic.

    Implements the CARS construction from Cai, Sun & Wang (2019):

                  f₀(Z) · (|T_τ|/m) · f*(S | T_τ)
      CARS(i) = ─────────────────────────────────────
                     correction · f(Z, S)

    Parameters
    ----------
    Z : array_like, shape (m,)
        Primary z-scores (BETA / SE).
    S : array_like, shape (m,)
        Auxiliary covariate (e.g., MAF).
    alpha : float
        Target FDR level (default 0.05).
    tau : float
        Screening threshold for null-like set (default 0.9).
    mu0, sigma0 : float or None
        Empirical null parameters. Auto-estimated if None.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    rejections : ndarray of bool
    cars_stat : ndarray of float   — The CARS statistic (bivariate lfdr analogue).
    threshold : float
    mu0 : float
    sigma0 : float
    diagnostics : dict              — Intermediate quantities for validation.
    """
    Z, S = np.asarray(Z, float), np.asarray(S, float)
    m = len(Z)

    # --- Step 0: Empirical null ---
    if mu0 is None or sigma0 is None:
        mu0, sigma0, pi0_jc = jin_cai_empirical_null(Z)
        if verbose:
            print(f"  Jin-Cai null: μ₀={mu0:.4f}, σ₀={sigma0:.4f}, π₀={pi0_jc:.4f}")

    # --- Step 1: Marginal lfdr for Z (screening only) ---
    f0 = stats.norm.pdf(Z, mu0, sigma0)
    grid_z, fhat_z = _fft_kde(Z, grid_n=8192)
    f_z = _interp_1d(Z, grid_z, fhat_z)
    pi0 = _storey_pi0(Z, mu0, sigma0)
    lfdr_marginal = np.clip(pi0 * f0 / f_z, 0.0, 1.0)

    if verbose:
        print(f"  Storey π₀={pi0:.4f}, marginal lfdr median={np.median(lfdr_marginal):.4f}")

    # --- Step 2: Null-like screening set T_τ ---
    T_tau = lfdr_marginal >= tau
    n_tau = T_tau.sum()
    if verbose:
        print(f"  T_τ (lfdr≥{tau}): {n_tau:,} / {m:,} ({100*n_tau/m:.1f}%)")

    # --- Step 3: Monte Carlo correction ---
    #   P(lfdr_marginal ≥ τ | H₀) — what fraction of true nulls land in T_τ
    rng = np.random.RandomState(42)
    z_null = rng.normal(mu0, sigma0, 50_000)
    f_null_kde = _interp_1d(z_null, grid_z, fhat_z)
    f_null_true = stats.norm.pdf(z_null, mu0, sigma0)
    lfdr_null = np.clip(pi0 * f_null_true / f_null_kde, 0.0, 1.0)
    correction = np.mean(lfdr_null >= tau)
    correction = max(correction, 0.01)  # floor to prevent blow-up
    if verbose:
        print(f"  Correction P(lfdr≥{tau}|H₀) = {correction:.4f}")

    # --- Step 4: Auxiliary density among null-like observations ---
    grid_s_null, fstar_s = _fft_kde(S[T_tau], grid_n=4096)
    f_s_given_null = _interp_1d(S, grid_s_null, fstar_s)

    # --- Step 5: Joint density f(Z, S) via 2D FFT-KDE ---
    grid_z2d, grid_s2d, f_joint_2d = _fft_kde_2d(Z, S)
    f_joint = _interp_2d(Z, S, grid_z2d, grid_s2d, f_joint_2d)

    if verbose:
        print(f"  2D KDE: grid {len(grid_z2d)}×{len(grid_s2d)}, "
              f"f_joint range [{f_joint.min():.2e}, {f_joint.max():.2e}]")

    # --- Step 6: CARS statistic ---
    numerator = f0 * (n_tau / m) * f_s_given_null / correction
    cars_stat = np.clip(numerator / f_joint, 0.0, 1.0)

    # --- Step 7: Step-up procedure ---
    rej, thr = _stepup(cars_stat, alpha)

    if verbose:
        print(f"  CARS-SDA: {rej.sum():,} rejections (threshold={thr:.5f})")

    diagnostics = {
        'lfdr_marginal': lfdr_marginal,
        'f0': f0, 'f_z': f_z, 'f_joint': f_joint,
        'f_s_given_null': f_s_given_null,
        'grid_z': grid_z, 'fhat_z': fhat_z,
        'grid_s_null': grid_s_null, 'fstar_s': fstar_s,
        'T_tau': T_tau, 'correction': correction,
        'pi0': pi0, 'mu0': mu0, 'sigma0': sigma0,
    }

    return rej, cars_stat, thr, mu0, sigma0, diagnostics


# --- Adaptive-Z (Sun & Cai 2007) with Empirical Null -------------------------

def adaptive_z(Z, alpha=0.05, verbose=True):
    """
    Sun & Cai (2007) procedure with Jin-Cai empirical null and FFT KDE.

    This is the marginal lfdr procedure — no covariate information used.
    Serves as the baseline to measure CARS' improvement.
    """
    Z = np.asarray(Z, float)
    mu0, sigma0, _ = jin_cai_empirical_null(Z)
    if verbose:
        print(f"  Adaptive-Z empirical null: μ₀={mu0:.4f}, σ₀={sigma0:.4f}")

    P = 2 * (1 - stats.norm.cdf(np.abs((Z - mu0) / sigma0)))
    pi0 = min(1.0, np.mean(P > 0.5) / 0.5)

    grid, f_kde_grid = _fft_kde(Z, grid_n=8192)
    f_z = np.interp(Z, grid, f_kde_grid)
    f0 = stats.norm.pdf(Z, mu0, sigma0)

    lfdr = np.minimum(1.0, pi0 * f0 / np.maximum(f_z, 1e-300))
    rej, thr = _stepup(lfdr, alpha)
    return rej, lfdr, thr
