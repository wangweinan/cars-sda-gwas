"""
Comprehensive validation for CARS-SDA v3.0.

Validates:
  1. Jin-Cai empirical null fit (QQ plot, KS test)
  2. Marginal KDE vs histogram + null overlay
  3. Conditional independence Z ⊥ MAF | H₀
  4. CARS statistic calibration curve
  5. Multi-seed FDR/power table
  6. Head-to-head: BH-raw, BH-GC, Adaptive-Z, CARS-SDA
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null, _fft_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'validation')
os.makedirs(OUT_DIR, exist_ok=True)


def simulate_gwas(m=1_000_000, sigma0=1.16, seed=42):
    """Simulate GWAS with non-Gaussian alternative and MAF-dependent sparsity."""
    rng = np.random.RandomState(seed)
    S = rng.uniform(0.01, 0.5, m)
    signal_prob = 0.005 + 0.015 * (S / 0.5)
    is_signal = rng.rand(m) < signal_prob
    Z = rng.normal(0, sigma0, m)

    n_sig = is_signal.sum()
    signs = rng.choice([-1, 1], n_sig)
    n_gauss = int(0.6 * n_sig)
    effects = np.zeros(n_sig)
    effects[:n_gauss] = np.abs(rng.normal(3.5, 1.0, n_gauss))
    effects[n_gauss:] = np.abs(rng.laplace(4.0, 1.5, n_sig - n_gauss))
    Z[is_signal] = signs * effects

    return Z, S, is_signal


def plot_null_fit(Z, mu0, sigma0, title_suffix=""):
    """Diagnostic 1: QQ plot + histogram with null overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- QQ Plot ---
    ax = axes[0]
    # Use only central observations (|Z| < 3*sigma0) for QQ
    central = np.abs(Z - mu0) < 3 * sigma0
    z_central = (Z[central] - mu0) / sigma0
    n_qq = min(len(z_central), 50_000)
    z_sample = np.sort(np.random.RandomState(0).choice(z_central, n_qq, replace=False))
    theoretical = stats.norm.ppf(np.linspace(1/(n_qq+1), n_qq/(n_qq+1), n_qq))
    ax.scatter(theoretical, z_sample, s=0.3, alpha=0.3, color='steelblue')
    lim = max(abs(theoretical.min()), abs(theoretical.max())) + 0.5
    ax.plot([-lim, lim], [-lim, lim], 'r-', lw=1.5, label='y=x')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Theoretical N(0,1) quantiles')
    ax.set_ylabel(f'Standardized Z = (Z−{mu0:.3f})/{sigma0:.3f}')
    ax.set_title(f'QQ Plot: Empirical Null Fit {title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)

    # KS test
    ks_stat, ks_p = stats.kstest(z_central[:100_000] if len(z_central) > 100_000 else z_central,
                                  'norm', args=(0, 1))
    ax.text(0.05, 0.92, f'KS statistic: {ks_stat:.4f}\nKS p-value: {ks_p:.2e}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Histogram + KDE + Null ---
    ax = axes[1]
    z_range = Z[(Z > -8) & (Z < 8)]
    ax.hist(z_range, bins=200, density=True, alpha=0.5, color='steelblue', label='Observed Z')

    grid, fhat = _fft_kde(Z, grid_n=8192)
    mask = (grid > -8) & (grid < 8)
    ax.plot(grid[mask], fhat[mask], 'orange', lw=2, label=f'KDE f(z)')
    x_null = np.linspace(-8, 8, 1000)
    ax.plot(x_null, stats.norm.pdf(x_null, mu0, sigma0), 'r--', lw=2,
            label=f'Null N({mu0:.3f}, {sigma0:.3f}²)')
    ax.plot(x_null, stats.norm.pdf(x_null, 0, 1), 'gray', lw=1, ls=':',
            label='N(0,1) naive')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Density')
    ax.set_title(f'Mixture: Histogram + KDE + Null {title_suffix}')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'null_fit{title_suffix.replace(" ","_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")
    return ks_stat, ks_p


def plot_conditional_independence(Z, S, mu0, sigma0, is_signal=None, title_suffix=""):
    """Diagnostic 3: Z ⊥ MAF check under estimated null."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Null-like observations: |Z - mu0| < 2*sigma0
    null_like = np.abs(Z - mu0) < 2 * sigma0

    ax = axes[0]
    # Binned mean |Z| vs MAF for null-like obs
    n_qbins = 20
    maf_edges = np.quantile(S[null_like], np.linspace(0, 1, n_qbins + 1))
    maf_mids, mean_absz, std_absz = [], [], []
    for i in range(n_qbins):
        mask = null_like & (S >= maf_edges[i]) & (S < maf_edges[i+1])
        if mask.sum() > 100:
            maf_mids.append((maf_edges[i] + maf_edges[i+1]) / 2)
            mean_absz.append(np.mean(np.abs(Z[mask] - mu0)))
            std_absz.append(np.std(np.abs(Z[mask] - mu0)) / np.sqrt(mask.sum()))

    ax.errorbar(maf_mids, mean_absz, yerr=std_absz, fmt='o-', color='steelblue',
                markersize=4, capsize=3)
    expected = sigma0 * np.sqrt(2 / np.pi)
    ax.axhline(expected, color='red', ls='--', label=f'E[|Z|] = {expected:.3f} (null)')
    ax.set_xlabel('MAF')
    ax.set_ylabel('Mean |Z − μ₀| among null-like')
    ax.set_title(f'Conditional Independence: Z ⊥ MAF | H₀ {title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)

    # Correlation test
    z_null = Z[null_like]
    s_null = S[null_like]
    corr, corr_p = stats.pearsonr(z_null[:500_000] if len(z_null) > 500_000 else z_null,
                                   s_null[:500_000] if len(s_null) > 500_000 else s_null)
    ax.text(0.05, 0.12, f'Pearson r = {corr:.4f}, p = {corr_p:.2e}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Calibration: CARS stat vs actual FDP (if ground truth available)
    ax = axes[1]
    if is_signal is not None:
        ax.set_title(f'Signal enrichment by MAF {title_suffix}')
        maf_mids2, signal_frac = [], []
        maf_edges2 = np.quantile(S, np.linspace(0, 1, n_qbins + 1))
        for i in range(n_qbins):
            mask = (S >= maf_edges2[i]) & (S < maf_edges2[i+1])
            if mask.sum() > 100:
                maf_mids2.append((maf_edges2[i] + maf_edges2[i+1]) / 2)
                signal_frac.append(is_signal[mask].mean())
        ax.plot(maf_mids2, signal_frac, 'o-', color='orangered')
        ax.set_xlabel('MAF')
        ax.set_ylabel('Signal fraction')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No ground truth available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Calibration (no ground truth)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'cond_indep{title_suffix.replace(" ","_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


def plot_cars_calibration(cars_stat, is_signal, title_suffix=""):
    """Diagnostic 4: CARS statistic calibration (binned CARS vs actual FDP)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_idx = np.argsort(cars_stat)
    # Bin into 50 groups by CARS value
    n_bins = 50
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids, actual_fdp = [], []
    for i in range(n_bins):
        mask = (cars_stat >= bin_edges[i]) & (cars_stat < bin_edges[i+1])
        if mask.sum() > 50:
            bin_mids.append((bin_edges[i] + bin_edges[i+1]) / 2)
            actual_fdp.append(1 - is_signal[mask].mean())  # FDP = 1 - true positive rate

    ax.scatter(bin_mids, actual_fdp, color='steelblue', s=30, zorder=5)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax.set_xlabel('CARS statistic (binned)')
    ax.set_ylabel('Actual false discovery proportion')
    ax.set_title(f'CARS Calibration {title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'calibration{title_suffix.replace(" ","_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Saved {path}")


def multi_seed_validation(n_seeds=10, m=500_000, sigma0=1.16):
    """Diagnostic 5: Multi-seed FDR/power table."""
    print(f"\n{'='*75}")
    print(f"  Multi-seed validation: {n_seeds} seeds, m={m:,}, σ₀={sigma0}")
    print(f"{'='*75}")

    results = {'BH_raw': [], 'BH_GC': [], 'AdaptZ': [], 'CARS': []}
    fdr_results = {'BH_raw': [], 'BH_GC': [], 'AdaptZ': [], 'CARS': []}

    for seed in range(n_seeds):
        Z, S, is_signal = simulate_gwas(m=m, sigma0=sigma0, seed=seed)

        # BH raw
        P = 2 * (1 - stats.norm.cdf(np.abs(Z)))
        rej_bh, _, _, _ = multipletests(P, alpha=0.05, method='fdr_bh')
        fdr_bh = 1 - is_signal[rej_bh].mean() if rej_bh.sum() > 0 else 0
        pwr_bh = (rej_bh & is_signal).sum() / max(1, is_signal.sum())
        results['BH_raw'].append(pwr_bh)
        fdr_results['BH_raw'].append(fdr_bh)

        # BH GC
        mu0, sigma0_est, _ = jin_cai_empirical_null(Z)
        Z_gc = (Z - mu0) / sigma0_est
        P_gc = 2 * (1 - stats.norm.cdf(np.abs(Z_gc)))
        rej_gc, _, _, _ = multipletests(P_gc, alpha=0.05, method='fdr_bh')
        fdr_gc = 1 - is_signal[rej_gc].mean() if rej_gc.sum() > 0 else 0
        pwr_gc = (rej_gc & is_signal).sum() / max(1, is_signal.sum())
        results['BH_GC'].append(pwr_gc)
        fdr_results['BH_GC'].append(fdr_gc)

        # Adaptive-Z
        rej_az, _, _ = adaptive_z(Z, alpha=0.05, verbose=False)
        fdr_az = 1 - is_signal[rej_az].mean() if rej_az.sum() > 0 else 0
        pwr_az = (rej_az & is_signal).sum() / max(1, is_signal.sum())
        results['AdaptZ'].append(pwr_az)
        fdr_results['AdaptZ'].append(fdr_az)

        # CARS-SDA
        rej_cs, cars_stat, _, _, _, _ = cars_sda(Z, S, alpha=0.05, verbose=False)
        fdr_cs = 1 - is_signal[rej_cs].mean() if rej_cs.sum() > 0 else 0
        pwr_cs = (rej_cs & is_signal).sum() / max(1, is_signal.sum())
        results['CARS'].append(pwr_cs)
        fdr_results['CARS'].append(fdr_cs)

        print(f"  Seed {seed}: BH_GC={100*pwr_gc:.1f}% Adapt={100*pwr_az:.1f}% "
              f"CARS={100*pwr_cs:.1f}% | FDR: {100*fdr_gc:.1f}% {100*fdr_az:.1f}% {100*fdr_cs:.1f}%")

    # Summary table
    print(f"\n{'─'*75}")
    print(f"  {'Method':<12} {'Mean FDR':>10} {'SD FDR':>10} {'Mean Power':>12} {'SD Power':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")
    for name in ['BH_raw', 'BH_GC', 'AdaptZ', 'CARS']:
        fdr_arr = np.array(fdr_results[name])
        pwr_arr = np.array(results[name])
        flag = '✗' if fdr_arr.mean() > 0.055 else '✓'
        print(f"  {name:<12} {100*fdr_arr.mean():>9.2f}% {100*fdr_arr.std():>9.2f}% "
              f"{100*pwr_arr.mean():>11.2f}% {100*pwr_arr.std():>9.2f}%  {flag}")
    print(f"{'─'*75}")

    return results, fdr_results


def run_single_simulation(seed=42, m=1_000_000, sigma0=1.16):
    """Run one detailed simulation with all diagnostics."""
    print(f"\n{'='*75}")
    print(f"  Single simulation: seed={seed}, m={m:,}, σ₀={sigma0}")
    print(f"{'='*75}")

    Z, S, is_signal = simulate_gwas(m=m, sigma0=sigma0, seed=seed)
    n_sig = is_signal.sum()
    print(f"  Signals: {n_sig:,} ({100*n_sig/m:.2f}%)")

    # Jin-Cai
    mu0, sigma0_est, pi0 = jin_cai_empirical_null(Z)
    print(f"  Jin-Cai: μ₀={mu0:.4f} (true 0), σ₀={sigma0_est:.4f} (true {sigma0}), π₀={pi0:.4f}")

    # Diagnostic plots
    print("\n  --- Diagnostic 1: Null fit ---")
    ks_stat, ks_p = plot_null_fit(Z, mu0, sigma0_est, title_suffix=" (simulation)")

    print("\n  --- Diagnostic 3: Conditional independence ---")
    plot_conditional_independence(Z, S, mu0, sigma0_est, is_signal, title_suffix=" (simulation)")

    # Run all methods
    print("\n  --- Method comparison ---")

    # BH raw
    P = 2 * (1 - stats.norm.cdf(np.abs(Z)))
    rej_bh, _, _, _ = multipletests(P, alpha=0.05, method='fdr_bh')
    fdr_bh = 1 - is_signal[rej_bh].mean() if rej_bh.sum() > 0 else 0
    pwr_bh = (rej_bh & is_signal).sum() / max(1, n_sig)
    print(f"  BH (raw):      rej={rej_bh.sum():>8,}  FDR={100*fdr_bh:.2f}%  Power={100*pwr_bh:.2f}%")

    # BH GC
    Z_gc = (Z - mu0) / sigma0_est
    P_gc = 2 * (1 - stats.norm.cdf(np.abs(Z_gc)))
    rej_gc, _, _, _ = multipletests(P_gc, alpha=0.05, method='fdr_bh')
    fdr_gc = 1 - is_signal[rej_gc].mean() if rej_gc.sum() > 0 else 0
    pwr_gc = (rej_gc & is_signal).sum() / max(1, n_sig)
    print(f"  BH (GC):       rej={rej_gc.sum():>8,}  FDR={100*fdr_gc:.2f}%  Power={100*pwr_gc:.2f}%")

    # Adaptive-Z
    rej_az, _, _ = adaptive_z(Z, alpha=0.05, verbose=False)
    fdr_az = 1 - is_signal[rej_az].mean() if rej_az.sum() > 0 else 0
    pwr_az = (rej_az & is_signal).sum() / max(1, n_sig)
    print(f"  Adaptive-Z:    rej={rej_az.sum():>8,}  FDR={100*fdr_az:.2f}%  Power={100*pwr_az:.2f}%")

    # CARS-SDA
    rej_cs, cars_stat, thr, _, _, diag = cars_sda(Z, S, alpha=0.05, verbose=True)
    fdr_cs = 1 - is_signal[rej_cs].mean() if rej_cs.sum() > 0 else 0
    pwr_cs = (rej_cs & is_signal).sum() / max(1, n_sig)
    print(f"  CARS-SDA:      rej={rej_cs.sum():>8,}  FDR={100*fdr_cs:.2f}%  Power={100*pwr_cs:.2f}%")

    # CARS calibration
    print("\n  --- Diagnostic 4: CARS calibration ---")
    plot_cars_calibration(cars_stat, is_signal, title_suffix=" (simulation)")

    print(f"\n  CARS vs BH(GC): +{100*(pwr_cs/max(pwr_gc,0.001)-1):.1f}% relative power gain")
    print(f"  CARS vs Adapt-Z: +{100*(pwr_cs/max(pwr_az,0.001)-1):.1f}% relative power gain")

    return {
        'fdr': {'BH_raw': fdr_bh, 'BH_GC': fdr_gc, 'AdaptZ': fdr_az, 'CARS': fdr_cs},
        'power': {'BH_raw': pwr_bh, 'BH_GC': pwr_gc, 'AdaptZ': pwr_az, 'CARS': pwr_cs},
        'diagnostics': diag, 'ks_stat': ks_stat, 'ks_p': ks_p
    }


if __name__ == "__main__":
    # 1. Single detailed simulation
    result = run_single_simulation(seed=42, m=1_000_000, sigma0=1.16)

    # 2. Multi-seed robustness
    multi_seed_validation(n_seeds=5, m=500_000, sigma0=1.16)

    print("\n✓ All validation complete. Figures saved to:", OUT_DIR)
