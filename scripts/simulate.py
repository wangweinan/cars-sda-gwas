"""Quick simulation validation for CARS-SDA v3.0."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from src.cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null

def run_simulation(m=1_000_000, sigma0_true=1.16, seed=42):
    """Simulate GWAS-like data matching PGC-SCZ inflation properties."""
    rng = np.random.RandomState(seed)
    S = rng.uniform(0.01, 0.5, m)
    signal_prob = 0.005 + 0.015 * (S / 0.5)
    is_signal = rng.rand(m) < signal_prob
    Z = rng.normal(0, sigma0_true, m)

    n_sig = is_signal.sum()
    signs = rng.choice([-1, 1], n_sig)
    n_gauss = int(0.6 * n_sig)
    effects = np.zeros(n_sig)
    effects[:n_gauss] = np.abs(rng.normal(3.5, 1.0, n_gauss))
    effects[n_gauss:] = np.abs(rng.laplace(4.0, 1.5, n_sig - n_gauss))
    Z[is_signal] = signs * effects

    print(f"=== Simulation ===")
    print(f"  m={m:,}, σ₀={sigma0_true}, signals={n_sig:,} ({100*n_sig/m:.2f}%)")
    print(f"  Alternative: 60% N(3.5,1) + 40% Laplace(4.0,1.5)")

    # Jin-Cai
    mu0, sigma0, pi0 = jin_cai_empirical_null(Z)
    print(f"  Jin-Cai: μ₀={mu0:.4f}, σ₀={sigma0:.4f}, π₀={pi0:.4f}")

    results = {}

    # BH raw
    P = 2 * (1 - stats.norm.cdf(np.abs(Z)))
    from statsmodels.stats.multitest import multipletests
    rej_bh, _, _, _ = multipletests(P, alpha=0.05, method='fdr_bh')
    fdr = 1 - is_signal[rej_bh].mean() if rej_bh.sum() > 0 else 0
    pwr = (rej_bh & is_signal).sum() / max(1, n_sig)
    results['BH (raw)'] = (rej_bh.sum(), fdr, pwr)

    # BH GC
    Z_gc = (Z - mu0) / sigma0
    P_gc = 2 * (1 - stats.norm.cdf(np.abs(Z_gc)))
    rej_gc, _, _, _ = multipletests(P_gc, alpha=0.05, method='fdr_bh')
    fdr = 1 - is_signal[rej_gc].mean() if rej_gc.sum() > 0 else 0
    pwr = (rej_gc & is_signal).sum() / max(1, n_sig)
    results['BH (GC)'] = (rej_gc.sum(), fdr, pwr)

    # Adaptive-Z
    rej_az, _, _ = adaptive_z(Z, alpha=0.05, verbose=False)
    fdr = 1 - is_signal[rej_az].mean() if rej_az.sum() > 0 else 0
    pwr = (rej_az & is_signal).sum() / max(1, n_sig)
    results['Adaptive-Z'] = (rej_az.sum(), fdr, pwr)

    # CARS-SDA (proper bivariate)
    rej_cs, cars_stat, thr, _, _, diag = cars_sda(Z, S, alpha=0.05, verbose=True)
    fdr = 1 - is_signal[rej_cs].mean() if rej_cs.sum() > 0 else 0
    pwr = (rej_cs & is_signal).sum() / max(1, n_sig)
    results['CARS-SDA'] = (rej_cs.sum(), fdr, pwr)

    # Summary
    print(f"\n{'='*65}")
    print(f"  {'Method':<20} {'Rej':>8} {'FDR':>8} {'Power':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for name, (n_rej, fdr, pwr) in results.items():
        flag = '✗' if fdr > 0.055 else '✓'
        print(f"  {name:<20} {n_rej:>8,} {100*fdr:>7.2f}% {100*pwr:>7.2f}%  {flag}")

    pwr_gc = results['BH (GC)'][2]
    pwr_az = results['Adaptive-Z'][2]
    pwr_cs = results['CARS-SDA'][2]
    fdr_cs = results['CARS-SDA'][1]
    print(f"\n  CARS vs BH(GC):  +{100*(pwr_cs/max(pwr_gc,0.001)-1):.1f}% power")
    print(f"  CARS vs Adapt-Z: +{100*(pwr_cs/max(pwr_az,0.001)-1):.1f}% power")
    print(f"  FDR={100*fdr_cs:.2f}% {'✓ controlled' if fdr_cs <= 0.055 else '✗ INFLATED'}")

if __name__ == "__main__":
    run_simulation()
