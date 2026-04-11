"""
Simulation validation: verify CARS-SDA controls FDR on data matching PGC-SCZ
distribution (overdispersed null σ₀ ≈ 1.16, covariate-dependent sparsity).
"""
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null, _stepup


def simulate(m=1_000_000, seed=42):
    """Generate GWAS-like data with overdispersed null."""
    rng = np.random.RandomState(seed)
    S = np.clip(rng.beta(0.8, 2.5, m), 0.005, 0.499)
    sigma0, mu0 = 1.16, -0.01
    pi1 = 0.01 + 0.04 / (1 + np.exp(-10 * (S - 0.25)))
    signal = rng.binomial(1, pi1).astype(bool)
    Z = np.where(signal,
                 rng.choice([-1, 1], m) * rng.normal(np.abs(2.5 + 4 * S + rng.normal(0, 0.5, m)), 1),
                 rng.normal(mu0, sigma0, m))
    return Z, S, signal, mu0, sigma0


def evaluate(name, rej, signal):
    td = int(np.sum(rej & signal))
    fd = int(np.sum(rej & ~signal))
    total = td + fd
    m1 = int(np.sum(signal))
    power = td / m1 * 100
    fdr = fd / max(total, 1) * 100
    ok = "✅" if fdr <= 5.5 else "❌"
    print(f"  {name:<35} {total:>7} {td:>7} {fd:>6} {power:>8.2f}% {fdr:>8.2f}%  {ok}")


def main():
    print("=" * 78)
    print("SIMULATION VALIDATION (m=1M, σ₀=1.16, covariate-dep sparsity)")
    print("=" * 78)

    Z, S, signal, mu0_true, sigma0_true = simulate()
    P = 2 * (1 - stats.norm.cdf(np.abs(Z)))
    m1 = int(np.sum(signal))
    print(f"  Signals: {m1:,} ({m1/len(Z)*100:.1f}%), λ_GC: {np.median(Z**2)/0.4549:.3f}\n")

    # Empirical null estimation
    mu0_jc, sigma0_jc, pi0_jc = jin_cai_empirical_null(Z)
    print(f"  True null:    μ₀={mu0_true:.4f}, σ₀={sigma0_true:.4f}")
    print(f"  Jin-Cai est:  μ₀={mu0_jc:.4f}, σ₀={sigma0_jc:.4f}, π₀={pi0_jc:.4f}\n")

    # Methods
    rej_bh, _, _, _ = multipletests(P, alpha=0.05, method='fdr_bh')

    lam = np.median(Z**2) / 0.4549
    P_gc = 2 * (1 - stats.norm.cdf(np.abs(Z / np.sqrt(lam))))
    rej_bh_gc, _, _, _ = multipletests(P_gc, alpha=0.05, method='fdr_bh')

    rej_az, _, _ = adaptive_z(Z, verbose=False)
    rej_cs, _, _, _, _ = cars_sda(Z, S, verbose=False)

    print(f"\n  {'Method':<35} {'Total':>7} {'True':>7} {'False':>6} {'Power':>9} {'FDR':>9}")
    print("  " + "-" * 76)
    evaluate("BH (theoretical null)", rej_bh, signal)
    evaluate("BH (GC-corrected)", rej_bh_gc, signal)
    evaluate("Adaptive-Z (empirical null)", rej_az, signal)
    evaluate("CARS-SDA (empirical null)", rej_cs, signal)


if __name__ == "__main__":
    main()
