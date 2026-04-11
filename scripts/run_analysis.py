"""
Run CARS-SDA v3.0 pipeline on PGC Schizophrenia GWAS (37.6M variants).
Uses proper bivariate CARS statistic with 2D FFT-KDE.
"""
import numpy as np, pandas as pd, os, sys, time
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cars_sda import cars_sda, adaptive_z, jin_cai_empirical_null

ALPHA = 0.05
DATA_PATH = os.environ.get("DATA_PATH", "../pgc_data/pgc-schizophrenia_scz2022.parquet")


def load_and_prepare(path):
    """Load PGC parquet and compute Z-scores + MAF."""
    df = pd.read_parquet(path)
    df = df.dropna(subset=["OR", "SE", "P"])
    df = df[df["SE"] > 0].copy()
    df["BETA"] = np.log(df["OR"].astype(float))
    df["Z"] = df["BETA"] / df["SE"].astype(float)
    df = df[np.isfinite(df["Z"])].copy()

    frq_cols = [c for c in df.columns if "FRQ" in c]
    df["MAF"] = np.nan
    for col in frq_cols:
        mask = df["MAF"].isna() & df[col].notna()
        vals = df.loc[mask, col].astype(float)
        df.loc[mask, "MAF"] = np.where(vals > 0.5, 1 - vals, vals)
    df = df[df["MAF"].between(0.001, 0.499)].copy()
    return df


def main():
    t0 = time.time()
    print("=" * 70)
    print("CARS-SDA v3.0 + Jin-Cai Empirical Null · PGC Schizophrenia")
    print("  Bivariate density-ratio CARS statistic with 2D FFT-KDE")
    print("=" * 70)

    # Load
    print(f"\n[1] Loading {DATA_PATH}...")
    df = load_and_prepare(DATA_PATH)
    Z, S, P = df["Z"].values, df["MAF"].values, df["P"].values
    m = len(Z)
    print(f"    {m:,} variants ready")

    # Empirical null
    print("\n[2] Estimating empirical null (Jin-Cai 2007)...")
    mu0, sigma0, pi0 = jin_cai_empirical_null(Z)
    lambda_gc = np.median(Z**2) / 0.4549
    print(f"    μ₀={mu0:.4f}, σ₀={sigma0:.4f}, π₀={pi0:.4f}, λ_GC={lambda_gc:.4f}")

    # BH baselines
    print("\n[3] Benchmarks...")
    rej_bh_raw, _, _, _ = multipletests(P, alpha=ALPHA, method="fdr_bh")
    Z_gc = (Z - mu0) / sigma0
    P_gc = 2 * (1 - stats.norm.cdf(np.abs(Z_gc)))
    rej_bh_gc, _, _, _ = multipletests(P_gc, alpha=ALPHA, method="fdr_bh")
    print(f"    BH (raw):          {np.sum(rej_bh_raw):>10,}")
    print(f"    BH (GC-corrected): {np.sum(rej_bh_gc):>10,}")

    # Adaptive-Z
    print("\n[4] Adaptive-Z...")
    rej_az, _, _ = adaptive_z(Z, alpha=ALPHA)
    print(f"    Adaptive-Z:        {np.sum(rej_az):>10,}")

    # CARS-SDA (bivariate)
    print("\n[5] CARS-SDA (bivariate density ratio)...")
    rej_cs, cars_stat, thr, _, _, diag = cars_sda(Z, S, alpha=ALPHA, mu0=mu0, sigma0=sigma0)
    print(f"    CARS-SDA:          {np.sum(rej_cs):>10,}")

    # Summary
    excl = rej_cs & ~rej_bh_gc
    print(f"\n{'=' * 70}")
    print(f"  BH (raw, inflated):        {np.sum(rej_bh_raw):>10,}  ⚠️")
    print(f"  BH (GC-corrected):         {np.sum(rej_bh_gc):>10,}  ✅ baseline")
    print(f"  Adaptive-Z:                {np.sum(rej_az):>10,}  (+{(np.sum(rej_az)/np.sum(rej_bh_gc)-1)*100:.1f}%)")
    print(f"  CARS-SDA:                  {np.sum(rej_cs):>10,}  (+{(np.sum(rej_cs)/np.sum(rej_bh_gc)-1)*100:.1f}%)")
    print(f"  CARS-exclusive:            {np.sum(excl):>10,}")
    print(f"{'=' * 70}")
    print(f"  Total time: {time.time()-t0:.1f}s")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame({
        "SNP": df["SNP"].values, "CHR": df["CHR"].values, "BP": df["BP"].values,
        "Z": Z, "P": P, "MAF": S, "CARS_stat": cars_stat,
        "BH_raw": rej_bh_raw, "BH_GC": rej_bh_gc, "AdaptZ": rej_az,
        "CARS": rej_cs, "CARS_exclusive": excl,
    })
    out_path = os.path.join(out_dir, 'cars_sda_results.parquet')
    out.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
