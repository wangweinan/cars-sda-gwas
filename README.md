# CARS-SDA v3.0 — Bivariate Covariate-Adaptive FDR Control for GWAS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

> **Result**: 1,745 independent loci (+14.3% over BH-GC baseline), with **371 net new loci** in PGC-Schizophrenia GWAS (37.6M variants, 86 seconds).

⚠️ **Variant-level counts (218K) are inflated ~127× by LD. We report locus-level numbers throughout.**

## The CARS Statistic

CARS-SDA implements the bivariate density-ratio statistic from [Cai, Sun & Wang (2019)](https://doi.org/10.1111/rssb.12298):

```
              f₀(Z) · (|T_τ|/m) · f*(S | T_τ)
CARS(i) = ─────────────────────────────────────
               correction · f(Z, S)
```

| Component | Description | Estimation |
|---|---|---|
| f₀(Z) | Empirical null density | Jin-Cai N(μ₀, σ₀²) |
| T_τ | Null-like screening set | {i : lfdr_marginal ≥ 0.9} |
| f*(S\|T_τ) | Covariate density among nulls | 1D FFT-KDE on MAF[T_τ] |
| f(Z, S) | **Joint bivariate density** | **2D FFT-KDE** (512×512) |
| correction | P(lfdr ≥ τ \| H₀) | Monte Carlo (50K draws) |

## Quick Start

```bash
git clone https://github.com/wangweinan/cars-sda-gwas.git
cd cars-sda-gwas
pip install -r requirements.txt
```

```python
from src.cars_sda import cars_sda

Z = beta / se              # z-scores from GWAS
S = minor_allele_freq      # auxiliary covariate

rejections, cars_stat, threshold, mu0, sigma0, diagnostics = cars_sda(Z, S, alpha=0.05)
print(f"Variant-level discoveries: {rejections.sum():,}")
# IMPORTANT: LD-clump before reporting locus-level counts
```

## PGC Schizophrenia Results (Locus-Level)

Applied to PGC3 (Trubetskoy et al., *Nature* 2022): 76,755 cases + 243,649 controls.

| Method | Variants | Independent Loci (500kb) | Locus Gain |
|---|---|---|---|
| GWAS-sig (p<5e-8) | 43,544 | 284 | — |
| BH (GC-corrected) | 168,638 | 1,527 | Baseline |
| Adaptive-Z | 172,749 | 1,562 | +2.3% |
| **CARS-SDA v3.0** | 218,476 | **1,745** | **+14.3%** |
| → Net new loci | — | **371** | Novel |

### Net new loci characteristics
- **P-value range**: [4.1e-5, 7.3e-4] — suggestive significance
- **|Z| range**: [3.39, 4.09]
- **Mean MAF**: 0.360 (enriched for common variants)
- Distributed across all 22 autosomes + X

### ⚠️ Theoretical Limitations
- **CARS assumes independent tests**. GWAS LD violates this (r² > 0.2 within ~500kb).
- FDR is controlled at the variant level; locus-level FDR requires post-hoc clumping.
- Our 500kb distance-based clumping is a conservative proxy for proper LD-based clumping (PLINK --clump with 1000G reference panel).
- The 371 net new loci require independent replication (e.g., SCHEMA rare variant data).

## Validation

```bash
python scripts/validate.py     # Full diagnostic suite
python scripts/simulate.py     # Quick simulation validation
```

### Simulation Results (m=1M, non-Gaussian alternative)

| Method | Rejections | FDR | Power |
|---|---|---|---|
| BH (raw) | 9,832 | 27.0% ✗ | 56.4% |
| BH (GC) | 3,753 | 4.9% ✓ | 28.1% |
| Adaptive-Z | 3,781 | 4.9% ✓ | 28.3% |
| **CARS-SDA** | **3,994** | **5.3%** ✓ | **29.8%** |

## Architecture

```
cars-sda-gwas/
├── src/
│   └── cars_sda.py         # Core engine: CARS + Jin-Cai + FFT-KDE
├── scripts/
│   ├── simulate.py          # Quick simulation validation
│   ├── validate.py          # Comprehensive diagnostic suite
│   └── run_analysis.py      # Full PGC-SCZ pipeline
├── results/
│   └── cars_net_new_loci.csv  # 371 net new loci details
├── figures/                  # Diagnostic & comparison plots
└── docs/                    # GitHub Pages site
```

## References

1. **CARS**: Cai, T.T., Sun, W., & Wang, W. (2019). Covariate Assisted Ranking and Screening for Large-Scale Two-Sample Inference. *JRSS-B*, 81(2), 187-234.
2. **SDA**: Du, L., et al. (2023). Step-Down Adaptive procedure. *Biometrika*.
3. **Jin-Cai**: Jin, J. & Cai, T.T. (2007). Estimating the null and the proportion of nonnull effects. *JASA*, 102, 495-506.
4. **PGC3**: Trubetskoy, V., et al. (2022). Mapping genomic loci implicates genes and synaptic biology in schizophrenia. *Nature*, 604, 502-508.

## License

MIT License. See [LICENSE](LICENSE).
