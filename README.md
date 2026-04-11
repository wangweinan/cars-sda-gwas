# CARS-SDA v3.0 — Bivariate Covariate-Adaptive FDR Control for GWAS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![GitHub Pages](https://img.shields.io/badge/Docs-GitHub%20Pages-green.svg)](https://wangweinan.github.io/cars-sda-gwas/)

> **Result**: 218,476 discoveries (+29.6% over BH-GC baseline) in PGC Schizophrenia GWAS with 37.6M variants, processed in **86 seconds**.

## The CARS Statistic

CARS-SDA implements the proper bivariate density-ratio statistic from [Cai, Sun & Wang (2019)](https://doi.org/10.1111/rssb.12298):

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

### Why not just Adaptive-Z?

| Approach | Density used | Power gain (PGC-SCZ) |
|---|---|---|
| Adaptive-Z | f(Z) marginal | +2.4% |
| Binned lfdr | f(Z\|bin) per MAF bin | ≈ Adaptive-Z |
| **CARS-SDA v3** | **f(Z, MAF) bivariate** | **+29.6%** |

The bivariate joint density captures how signal enrichment varies continuously with MAF — information lost by marginal or binned approaches.

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
print(f"Discoveries: {rejections.sum():,}")
print(f"Empirical null: N({mu0:.3f}, {sigma0:.3f}²)")
```

## Validation

Run the comprehensive validation suite:

```bash
python scripts/validate.py     # Full diagnostic suite (QQ, calibration, independence, multi-seed)
python scripts/simulate.py     # Quick simulation validation
```

### Simulation Results (m=1M, non-Gaussian alternative)

| Method | Rejections | FDR | Power |
|---|---|---|---|
| BH (raw) | 9,832 | 27.0% ✗ | 56.4% |
| BH (GC) | 3,753 | 4.9% ✓ | 28.1% |
| Adaptive-Z | 3,781 | 4.9% ✓ | 28.3% |
| **CARS-SDA** | **3,994** | **5.3%** ✓ | **29.8%** |

### Diagnostic Checks

1. **QQ Plot**: KS p-value = 0.96 — excellent null fit
2. **KDE vs Histogram**: Log-scale overlay confirms mixture model
3. **Z ⊥ MAF | H₀**: Pearson r = 0.0002, p = 0.88 — verified
4. **CARS Calibration**: Points track diagonal — well-calibrated bivariate lfdr

## PGC Schizophrenia Results

Applied to PGC3 (Trubetskoy et al., *Nature* 2022): 76,755 cases + 243,649 controls.

| Method | Discoveries | vs. Baseline |
|---|---|---|
| BH (raw, inflated) | 334,061 | ⚠️ |
| BH (GC-corrected) | 168,638 | Baseline |
| Adaptive-Z | 172,749 | +2.4% |
| **CARS-SDA v3.0** | **218,476** | **+29.6%** |
| CARS-exclusive | **58,213** | Novel |

**58,213 CARS-exclusive discoveries** across **1,096 unique genes**, organized into 6 coherent biological networks:
- 🧠 **Glutamatergic Synapse**: GRIN2A, GRM3, GRM4, GRM5, CNIH3, SYT2
- ⚡ **Ion Channels**: CACNA1I, KCNB1, HCN1, CACNA2D1, CACNB2
- 🔗 **Cell Adhesion**: NRXN1, NRXN3, CNTNAP2, PCDHA7, CTNNA2
- 🏭 **Mitochondria**: ALAS1, DLST, NDUFV2, MVK
- 📦 **Vesicular Trafficking**: TSNARE1, NUP88, EXOC4, GULP1
- 📋 **Transcription**: SP4, CHD2, CHD7, HDAC9, SREBF2

## Architecture

```
cars-sda-gwas/
├── src/
│   └── cars_sda.py         # Core engine: CARS + Jin-Cai + FFT-KDE
├── scripts/
│   ├── simulate.py          # Quick simulation validation
│   ├── validate.py          # Comprehensive diagnostic suite
│   └── run_analysis.py      # Full PGC-SCZ pipeline
├── docs/                     # GitHub Pages site
│   ├── index.html
│   ├── gene_data.js
│   └── assets/
└── figures/validation/       # Generated diagnostic plots
```

## Key Dependencies

- `numpy`, `scipy` — Core computation + 2D FFT
- `statsmodels` — BH procedure
- `pandas`, `pyarrow` — Data I/O

## References

1. **CARS**: Cai, T.T., Sun, W., & Wang, W. (2019). CARS: Covariate Assisted Ranking and Screening for Large-Scale Two-Sample Inference. *JRSS-B*, 81(2), 187-234.
2. **SDA**: Du, L., et al. (2023). Step-Down Adaptive procedure. *Biometrika*.
3. **Jin-Cai**: Jin, J. & Cai, T.T. (2007). Estimating the null and the proportion of nonnull effects in large-scale multiple comparisons. *JASA*, 102, 495-506.
4. **PGC3**: Trubetskoy, V., et al. (2022). Mapping genomic loci implicates genes and synaptic biology in schizophrenia. *Nature*, 604, 502-508.

## License

MIT License. See [LICENSE](LICENSE).
