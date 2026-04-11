"""Generate publication-quality figures for the CARS-SDA paper."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def fig1_power_comparison():
    """Bar chart: discoveries by method on PGC-SCZ (37.6M variants)."""
    methods = ['BH\n(raw)', 'BH\n(GC-corr)', 'Adaptive-Z\n(emp null)', 'CARS-SDA\n(emp null)']
    counts = [334061, 168638, 170224, 251043]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#6c5ce7']
    fdr_ok = [False, True, True, True]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, [c / 1000 for c in counts], color=colors,
                  edgecolor='white', linewidth=1.5, width=0.65)

    for bar, count, ok in zip(bars, counts, fdr_ok):
        y = bar.get_height()
        label = f'{count:,}'
        ax.text(bar.get_x() + bar.get_width()/2, y + 5, label,
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        if not ok:
            ax.text(bar.get_x() + bar.get_width()/2, y - 15,
                    '⚠ inflated FDR', ha='center', va='top', fontsize=8,
                    color='white', fontweight='bold')

    # CARS-SDA vs BH annotation
    ax.annotate('', xy=(3, counts[3]/1000), xytext=(1, counts[1]/1000),
                arrowprops=dict(arrowstyle='->', color='#e17055', lw=2))
    mid_x = 2; mid_y = (counts[1] + counts[3]) / 2000
    ax.text(mid_x, mid_y + 15, '+48.9%\npower gain',
            ha='center', fontsize=11, fontweight='bold', color='#e17055')

    ax.set_ylabel('Discoveries (thousands)', fontsize=12)
    ax.set_title('FDR-Controlled Discoveries · PGC Schizophrenia (37.6M variants)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 400)
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    fig.savefig('../figures/power_comparison.png')
    print('  Saved figures/power_comparison.png')
    plt.close()


def fig2_simulation_validation():
    """Simulation results: FDR and power for each method."""
    methods = ['BH (raw)', 'BH (GC)', 'Adaptive-Z', 'CARS-SDA']
    fdr =   [21.78, 3.17, 3.5, 2.89]
    power = [63.61, 36.55, 37.0, 39.67]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#6c5ce7']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # FDR panel
    bars1 = ax1.bar(methods, fdr, color=colors, edgecolor='white', linewidth=1.5)
    ax1.axhline(y=5, color='#e74c3c', linestyle='--', linewidth=1.5, label='Nominal α = 0.05')
    ax1.set_ylabel('Empirical FDR (%)')
    ax1.set_title('A. FDR Control on Simulation', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 25)
    for bar, v in zip(bars1, fdr):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%',
                 ha='center', fontsize=9, fontweight='bold')

    # Power panel
    bars2 = ax2.bar(methods, power, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Power (%)')
    ax2.set_title('B. Statistical Power', fontweight='bold')
    ax2.set_ylim(0, 75)
    for bar, v in zip(bars2, power):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%',
                 ha='center', fontsize=9, fontweight='bold')

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Simulation Validation (m = 1M, σ₀ = 1.16, covariate-dependent sparsity)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('../figures/simulation_validation.png')
    print('  Saved figures/simulation_validation.png')
    plt.close()


def fig3_pipeline_diagram():
    """Pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis('off')

    boxes = [
        (1, 3, 2.2, 1.5, 'Input\nZ-scores\n+ MAF', '#dfe6e9'),
        (4, 4, 2.2, 1.2, 'Jin-Cai\nEmpirical Null\n(μ₀, σ₀)', '#74b9ff'),
        (4, 1.5, 2.2, 1.2, 'K-Fold\nCross-Fitting', '#a29bfe'),
        (7.5, 4, 2.5, 1.2, 'Per-Bin EM\nf = π₀N(μ₀,σ₀²)\n+ (1-π₀)N(μ₁,σ₁²)', '#fd79a8'),
        (7.5, 1.5, 2.5, 1.2, 'Density-Ratio\nlfdr(z|s) =\nπ₀f₀/f_mix', '#ffeaa7'),
        (11, 3, 2.2, 1.5, 'Step-Up\nRejections\n(FDR ≤ α)', '#55efc4'),
    ]

    for x, y, w, h, txt, c in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=c, edgecolor='#2d3436',
                              linewidth=1.5, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)

    # Arrows
    arrows = [(3.2, 3.75, 4, 4.5), (3.2, 3.75, 4, 2.1),
              (6.2, 4.6, 7.5, 4.6), (6.2, 2.1, 7.5, 2.1),
              (10, 4.6, 11, 4), (10, 2.1, 11, 3.2)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2d3436', lw=2))

    ax.set_title('CARS-SDA Pipeline with Jin-Cai Empirical Null Estimation',
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig('../figures/pipeline_diagram.png')
    print('  Saved figures/pipeline_diagram.png')
    plt.close()


def fig4_gene_network():
    """Biological network diagram of CARS-exclusive discoveries."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis('off')

    # Network nodes (pathway → genes)
    networks = {
        'Glutamatergic\nSynapse': {
            'pos': (3, 8), 'color': '#0984e3',
            'genes': ['GRIN2A*', 'CNIH3', 'ELFN1*', 'SYT2', 'CALN1']
        },
        'Ion Channels\n& Excitability': {
            'pos': (10, 8), 'color': '#6c5ce7',
            'genes': ['CACNA1I*', 'KCNB1', 'KCNJ15', 'SLC8A1']
        },
        'Cell Adhesion\n& Wiring': {
            'pos': (3, 5), 'color': '#e17055',
            'genes': ['PCDHA7*', 'CTNNA2', 'PTK7', 'BRINP1', 'NELL1']
        },
        'Mitochondrial\nBioenergetics': {
            'pos': (10, 5), 'color': '#d63031',
            'genes': ['ALAS1', 'DLST', 'NDUFAF2']
        },
        'Vesicular\nTrafficking': {
            'pos': (3, 2), 'color': '#00b894',
            'genes': ['TSNARE1*', 'TOM1L2', 'NUP88', 'GULP1']
        },
        'Transcription\n& Chromatin': {
            'pos': (10, 2), 'color': '#fdcb6e',
            'genes': ['SP4*', 'CHD2*', 'SREBF2', 'FANCA']
        },
    }

    for name, info in networks.items():
        x, y = info['pos']
        # Hub circle
        circle = plt.Circle((x, y), 0.9, facecolor=info['color'], alpha=0.2,
                            edgecolor=info['color'], linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=10,
                fontweight='bold', color=info['color'], zorder=3)
        # Gene labels
        for i, gene in enumerate(info['genes']):
            angle = -90 + i * (180 / max(len(info['genes']) - 1, 1))
            rad = np.radians(angle)
            gx = x + 1.6 * np.cos(rad)
            gy = y + 1.3 * np.sin(rad)
            style = 'bold' if '*' in gene else 'normal'
            color = '#c0392b' if '*' in gene else '#2d3436'
            ax.text(gx, gy, gene.replace('*',''), ha='center', va='center',
                    fontsize=8, fontweight=style, color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor=info['color'], alpha=0.8), zorder=4)
            ax.plot([x, gx], [y, gy], color=info['color'], alpha=0.3, linewidth=1, zorder=1)

    # Cross-network connections
    connections = [
        ((3, 8), (10, 8), '#636e72'),  # Glutamate ↔ Channels
        ((3, 8), (3, 5), '#636e72'),   # Glutamate ↔ Adhesion
        ((3, 8), (3, 2), '#636e72'),   # Glutamate ↔ Trafficking
        ((10, 5), (3, 8), '#636e72'),  # Mito ↔ Glutamate
        ((10, 2), (3, 5), '#636e72'),  # TF ↔ Adhesion
    ]
    for (x1, y1), (x2, y2), c in connections:
        ax.plot([x1, x2], [y1, y2], color=c, alpha=0.15, linewidth=2,
                linestyle='--', zorder=0)

    ax.text(7, 9.5, '* = PGC3/SCHEMA confirmed (positive control)',
            ha='center', fontsize=9, style='italic', color='#c0392b')
    ax.set_title('CARS-Exclusive Gene Discovery Network · 6 Biological Pathways',
                 fontsize=14, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig('../figures/gene_network.png')
    print('  Saved figures/gene_network.png')
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('../figures', exist_ok=True)
    fig1_power_comparison()
    fig2_simulation_validation()
    fig3_pipeline_diagram()
    fig4_gene_network()
    print('\nAll figures generated.')
