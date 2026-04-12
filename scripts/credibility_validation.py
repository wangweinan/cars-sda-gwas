#!/usr/bin/env python3
"""
Credibility Validation Pipeline for 371 Net New CARS-SDA Loci

Three orthogonal validation strategies:
1. FUMA input preparation (for manual upload)
2. GTEx brain eQTL lookup via API
3. PLINK-format LD reference preparation

Usage:
    python scripts/credibility_validation.py
"""

import os, sys, time, json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
NET_NEW = ROOT / 'results' / 'cars_net_new_loci.csv'
RESULTS = ROOT / 'results' / 'cars_sda_results.parquet'
CRED_DIR = ROOT / 'results' / 'credibility'
CRED_DIR.mkdir(exist_ok=True)

def prepare_fuma_input():
    """
    Prepare FUMA-compatible input file from net new loci.
    FUMA expects: SNP, CHR, BP, P, beta/z-score
    Also create a full GWAS summary stats file for FUMA SNP2GENE.
    """
    print('\n' + '='*60)
    print('STEP 1: Preparing FUMA Input Files')
    print('='*60)
    
    net_new = pd.read_csv(NET_NEW)
    
    # FUMA SNP2GENE format (full GWAS summary stats)
    # Needs: SNP, CHR, BP, P, and optionally BETA/Z/OR
    res = pd.read_parquet(RESULTS)
    
    # Format for FUMA: only keep one entry per SNP (de-duplicate)
    # Group by SNP and keep the entry with lowest P
    res_dedup = res.sort_values('P').drop_duplicates(subset='SNP', keep='first')
    
    fuma_full = res_dedup[['SNP', 'CHR', 'BP', 'P', 'Z']].copy()
    fuma_full.columns = ['SNP', 'CHR', 'BP', 'P', 'Z']
    fuma_full['CHR'] = fuma_full['CHR'].astype(str)
    
    # Save full GWAS summary stats for FUMA
    fuma_path = CRED_DIR / 'fuma_input_full_gwas.txt.gz'
    fuma_full.to_csv(fuma_path, sep='\t', index=False, compression='gzip')
    print(f'  Full GWAS input: {fuma_path} ({len(fuma_full):,} SNPs)')
    
    # Also save a candidate SNP list (just the 371 lead SNPs)
    fuma_candidates = net_new[['SNP', 'CHR', 'BP', 'P', 'Z']].copy()
    fuma_candidates.to_csv(CRED_DIR / 'fuma_candidate_snps.txt', sep='\t', index=False)
    print(f'  Candidate SNPs: {CRED_DIR / "fuma_candidate_snps.txt"} (371 SNPs)')
    
    # FUMA gene-set file (for GENE2FUNC)
    # List of genes near net new loci
    genes = pd.read_csv(ROOT.parent / 'cars_exclusive_ALL_genes.csv')
    unique_genes = genes['gene'].unique()
    with open(CRED_DIR / 'fuma_gene_list.txt', 'w') as f:
        for g in sorted(unique_genes):
            f.write(f'{g}\n')
    print(f'  Gene list: {CRED_DIR / "fuma_gene_list.txt"} ({len(unique_genes)} genes)')
    
    print('\n  📋 Next Steps for FUMA:')
    print('    1. Go to https://fuma.ctglab.nl/')
    print('    2. Create account / Login')
    print('    3. SNP2GENE → Upload fuma_input_full_gwas.txt.gz')
    print('       - Set GWAS summary statistic columns')
    print('       - Use Pre-defined lead SNPs: upload fuma_candidate_snps.txt')
    print('    4. GENE2FUNC → Paste genes from fuma_gene_list.txt')
    print('    5. Download results (enrichment, eQTL mapping, etc.)')
    

def query_gtex_eqtls():
    """
    Query GTEx v8 API for brain eQTLs at each of the 371 net new lead SNPs.
    Uses the GTEx REST API to check if variants are significant eQTLs 
    in any of 13 brain tissues.
    """
    print('\n' + '='*60)
    print('STEP 2: GTEx Brain eQTL Lookup')  
    print('='*60)
    
    import requests
    
    net_new = pd.read_csv(NET_NEW)
    
    # GTEx brain tissues
    brain_tissues = [
        'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24',
        'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere',
        'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
        'Brain_Hippocampus', 'Brain_Hypothalamus',
        'Brain_Nucleus_accumbens_basal_ganglia',
        'Brain_Putamen_basal_ganglia',
        'Brain_Spinal_cord_cervical_c-1',
        'Brain_Substantia_nigra'
    ]
    
    # GTEx API base URL (v2)
    BASE_URL = 'https://gtexportal.org/api/v2'
    
    results = []
    errors = 0
    
    print(f'  Querying {len(net_new)} SNPs against {len(brain_tissues)} brain tissues...')
    print(f'  (This may take 10-15 minutes due to API rate limits)')
    
    for i, row in net_new.iterrows():
        snp = row['SNP']
        chrom = int(row['CHR'])
        bp = int(row['BP'])
        
        if i > 0 and i % 50 == 0:
            print(f'  Progress: {i}/{len(net_new)} SNPs ({100*i/len(net_new):.0f}%)')
        
        # GTEx uses variant_id format: chr{chr}_{pos}_{ref}_{alt}_b38
        # But we can also query by rsid using the eQTL endpoint
        try:
            # Try the eQTL by SNP endpoint
            url = f'{BASE_URL}/association/singleTissueEqtl'
            params = {
                'snpId': snp,
                'datasetId': 'gtex_v8',
                'tissueSiteDetailId': 'Brain_Cortex'  # Start with cortex
            }
            
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if data and 'data' in data and len(data['data']) > 0:
                    for eqtl in data['data']:
                        results.append({
                            'SNP': snp,
                            'CHR': chrom,
                            'BP': bp,
                            'P_gwas': row['P'],
                            'Gene_eQTL': eqtl.get('geneSymbol', ''),
                            'Gene_id': eqtl.get('geneSymbolUpper', eqtl.get('gencodeId', '')),
                            'Tissue': eqtl.get('tissueSiteDetailId', ''),
                            'P_eqtl': eqtl.get('pValue', ''),
                            'NES': eqtl.get('nes', ''),
                        })
            elif resp.status_code == 429:
                # Rate limited - wait and retry
                time.sleep(2)
                continue
            else:
                errors += 1
                
            # Be nice to the API
            time.sleep(0.3)
            
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f'  Warning: {snp} - {str(e)[:50]}')
            elif errors == 5:
                print(f'  (suppressing further warnings...)')
            time.sleep(1)
    
    # Also try a broader query with multiple brain tissues for significant hits
    print(f'\n  Phase 1 complete: {len(results)} eQTL hits from Brain_Cortex')
    
    if len(results) > 0:
        # For SNPs with cortex hits, also check other brain tissues 
        cortex_snps = set(r['SNP'] for r in results)
        print(f'  Checking {len(cortex_snps)} cortex-positive SNPs across all brain tissues...')
        
        for snp in cortex_snps:
            for tissue in brain_tissues:
                if tissue == 'Brain_Cortex':
                    continue
                try:
                    url = f'{BASE_URL}/association/singleTissueEqtl'
                    params = {'snpId': snp, 'datasetId': 'gtex_v8', 'tissueSiteDetailId': tissue}
                    resp = requests.get(url, params=params, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data and 'data' in data:
                            for eqtl in data['data']:
                                results.append({
                                    'SNP': snp,
                                    'Gene_eQTL': eqtl.get('geneSymbol', ''),
                                    'Tissue': tissue,
                                    'P_eqtl': eqtl.get('pValue', ''),
                                    'NES': eqtl.get('nes', ''),
                                })
                    time.sleep(0.3)
                except:
                    pass
    
    # Save results
    if results:
        eqtl_df = pd.DataFrame(results)
        eqtl_df.to_csv(CRED_DIR / 'gtex_brain_eqtls.csv', index=False)
        
        n_snps_with_eqtl = eqtl_df['SNP'].nunique()
        n_genes = eqtl_df['Gene_eQTL'].nunique()
        n_tissues = eqtl_df['Tissue'].nunique()
        
        print(f'\n  ✅ GTEx Results:')
        print(f'    {n_snps_with_eqtl}/{len(net_new)} SNPs are brain eQTLs ({100*n_snps_with_eqtl/len(net_new):.1f}%)')
        print(f'    {n_genes} target genes')
        print(f'    {n_tissues} brain tissues')
        print(f'    Saved: {CRED_DIR / "gtex_brain_eqtls.csv"}')
    else:
        print(f'\n  ⚠️ No eQTL hits found (API may have changed or rate limited)')
        print(f'    Alternative: Download bulk GTEx files from gtexportal.org/home/datasets')
    
    print(f'  Errors: {errors}')
    return results


def prepare_cross_trait():
    """
    Cross-trait validation: check net new loci against other psychiatric GWAS
    using the GWAS Catalog API.
    """
    print('\n' + '='*60)
    print('STEP 3: Cross-Trait GWAS Catalog Lookup')
    print('='*60)
    
    import requests
    
    net_new = pd.read_csv(NET_NEW)
    
    # GWAS Catalog API
    BASE_URL = 'https://www.ebi.ac.uk/gwas/rest/api'
    
    # Psychiatric traits to check
    traits = {
        'EFO_0000692': 'Schizophrenia',
        'EFO_0000289': 'Bipolar disorder',
        'EFO_0003761': 'Major depressive disorder',
        'EFO_0003758': 'Autism spectrum disorder',
        'EFO_0004859': 'Attention deficit hyperactivity disorder',
        'EFO_0004785': 'Anxiety',
    }
    
    results = []
    
    print(f'  Querying {len(net_new)} SNPs against GWAS Catalog...')
    
    for i, row in net_new.iterrows():
        snp = row['SNP']
        
        if i > 0 and i % 100 == 0:
            print(f'  Progress: {i}/{len(net_new)} ({100*i/len(net_new):.0f}%)')
        
        # Skip non-rs IDs
        if not snp.startswith('rs'):
            continue
            
        try:
            url = f'{BASE_URL}/singleNucleotidePolymorphisms/{snp}/associations'
            resp = requests.get(url, 
                              headers={'Accept': 'application/json'},
                              timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if '_embedded' in data and 'associations' in data['_embedded']:
                    for assoc in data['_embedded']['associations']:
                        p_val = assoc.get('pvalue', None)
                        # Get trait
                        trait_names = []
                        if 'efoTraits' in assoc:
                            trait_names = [t.get('trait', '') for t in assoc['efoTraits']]
                        
                        # Check if it's a psychiatric trait
                        for trait_name in trait_names:
                            trait_lower = trait_name.lower()
                            is_psych = any(x in trait_lower for x in [
                                'schizo', 'bipolar', 'depress', 'autism', 
                                'psychiatric', 'adhd', 'anxiety', 'psycho',
                                'cogniti', 'intelligence', 'brain', 'neurot'
                            ])
                            if is_psych:
                                results.append({
                                    'SNP': snp,
                                    'CHR': int(row['CHR']),
                                    'BP': int(row['BP']),
                                    'P_cars': row['P'],
                                    'Trait': trait_name,
                                    'P_catalog': p_val,
                                })
            
            time.sleep(0.2)  # Rate limit
            
        except Exception as e:
            pass
    
    if results:
        cross_df = pd.DataFrame(results)
        cross_df.to_csv(CRED_DIR / 'cross_trait_associations.csv', index=False)
        
        n_snps = cross_df['SNP'].nunique()
        n_traits = cross_df['Trait'].nunique()
        
        print(f'\n  ✅ Cross-Trait Results:')
        print(f'    {n_snps}/{len(net_new)} SNPs have prior psychiatric associations ({100*n_snps/len(net_new):.1f}%)')
        print(f'    {n_traits} unique psychiatric traits')
        print(f'    Saved: {CRED_DIR / "cross_trait_associations.csv"}')
        
        # Show top hits
        print(f'\n  Top cross-trait associations:')
        for trait in cross_df['Trait'].value_counts().head(10).index:
            count = cross_df[cross_df['Trait'] == trait]['SNP'].nunique()
            print(f'    {trait}: {count} SNPs')
    else:
        print(f'\n  No cross-trait associations found')
    
    return results


def create_summary_report(eqtl_results, cross_results):
    """Generate final credibility summary."""
    print('\n' + '='*60)
    print('CREDIBILITY VALIDATION SUMMARY')
    print('='*60)
    
    net_new = pd.read_csv(NET_NEW)
    n_total = len(net_new)
    
    print(f'\n  Total net new loci: {n_total}')
    print(f'\n  Validation results:')
    
    if eqtl_results:
        eqtl_snps = len(set(r['SNP'] for r in eqtl_results))
        print(f'    Brain eQTLs (GTEx):    {eqtl_snps}/{n_total} ({100*eqtl_snps/n_total:.1f}%)')
    
    if cross_results:
        cross_snps = len(set(r['SNP'] for r in cross_results))
        print(f'    Cross-trait (Catalog):  {cross_snps}/{n_total} ({100*cross_snps/n_total:.1f}%)')
    
    print(f'\n  Files generated in {CRED_DIR}:')
    for f in sorted(CRED_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f'    {f.name} ({size_kb:.1f} KB)')
    
    print(f'\n  📋 FUMA Upload Instructions:')
    print(f'    1. Visit https://fuma.ctglab.nl/')
    print(f'    2. SNP2GENE → Upload: fuma_input_full_gwas.txt.gz')
    print(f'    3. Under "Pre-defined lead SNPs", upload: fuma_candidate_snps.txt')
    print(f'    4. Set parameters: LD r²=0.6, population=EUR, p-threshold < 1e-3')
    print(f'    5. GENE2FUNC → Paste contents of: fuma_gene_list.txt')


if __name__ == '__main__':
    print('CARS-SDA Credibility Validation Pipeline')
    print(f'Working directory: {ROOT}')
    
    # Step 1: Always prepare FUMA input
    prepare_fuma_input()
    
    # Step 2: GTEx eQTL lookup 
    eqtl_results = query_gtex_eqtls()
    
    # Step 3: Cross-trait GWAS catalog 
    cross_results = prepare_cross_trait()
    
    # Summary
    create_summary_report(eqtl_results, cross_results)
