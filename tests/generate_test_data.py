#!/usr/bin/env python
"""Generate small test fixtures from the full data files in data/.

Run once from the repo root:
    python tests/generate_test_data.py

Creates three small .h5ad files in tests/data/:
  - pbmc_ref_small.h5ad      (~320 cells, 2000 HVGs)
  - pbmc_query_small.h5ad    (~500 cells, common genes)
  - pijuan_small_test.h5ad   (~varies, 2000 HVGs)
"""

import os
import sys
import numpy as np
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def _stratified_sample(adata, groupby, n_per_group, seed=42):
    """Sample up to n_per_group cells per group, stratified by groupby."""
    rng = np.random.default_rng(seed)
    indices = []
    for group in adata.obs[groupby].unique():
        group_idx = np.where(adata.obs[groupby] == group)[0]
        n = min(len(group_idx), n_per_group)
        chosen = rng.choice(group_idx, size=n, replace=False)
        indices.extend(chosen)
    return adata[sorted(indices)].copy()


def _top_hvg(adata, n_top=2000):
    """Select top n highly variable genes."""
    ad = adata.copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor="seurat_v3")
    return ad[:, ad.var["highly_variable"]].copy()


def generate_pbmc_ref(n_per_type=40, n_hvg=2000):
    path = os.path.join(DATA_DIR, "adPBMC_ref_040623.h5ad")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    print(f"  Loading {path}")
    ad = sc.read_h5ad(path)
    ad = _stratified_sample(ad, "cell_type", n_per_type)
    ad = _top_hvg(ad, n_hvg)
    out = os.path.join(OUT_DIR, "pbmc_ref_small.h5ad")
    ad.write_h5ad(out)
    print(f"  Wrote {out} ({ad.shape})")
    return ad


def generate_pbmc_query(ref_ad=None, n_cells=500):
    path = os.path.join(DATA_DIR, "adPBMC_query_1_20k_HT_040723.h5ad")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    print(f"  Loading {path}")
    ad = sc.read_h5ad(path)
    rng = np.random.default_rng(42)
    idx = rng.choice(ad.n_obs, size=min(n_cells, ad.n_obs), replace=False)
    ad = ad[sorted(idx)].copy()
    # Subset to common genes if ref is available
    if ref_ad is not None:
        common = list(set(ad.var_names) & set(ref_ad.var_names))
        ad = ad[:, common].copy()
    out = os.path.join(OUT_DIR, "pbmc_query_small.h5ad")
    ad.write_h5ad(out)
    print(f"  Wrote {out} ({ad.shape})")
    return ad


def generate_pijuan(n_per_type=30, n_hvg=2000):
    path = os.path.join(DATA_DIR, "adPijuan_small.h5ad")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    print(f"  Loading {path}")
    ad = sc.read_h5ad(path)
    groupby = "ct1" if "ct1" in ad.obs.columns else ad.obs.columns[0]
    ad = _stratified_sample(ad, groupby, n_per_type)
    ad = _top_hvg(ad, n_hvg)
    out = os.path.join(OUT_DIR, "pijuan_small_test.h5ad")
    ad.write_h5ad(out)
    print(f"  Wrote {out} ({ad.shape})")
    return ad


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating test fixtures...")

    ref = generate_pbmc_ref()
    generate_pbmc_query(ref)
    generate_pijuan()

    print("Done.")


if __name__ == "__main__":
    main()
