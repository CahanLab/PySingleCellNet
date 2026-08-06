# Changelog

All notable changes to PySingleCellNet should be listed here. The definition of 'notable' is dynamic.

## [Unreleased]

# TO FIX

tl.score_gene_sets should add scores, one matrix to .obsm and not as columsn to .obs

### Added
- `tl.suggest_pcs_to_drop`: automatically identify PCs driven by a confounder gene list (e.g., cell-cycle genes) using FDR, correlation, and impact-score thresholds
- `ut.remove_confounder_pcs`: convenience wrapper that flags and removes confounder PCs in one call
- Confounder PC removal tutorial notebook (`docs/notebooks/confounder_pcs.ipynb`)
- `requirements.txt` and `environment.yml` for reproducible environment setup
- Installation instructions in `docs/install.md` for pip and conda workflows
- Test suite (`tests/`) covering classifier, clustering, categorize, utils, and plotting
- `tests/generate_test_data.py` to create small h5ad fixtures from full datasets
- `CLAUDE.md` for AI-assisted development guidelines
- Gene modules tutorial notebook (`docs/notebooks/gene_modules.ipynb`)
- Exported `tl.subset_modules_top_genes` (was defined but not in `__init__.py`)

### Changed
- `tl.correlate_module_scores_with_pcs`: now returns `fdr` (Benjamini-Hochberg adjusted p-values) and `impact_score` (correlation² × variance_ratio) columns
- tl.cluster_subclusters layer should be None instead of counts
- Completed Google-style docstrings (Args/Returns/Raises) across all public functions for mkdocs auto-extraction
- `tl.find_gene_modules`: `groupby` is now optional (`None` by default); only required when `mean_cluster=True` (defaults to `'leiden'` in that case)

### Fixed

- tl.cluster_subclusters calls sc.pp.pca with mask_var instead of use_highly_variable
- `tl.train_classifier`: `n_rand` auto-calculation now returns `int` instead of `np.float64` (fixes pandas iloc error)
- `tl.comp_ct_thresh`: handle cell types with no classified cells (empty array in `np.quantile`)
- `pl.bar_classifier_f1`: fall back to `SCN_class_argmax_colors` when `SCN_class_colors` is missing

### Planned
- PCA-loading-based gene module discovery (extract per-PC gene modules from PCA loadings)


## [0.1.3] - 2025-09-15

### Changed

- replace cl with tl
- moved functions to more fitting files, like unused ones to utils.misc.py
- do not export unused functions

### Added

- tl.discover_cell_cliques labels cells by consensus cluster labels, kind of
- tl.clustering_quality_vs_nn_summary computes metrics of clustering quality
- tl.cluster_alot
- tl.cluster_subcluster
- resurrected gene clustering functions 
- notebook tutorial on cell clustering

### Fixed

- `filter_anndata_slots` to handle .uns and dependencies across slots
- `classify_anndata` bug that prevented writing h5ad. see https://github.com/CahanLab/PySingleCellNet/issues/13

### Removed

- ut.mito_rib_heme
- lots of stuff that is old or has been moved to other packages like STUF

## [0.1.2] - 2025-08-05

### Removed

- stale functions in old/

## [0.1.1] - 2025-08-04

### Fixed

- Wrestled pyproject.toml into shape

### Removed

- stale functions in old/

## [0.1.0] - 2025-08-04

### Added

- new versioning system

### Fixed

- typos in docs