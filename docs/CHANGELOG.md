# Changelog

All notable changes to PySingleCellNet should be listed here. The definition of 'notable' is dynamic.

## [Unreleased]

# TO FIX

tl.score_gene_sets should add scores, one matrix to .obsm and not as columsn to .obs

### Changed
- tl.cluster_subclusters layer should be None instead of counts 

### Fixed

- tl.cluster_subclusters calls sc.pp.pca with mask_var instead of use_highly_variable


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