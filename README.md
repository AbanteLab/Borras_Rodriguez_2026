# Code for "Accelerated biological aging in major depressive disorder primarily affects the immune system and can be counteracted by reprogramming"

This repo depends on [`proteomics_tools`](https://github.com/AbanteLab/proteomics_tools)
for all generic proteomics functions (GLM fitting, delta-method contrasts, GSEA,
volcano plots, PCA).  Everything study-specific (data paths, contrasts, MOFA
integration, figure generation) lives here.

> **Status:** Code associated with a manuscript currently under revision.

---

## Repository layout

```
borras_rodriguez_2026/
├── pyproject.toml
├── README.md
└── src/borras_rodriguez_2026/
    ├── config.py                  # ALL study-specific parameters & paths
    ├── utils.py                   # Paper-specific helpers (data parsing, MOFA)
    ├── pathway_classes.py         # Keyword → biological class mapping
    │
    ├── preprocess_hp_adrenal.py   # Stage 1a — hippocampus & adrenal preprocessing
    ├── preprocess_spleen.py       # Stage 1b — spleen preprocessing
    ├── run_glm_gsea.py            # Stage 2  — per-tissue GLM + contrasts + GSEA
    ├── run_mofa.py                # Stage 3  — MOFA integration + factor GSEA
    │
    ├── results_paper_fig_4.py     # Stage 4a — generate Figure 4
    ├── results_paper_fig_5.py     # Stage 4b — generate Figure 5
    ├── utils_figures.py           # Paper-specific figure helpers
    │
    └── pipeline.py                # Orchestrator — runs all stages in order
```

---

## Output directory layout

```
<DATA_DIR>/processed_data/
├── hipocamp/
│   ├── log2_hipocamp_long_format_processed.csv
│   ├── log2_hipocamp_long_format_processed_sex_regout.csv
│   └── sample_nan_counts_hipocamp.csv
├── adrenal/
│   ├── log2_adrenal_long_format_processed.csv
│   ├── log2_adrenal_long_format_processed_sex_regout.csv
│   └── sample_nan_counts_adrenal.csv
└── spleen/
    ├── spleen_long_format_processed.csv
    └── spleen_long_format_processed_sex_regout.csv

<RESULTS_DIR>/
├── sample_info.csv
├── glm_results/
│   ├── hipocamp/
│   │   ├── hipocamp_differential_expression_results.csv
│   │   ├── hipocamp_glm_contrasts_per_protein_with_fdr.csv
│   │   ├── hipocamp_explained_variance.csv
│   │   ├── hipocamp_normality_results.csv
│   │   ├── hipocamp_explained_variance_histogram.png
│   │   ├── hipocamp_shapiro_W_histogram.png
│   │   ├── glm/volcano_plots/
│   │   └── contrasts/volcano_plots/
│   ├── adrenal/   (same structure)
│   └── spleen/    (same structure)
├── gsea_results/
│   ├── hipocamp/
│   │   ├── glm_coefficients/<covariate>/   (one dir per GLM covariate)
│   │   └── contrasts/<contrast>/           (one dir per contrast)
│   ├── adrenal/   (same structure)
│   └── spleen/    (same structure)
├── mofa/
│   ├── mofa_model_K11_seed0_Rdecay0.01_varexpl0.9.hdf5
│   ├── df_all_for_mofa.csv
│   ├── factors_matrix.csv / .xlsx
│   ├── factor_weights_per_feature.csv / .xlsx
│   ├── variance_explained_by_factor_and_view.csv
│   ├── variance_explained_by_factor_sex_and_view.csv
│   ├── variance_explained_heatmap.png
│   ├── mann_whitney_u_test_results_all_factors.csv
│   └── gsea/
│       └── <Factor>/
│           └── <view>/   (GSEA outputs per factor × tissue view)
└── paper_figures/
    ├── fig4_top_genes_<tissue>_<contrast>.pdf   (one per tissue × contrast)
    ├── fig4_gsea_sankey_summary.pdf
    ├── fig5_variance_explained_by_factor_and_group.pdf
    ├── fig5_factor_values_by_group_and_sex.pdf
    ├── fig5_PCA_of_factors_colored_by_group.pdf
    └── fig5_top_pathways_Factor6_heatmap.pdf
```

---

## Installation

```bash
# 1. Install the generic proteomics_tools dependency
pip install -e /path/to/proteomics_tools

# 2. Install this repo
pip install -e /path/to/borras_rodriguez_2026
```

---

## Configuration

Edit **`src/borras_rodriguez_2026/config.py`** before running anything:

| Variable | What to set |
|---|---|
| `DATA_DIR` | Path to the folder containing raw TSVs and `metadata.csv` |
| `RESULTS_DIR` | Where all outputs are written |
| `GMT_DIR` | Folder containing `.gmt` gene-set files |
| `GMT_FILENAME_TEMPLATE` | Filename pattern, e.g. `{collection}.v2025.1.Mm.symbols.gmt` |
| `GLM_FAMILY_NAME` | `"Gaussian"`, `"Gamma"`, `"Poisson"`, or `"NegativeBinomial"` |
| `GLM_LINK_NAME` | `"identity"`, `"log"`, `"logit"`, or `"sqrt"` |
| `GSEA_COLLECTIONS` | List of collection identifiers to test |
| `MOFA_K`, `MOFA_SEED`, `MOFA_R_DECAY` | MOFA hyperparameters |

---

## Running the pipeline

### Full pipeline (all stages)
```bash
python -m borras_rodriguez_2026.pipeline
```

### Individual stages
```bash
# Stage 1 — preprocessing only
python -m borras_rodriguez_2026.preprocess_hp_adrenal
python -m borras_rodriguez_2026.preprocess_spleen

# Stage 2 — GLM + GSEA (all tissues, or one at a time)
python -m borras_rodriguez_2026.run_glm_gsea
python -m borras_rodriguez_2026.run_glm_gsea --tissue hipocamp

# Stage 3 — MOFA integration
python -m borras_rodriguez_2026.run_mofa

# Stage 4 — paper figures
python -m borras_rodriguez_2026.results_paper_fig_4
python -m borras_rodriguez_2026.results_paper_fig_5
```

### Skip stages when re-running
```bash
# Re-generate figures only (skip all computation)
python -m borras_rodriguez_2026.pipeline --skip-preprocess --skip-glm --skip-mofa
```

---

## Study design

| Variable | Values |
|---|---|
| Tissues | Hippocampus (`hipocamp`), Adrenal (`adrenal`), Spleen (`spleen`) |
| Groups | `ctrl_vehicle` (reference), `cus_vehicle`, `cus_dox` |
| Sexes | `female` (reference), `male` |
| GLM formula | `intensity ~ C(group) * C(sex) + nan_count` |
| Contrasts | 6 sex-stratified contrasts (see `config.CONTRASTS`) |
| MOFA K | 11 factors |
