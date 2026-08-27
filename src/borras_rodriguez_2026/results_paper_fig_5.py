"""
results_paper_fig_5.py — Generate Figure 5 of the repro paper.

Figure 5 presents the MOFA integration results:
  - Panel A: Heatmap of variance explained by factor and group.
  - Panel B: Factor values by group and sex (violin + strip plots).
  - Panel C: PCA of factor scores, colored by group.
  - Panel D: Heatmap of top GSEA pathways for the key factor (Factor6)
             across tissue views.

Reads from:
    RESULTS_DIR/mofa/
    RESULTS_DIR/sample_info.csv

Writes to:
    RESULTS_DIR/paper_figures/

Run directly:
    python -m repro_paper.results_paper_fig_5
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, MOFA_DIR, RESULTS_DIR
from .pathway_classes import PATHWAY_CLASSES
from .utils_figures import (
    assign_pathway_class,
    attach_sample_metadata,
    calculate_variance_explained_by_group,
    compute_factor_pca,
    define_pathway_class_palette,
    get_fig5_paths,
    load_gsea_results,
    load_mofa_model,
    plot_factor_pca,
    plot_factor_pca_by_sex,
    plot_factor_values_by_group,
    plot_top_pathway_heatmap,
    plot_variance_explained_heatmap,
    prepare_top_pathway_heatmap,
)

log = logging.getLogger(__name__)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"]   = 12

# ── Study-specific figure parameters ─────────────────────────────────────────
FACTOR_TO_SUMMARIZE  = "Factor6"
FACTORS_TO_PLOT      = ["Factor4", "Factor5", "Factor6", "Factor8"]
TISSUE_ORDER         = ["hp", "sr", "Spleen"]
MOFA_GSEA_COLLECTIONS = [
    "m2.cgp", "m2.cp.biocarta", "m2.cp.reactome",
    "m2.cp.wikipathways", "m7.all", "m8.all", "mh.all",
]
GROUP_COLORS = {
    "ctrl_vehicle": (0.831, 0.831, 0.831),
    "cus_dox":      (1.000, 0.501, 0.006),
    "cus_vehicle":  (1.000, 0.753, 0.502),
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths           = get_fig5_paths()
    class_color_map = define_pathway_class_palette(PATHWAY_CLASSES.keys())

    # ── Load MOFA model ───────────────────────────────────────────────────
    log.info("Loading MOFA model …")
    mofa_model = load_mofa_model(paths["mofa_dir"])
    mofa_model = attach_sample_metadata(
        mofa_model, RESULTS_DIR / "sample_info.csv"
    )

    factors = mofa_model.get_factors(df=True)

    # ── Panel A: Variance explained ───────────────────────────────────────
    log.info("Plotting variance explained heatmap …")
    r2 = calculate_variance_explained_by_group(mofa_model, factors)
    plot_variance_explained_heatmap(
        r2,
        paths["figures_dir"] / "fig5_variance_explained_by_factor_and_group.pdf",
    )

    # ── Panel B: Factor values by group and sex ───────────────────────────
    log.info("Plotting factor values by group and sex …")
    plot_factor_values_by_group(
        mofa_model,
        FACTORS_TO_PLOT,
        paths["figures_dir"] / "fig5_factor_values_by_group_and_sex.pdf",
        GROUP_COLORS,
    )

    # ── Panel C: PCA of factor scores ─────────────────────────────────────
    log.info("Computing factor PCA …")
    pca_df, pca = compute_factor_pca(mofa_model)
    plot_factor_pca(
        pca_df,
        pca,
        paths["figures_dir"] / "fig5_PCA_of_factors_colored_by_group.pdf",
        GROUP_COLORS,
    )
    plot_factor_pca_by_sex(pca_df, pca)

    # Print top contributing factors for reference.
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=factors.columns,
    )
    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings["abs_PC2"] = loadings["PC2"].abs()
    log.info("Top 5 factors → PC1:\n%s", loadings.sort_values("abs_PC1", ascending=False).head(5))
    log.info("Top 5 factors → PC2:\n%s", loadings.sort_values("abs_PC2", ascending=False).head(5))

    # ── Panel D: Top-pathway heatmap for the key factor ───────────────────
    log.info("Loading MOFA GSEA results …")
    gsea_results_factors = load_gsea_results(
        paths["mofa_dir"],
        factors=list(factors.columns),
        collections=MOFA_GSEA_COLLECTIONS,
        tissue_order=TISSUE_ORDER,
    )

    if not gsea_results_factors.empty:
        gsea_results_factors["Pathway_Class"] = gsea_results_factors["Term"].apply(
            assign_pathway_class
        )

        data_plot, col_colors, legend_handles = prepare_top_pathway_heatmap(
            gsea_results_factors,
            factor=FACTOR_TO_SUMMARIZE,
            tissue_order=TISSUE_ORDER,
            top_n=10,
            class_color_map=class_color_map,
        )
        plot_top_pathway_heatmap(
            data_plot,
            col_colors,
            legend_handles,
            paths["figures_dir"] / f"fig5_top_pathways_{FACTOR_TO_SUMMARIZE}_heatmap.pdf",
        )
    else:
        log.warning("No MOFA GSEA results found — skipping pathway heatmap.")

    log.info("Figure 5 complete — outputs in %s", paths["figures_dir"])
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    if __spec__ is None:
        # Running as plain script: bootstrap the package then re-run as module.
        _here = _Path(__file__).resolve()
        for _parent in _here.parents:
            _src = _parent / "src"
            if (_src / "repro_paper" / "__init__.py").exists():
                if str(_src) not in sys.path:
                    sys.path.insert(0, str(_src))
                break
        import runpy
        runpy.run_module("repro_paper.results_paper_fig_5", run_name="__main__", alter_sys=True)
    else:
        # Already running as module: just call the entry point.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        main()
