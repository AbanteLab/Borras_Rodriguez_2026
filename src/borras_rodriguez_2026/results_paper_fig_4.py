"""
results_paper_fig_4.py — Generate Figure 4 of the repro paper.

Figure 4 summarises the per-tissue GLM + GSEA results:
  - Panel A: Top-protein bar plots per contrast, colored by pathway class.
  - Panel B: Sankey / dot-plot summary of significant GSEA terms per tissue.

Reads from:
    RESULTS_DIR/glm_results/<tissue>/
    RESULTS_DIR/gsea_results/<tissue>/contrasts/

Writes to:
    RESULTS_DIR/paper_figures/

Run directly:
    python -m repro_paper.results_paper_fig_4
"""

import logging

import matplotlib.pyplot as plt

from .config import FIGURES_DIR
from .pathway_classes import PATHWAY_CLASSES
from .utils_figures import (
    assign_pathway_class,
    define_pathway_class_palette,
    get_paper_paths,
    load_fig4_glm_results,
    load_fig4_gsea_results,
    plot_fig4_sankey,
    plot_fig4_top_gene_barplots,
)

log = logging.getLogger(__name__)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"]   = 12

TISSUES                   = ["hipocamp", "adrenal", "spleen"]
TOP_N_GENES_PER_CONTRAST  = 20


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths           = get_paper_paths()
    class_color_map = define_pathway_class_palette(PATHWAY_CLASSES.keys())

    log.info("Loading GLM contrast results …")
    all_contrasts = load_fig4_glm_results(
        paths["results_dir"],
        tissues=TISSUES,
    )

    log.info("Loading GSEA results …")
    gsea_results_all = load_fig4_gsea_results(
        paths["results_dir"],
        tissues=TISSUES,
        collections=[],      # collections arg is informational only; CSVs already merged
    )

    if not gsea_results_all.empty:
        gsea_results_all["Pathway_Class"] = gsea_results_all["Term"].apply(
            assign_pathway_class
        )
        gsea_results_all["NES_abs"] = gsea_results_all["NES"].abs()

    log.info("Writing top-gene bar plots …")
    plot_fig4_top_gene_barplots(
        all_contrasts=all_contrasts,
        gsea_results_all=gsea_results_all,
        class_color_map=class_color_map,
        figures_dir=paths["figures_dir"],
        ntop_genes_per_coef=TOP_N_GENES_PER_CONTRAST,
    )

    log.info("Writing GSEA summary (Sankey) …")
    plot_fig4_sankey(gsea_results_all, paths["figures_dir"])

    log.info("Figure 4 complete — outputs in %s", paths["figures_dir"])
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
        runpy.run_module("repro_paper.results_paper_fig_4", run_name="__main__", alter_sys=True)
    else:
        # Already running as module: just call the entry point.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        main()
