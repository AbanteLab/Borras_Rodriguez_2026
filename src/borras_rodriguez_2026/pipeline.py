"""
pipeline.py — Top-level orchestrator for the full repro_paper pipeline.

Runs all stages in order:
  1. Preprocess hippocampus and adrenal data.
  2. Preprocess spleen data.
  3. GLM + GSEA per tissue.
  4. MOFA integration + GSEA on factors.
  5. Generate Figure 4.
  6. Generate Figure 5.

Each stage is idempotent: if its output files already exist it loads them
from disk rather than recomputing.  You can therefore re-run the full
pipeline safely after a partial run, or start from any intermediate stage
by calling the individual module scripts directly.

Usage:
    python -m repro_paper.pipeline [--skip-preprocess] [--skip-glm]
                                   [--skip-mofa]       [--skip-figures]
                                   [--tissue hipocamp] [--tissue adrenal]
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full repro_paper analysis pipeline."
    )
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (assume CSVs already exist).")
    parser.add_argument("--skip-glm",        action="store_true",
                        help="Skip GLM + GSEA (assume results already exist).")
    parser.add_argument("--skip-mofa",       action="store_true",
                        help="Skip MOFA integration (assume model already exists).")
    parser.add_argument("--skip-figures",    action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--tissue", action="append", dest="tissues",
                        choices=["hipocamp", "adrenal", "spleen"],
                        help="Limit GLM/GSEA to specific tissues (default: all).")
    args = parser.parse_args()

    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    if not args.skip_preprocess:
        log.info("━━━ Stage 1: Preprocessing ━━━")
        from .preprocess_hp_adrenal import preprocess_tissue
        from .preprocess_spleen import preprocess_spleen

        for tissue in ("hipocamp", "adrenal"):
            if args.tissues is None or tissue in args.tissues:
                log.info("Preprocessing %s …", tissue)
                preprocess_tissue(tissue)

        if args.tissues is None or "spleen" in args.tissues:
            log.info("Preprocessing spleen …")
            preprocess_spleen()
    else:
        log.info("Skipping preprocessing.")

    # ── Stage 2: GLM + GSEA ───────────────────────────────────────────────
    if not args.skip_glm:
        log.info("━━━ Stage 2: GLM + GSEA ━━━")
        from .run_glm_gsea import run_tissue
        tissues = args.tissues or ["hipocamp", "adrenal", "spleen"]
        for tissue in tissues:
            run_tissue(tissue)
    else:
        log.info("Skipping GLM + GSEA.")

    # ── Stage 3: MOFA ─────────────────────────────────────────────────────
    if not args.skip_mofa:
        log.info("━━━ Stage 3: MOFA integration ━━━")
        from .run_mofa import run_mofa_pipeline
        run_mofa_pipeline()
    else:
        log.info("Skipping MOFA.")

    # ── Stage 4: Figures ──────────────────────────────────────────────────
    if not args.skip_figures:
        log.info("━━━ Stage 4: Paper figures ━━━")
        from .results_paper_fig_4 import main as fig4
        from .results_paper_fig_5 import main as fig5
        log.info("Generating Figure 4 …")
        fig4()
        log.info("Generating Figure 5 …")
        fig5()
    else:
        log.info("Skipping figure generation.")

    log.info("Pipeline complete.")
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
        runpy.run_module("repro_paper.pipeline", run_name="__main__", alter_sys=True)
    else:
        # Already running as module: just call the entry point.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        main()
