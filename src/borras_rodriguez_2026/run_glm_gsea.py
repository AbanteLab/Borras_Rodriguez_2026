"""
run_glm_gsea.py — Per-tissue GLM fitting, delta-method contrasts, and GSEA.

For each tissue (hippocampus, adrenal, spleen):
  1. Load the processed long-format CSV.
  2. Merge the per-sample nan_count covariate (hp and adrenal only).
  3. Fit one GLM per assay using proteomics_analysis.glm.fit_GLMs.
  4. Compute delta-method contrasts via proteomics_analysis.glm.contrasts_delta_method.
  5. Generate volcano plots for all GLM coefficients and all contrasts.
  6. Run preranked GSEA on each GLM coefficient and each contrast coefficient.

Outputs are written under:
    RESULTS_DIR/glm_results/<tissue>/
    RESULTS_DIR/gsea_results/<tissue>/

Run directly:
    python -m repro_paper.run_glm_gsea [--tissue hipocamp] [--tissue adrenal] ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from proteomics_analysis.glm import fit_GLMs
from proteomics_analysis.gsea import run_gsea
from proteomics_analysis.plotting import make_volcano_plots

from .config import (
    CONTRASTS,
    GLM_FAMILY_NAME,
    GLM_FORMULA,
    GLM_LINK_NAME,
    GLM_RESPONSE_VAR,
    GMT_DIR,
    GMT_FILENAME_TEMPLATE,
    GSEA_COLLECTIONS,
    GSEA_FDR_CUTOFF,
    GSEA_MIN_OVERLAP,
    GSEA_MIN_SIZE,
    GSEA_MAX_SIZE,
    GSEA_PERMUTATIONS,
    GSEA_SEED,
    GLM_RESULTS_DIR,
    GSEA_RESULTS_DIR,
    PROCESSED_DATA_DIR,
    VOLCANO_COEF_THRESHOLD,
    VOLCANO_FDR_THRESHOLD,
    get_glm_family,
)

log = logging.getLogger(__name__)

ALL_TISSUES = ["hipocamp", "adrenal", "spleen"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processed(tissue: str) -> pd.DataFrame:
    """Load the processed long-format CSV for a tissue."""
    if tissue in ("hipocamp", "adrenal"):
        path = PROCESSED_DATA_DIR / tissue / f"log2_{tissue}_long_format_processed.csv"
    else:
        path = PROCESSED_DATA_DIR / "spleen" / "spleen_long_format_processed.csv"
    log.info("Loading %s from %s", tissue, path)
    df = pd.read_csv(path, header=0)
    df.columns = df.columns.str.lower()
    return df


def _merge_nan_count(df: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """Attach the nan_count fraction covariate for hp/adrenal (already in df
    for spleen, not needed since spleen formula omits it—but harmless)."""
    if "nan_count" in df.columns:
        return df
    nan_path = PROCESSED_DATA_DIR / tissue / f"sample_nan_counts_{tissue}.csv"
    if not nan_path.exists():
        raise FileNotFoundError(
            f"nan_count file not found: {nan_path}. "
            "Run preprocess_hp_adrenal.py first."
        )
    nc = pd.read_csv(nan_path).rename(
        columns={"Sample_ID": "sample_id", "Nan_Count": "nan_count"}
    )
    nc.columns = nc.columns.str.lower()
    nc = nc.set_index("sample_id")
    n_assays = df["assay"].nunique()
    df = df.merge(nc[["nan_count"]], left_on="sample_id", right_index=True, how="left")
    df["nan_count"] = df["nan_count"] / n_assays
    return df


def _run_gsea_for_series(
    ranked: pd.Series,
    base: str,
    out_dir: Path,
    background_genes: set[str],
) -> None:
    """Thin wrapper that calls proteomics_analysis.gsea.run_gsea."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_gsea(
            ranked_series=ranked,
            base=base,
            out_dir=str(out_dir),
            gmt_dir=str(GMT_DIR),
            collections=GSEA_COLLECTIONS,
            cutoff=GSEA_FDR_CUTOFF,
            permutation_num=GSEA_PERMUTATIONS,
            min_size=GSEA_MIN_SIZE,
            max_size=GSEA_MAX_SIZE,
            seed=GSEA_SEED,
            background_genes=background_genes,
            min_overlap=GSEA_MIN_OVERLAP,
            gmt_filename_template=GMT_FILENAME_TEMPLATE,
            make_plots=True,
        )
    except Exception as exc:
        log.warning("GSEA failed for %s: %s", base, exc)


# ---------------------------------------------------------------------------
# Main per-tissue routine
# ---------------------------------------------------------------------------

def run_tissue(tissue: str) -> None:
    """Run GLM + GSEA for one tissue end-to-end."""
    log.info("===== %s =====", tissue.upper())

    # ── directories ──────────────────────────────────────────────────────
    glm_out  = GLM_RESULTS_DIR  / tissue
    gsea_out = GSEA_RESULTS_DIR / tissue
    glm_out.mkdir(parents=True, exist_ok=True)
    gsea_out.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────
    df = _load_processed(tissue)
    if tissue in ("hipocamp", "adrenal"):
        df = _merge_nan_count(df, tissue)

    # Spleen formula does not include nan_count; drop it from the formula
    # string if the tissue is spleen to avoid statsmodels errors.
    formula = GLM_FORMULA
    if tissue == "spleen" and "nan_count" not in df.columns:
        formula = GLM_FORMULA.replace("+ nan_count", "").replace("nan_count +", "")

    background_genes: set[str] = set(df["assay"].dropna().unique())

    # ── check if results already exist ───────────────────────────────────
    results_path   = glm_out / f"{tissue}_differential_expression_results.csv"
    contrasts_path = glm_out / f"{tissue}_glm_contrasts_per_protein_with_fdr.csv"

    if results_path.exists() and contrasts_path.exists():
        log.info("GLM results already exist, loading from disk.")
        results_df   = pd.read_csv(results_path,   index_col=0)
        contrasts_df = pd.read_csv(contrasts_path, index_col=0)
    else:
        # ── fit GLMs ─────────────────────────────────────────────────────
        family = get_glm_family()
        results_df, contrasts_df, exp_vars_df, normality_results = fit_GLMs(
            df,
            formula=formula,
            family=family,
            response_var=GLM_RESPONSE_VAR,
            output_dir=str(glm_out),
            contrasts=CONTRASTS,
            id_col="assay",
            label=tissue,
            make_plots=True,
        )

        # ── normality summary ─────────────────────────────────────────────
        p_values = [p for _, _, p in normality_results]
        if p_values:
            reject, _, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
            log.info("Genes failing normality (BH-corrected): %d", reject.sum())
            W_values = [W for _, W, _ in normality_results]
            plt.figure(figsize=(6, 4))
            plt.hist(W_values, bins=50)
            plt.xlabel("Shapiro-Wilk W")
            plt.ylabel("Number of genes")
            plt.title(f"Residual normality — {tissue}")
            plt.tight_layout()
            plt.savefig(glm_out / f"{tissue}_shapiro_W_histogram.png", dpi=150)
            plt.close()

        # ── save GLM results ──────────────────────────────────────────────
        results_df.to_csv(results_path)
        contrasts_df.to_csv(contrasts_path)
        exp_vars_df.to_csv(glm_out / f"{tissue}_explained_variance.csv")
        norm_df = pd.DataFrame(normality_results, columns=["assay", "W", "p_value"])
        norm_df.to_csv(glm_out / f"{tissue}_normality_results.csv", index=False)
        log.info(
            "GLM results saved.  proteins=%d  contrasts=%d",
            len(results_df), len(contrasts_df),
        )

    # ── volcano plots ─────────────────────────────────────────────────────
    for label, df_plot in [("glm", results_df), ("contrasts", contrasts_df)]:
        try:
            make_volcano_plots(
                df_plot,
                output_dir=str(glm_out / label),
                fdr_threshold=VOLCANO_FDR_THRESHOLD,
                coef_threshold=VOLCANO_COEF_THRESHOLD,
            )
        except Exception as exc:
            log.warning("Volcano plots failed for %s/%s: %s", tissue, label, exc)

    # ── GSEA on GLM coefficients ───────────────────────────────────────────
    log.info("Running GSEA on GLM coefficients for %s …", tissue)
    param_bases = sorted({c.rsplit("_", 1)[0] for c in results_df.columns if c.endswith("_coef")})
    for base in param_bases:
        coef_col = f"{base}_coef"
        if coef_col not in results_df.columns:
            continue
        ranked = results_df[coef_col].dropna().sort_values(ascending=False)
        if ranked.empty:
            continue
        _run_gsea_for_series(
            ranked=ranked,
            base=base,
            out_dir=gsea_out / "glm_coefficients" / base,
            background_genes=background_genes,
        )

    # ── GSEA on contrast coefficients ─────────────────────────────────────
    log.info("Running GSEA on contrasts for %s …", tissue)
    contrast_bases = sorted(
        {c.rsplit("_", 1)[0] for c in contrasts_df.columns if c.endswith("_coef")}
    )
    for base in contrast_bases:
        coef_col = f"{base}_coef"
        if coef_col not in contrasts_df.columns:
            continue
        ranked = contrasts_df[coef_col].dropna().sort_values(ascending=False)
        if ranked.empty:
            continue
        _run_gsea_for_series(
            ranked=ranked,
            base=base,
            out_dir=gsea_out / "contrasts" / base,
            background_genes=background_genes,
        )

    log.info("Done with %s.", tissue)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GLM + GSEA for one or more tissues."
    )
    parser.add_argument(
        "--tissue",
        action="append",
        choices=ALL_TISSUES,
        dest="tissues",
        help="Tissue to process (can be repeated). Default: all three.",
    )
    args = parser.parse_args()
    tissues = args.tissues or ALL_TISSUES

    for tissue in tissues:
        run_tissue(tissue)
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
        runpy.run_module("repro_paper.run_glm_gsea", run_name="__main__", alter_sys=True)
    else:
        # Already running as module: just call the entry point.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        main()
