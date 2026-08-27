"""
run_mofa.py — Multi-omics factor analysis (MOFA+) integration pipeline.

Steps:
  1. Load sex-regressed long-format data for all three tissues.
  2. Merge into a single MOFA-format DataFrame (feature, sample, view, value).
  3. Select top-variable features per tissue (covering MOFA_VAR_EXPL_THRESHOLD
     of the total within-tissue variance).
  4. Train MOFA+ (or load an existing model if the HDF5 already exists).
  5. Save factor scores, weights, and variance-explained tables.
  6. Run Mann-Whitney U tests between groups for every factor × sex.
  7. Run preranked GSEA on each factor's weights, per tissue view.

Outputs are written under RESULTS_DIR/mofa/.

Run directly:
    python -m repro_paper.run_mofa
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mofax as mfx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from proteomics_analysis.gsea import run_gsea

from .config import (
    GMT_DIR,
    GMT_FILENAME_TEMPLATE,
    GSEA_FDR_CUTOFF,
    GSEA_MIN_OVERLAP,
    GSEA_MIN_SIZE,
    GSEA_MAX_SIZE,
    GSEA_PERMUTATIONS,
    GSEA_SEED,
    MOFA_DIR,
    MOFA_GSEA_COLLECTIONS,
    MOFA_K,
    MOFA_R_DECAY,
    MOFA_SEED,
    MOFA_VAR_EXPL_THRESHOLD,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SAMPLES_TO_EXCLUDE,
    TISSUE_VIEW_NAMES,
)
from .utils import (
    build_sample_info,
    compute_n_genes_for_variance_threshold,
    get_top_variable_genes,
    regress_out_sex,
    train_mofa,
)

log = logging.getLogger(__name__)

GROUPS = ["ctrl_vehicle", "cus_vehicle", "cus_dox"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_tissue(tissue: str) -> pd.DataFrame:
    if tissue in ("hipocamp", "adrenal"):
        path = PROCESSED_DATA_DIR / tissue / f"log2_{tissue}_long_format_processed.csv"
    else:
        path = PROCESSED_DATA_DIR / "spleen" / "spleen_long_format_processed.csv"
    df            = pd.read_csv(path, header=0)
    df.columns    = df.columns.str.lower()
    df["tissue"]  = TISSUE_VIEW_NAMES[tissue]
    return df


def _load_nan_counts(tissue: str) -> pd.DataFrame:
    """Load per-sample nan counts for hp / adrenal."""
    path = PROCESSED_DATA_DIR / tissue / f"sample_nan_counts_{tissue}.csv"
    nc   = pd.read_csv(path)
    nc.columns = nc.columns.str.lower()
    return nc


# ---------------------------------------------------------------------------
# Sex regression helpers
# ---------------------------------------------------------------------------

def _get_sex_regressed(df: pd.DataFrame, tissue_key: str) -> pd.DataFrame:
    """Load sex-regressed CSV if it exists, otherwise compute and save it."""
    if tissue_key in ("hipocamp", "adrenal"):
        cache_path = PROCESSED_DATA_DIR / tissue_key / f"log2_{tissue_key}_long_format_processed_sex_regout.csv"
    else:
        cache_path = PROCESSED_DATA_DIR / "spleen" / "spleen_long_format_processed_sex_regout.csv"

    if cache_path.exists():
        log.info("Loading sex-regressed data from %s", cache_path)
        df_out = pd.read_csv(cache_path)
        df_out.columns = df_out.columns.str.lower()
    else:
        log.info("Regressing out sex for %s …", tissue_key)
        df_out = regress_out_sex(df.copy())
        df_out.to_csv(cache_path, index=False)
        log.info("Saved sex-regressed data to %s", cache_path)

    return df_out


# ---------------------------------------------------------------------------
# GSEA helper
# ---------------------------------------------------------------------------

def _run_factor_gsea(
    weights: pd.Series,
    factor: str,
    view: str,
    out_dir: Path,
    background_genes: set[str],
) -> None:
    """Run GSEA for one factor × view weight vector."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_gsea(
            ranked_series=weights.sort_values(ascending=False),
            base=f"{factor}_{view}",
            out_dir=str(out_dir),
            gmt_dir=str(GMT_DIR),
            collections=MOFA_GSEA_COLLECTIONS,
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
        log.warning("GSEA failed for %s / %s: %s", factor, view, exc)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_mofa_pipeline() -> None:
    mofa_out = MOFA_DIR
    mofa_out.mkdir(parents=True, exist_ok=True)
    model_path = (
        mofa_out
        / f"mofa_model_K{MOFA_K}_seed{MOFA_SEED}_Rdecay{MOFA_R_DECAY}"
          f"_varexpl{MOFA_VAR_EXPL_THRESHOLD}.hdf5"
    )

    # ── 1. Load all three tissues ─────────────────────────────────────────
    df_hp  = _load_tissue("hipocamp")
    df_sr  = _load_tissue("adrenal")
    df_spl = _load_tissue("spleen")

    # Standardise spleen labels (already done in preprocess_spleen, but be safe).
    from .config import SPLEEN_TREATMENT_MAP, SPLEEN_CONDITION_MAP
    df_spl["treatment"] = df_spl["treatment"].replace(SPLEEN_TREATMENT_MAP)
    df_spl["condition"] = df_spl["condition"].replace(SPLEEN_CONDITION_MAP)
    if "group" not in df_spl.columns:
        df_spl["group"] = df_spl["condition"] + "_" + df_spl["treatment"]

    # Exclude bad samples.
    for sid in SAMPLES_TO_EXCLUDE:
        for df_ in (df_hp, df_sr, df_spl):
            df_.drop(df_[df_["sample_id"].astype(str) == str(sid)].index, inplace=True)

    # ── 2. Build sample-info table and save ───────────────────────────────
    nc_hp = _load_nan_counts("hipocamp")
    nc_sr = _load_nan_counts("adrenal")
    sample_info = build_sample_info(df_spl, df_hp, df_sr, nc_hp, nc_sr)
    sample_info = sample_info[sample_info["sample_id"] != str(SAMPLES_TO_EXCLUDE[0])]
    sample_info.to_csv(RESULTS_DIR / "sample_info.csv", index=False)
    log.info("Sample info saved  n_samples=%d", len(sample_info))

    # ── 3. Sex regression ─────────────────────────────────────────────────
    df_hp  = _get_sex_regressed(df_hp,  "hipocamp")
    df_sr  = _get_sex_regressed(df_sr,  "adrenal")
    df_spl = _get_sex_regressed(df_spl, "spleen")

    # ── 4. Build MOFA-format DataFrame ────────────────────────────────────
    intensity_var = "intensity_no_sex"
    cols_keep     = ["assay", "sample_id", "group", "tissue", intensity_var]

    def _to_mofa_fmt(df_: pd.DataFrame) -> pd.DataFrame:
        sub = df_[cols_keep].copy()
        sub = sub.rename(columns={
            "assay":       "feature",
            intensity_var: "value",
            "sample_id":   "sample",
            "tissue":      "view",
        })
        sub["sample"] = sub["sample"].astype(str)
        return sub

    df_all = pd.concat(
        [_to_mofa_fmt(df_hp), _to_mofa_fmt(df_sr), _to_mofa_fmt(df_spl)],
        ignore_index=True,
    )
    # Remove excluded sample.
    for sid in SAMPLES_TO_EXCLUDE:
        df_all = df_all[df_all["sample"] != str(sid)]

    df_all.to_csv(mofa_out / "df_all_for_mofa.csv", index=False)
    log.info("MOFA input DataFrame  shape=%s", df_all.shape)

    # ── 5. Select top-variable features ───────────────────────────────────
    n_genes_per_view = compute_n_genes_for_variance_threshold(df_all, MOFA_VAR_EXPL_THRESHOLD)
    log.info("Top-variable genes (%.0f%% var): %s", MOFA_VAR_EXPL_THRESHOLD * 100, n_genes_per_view)
    top_genes = get_top_variable_genes(df_all, n=n_genes_per_view)

    df_mofa = df_all[
        df_all.apply(lambda row: row["feature"] in top_genes[row["view"]], axis=1)
    ].copy()

    # ── 6. Train or load MOFA model ───────────────────────────────────────
    if not model_path.exists():
        log.info("Training MOFA model  K=%d  seed=%d  R_decay=%s …", MOFA_K, MOFA_SEED, MOFA_R_DECAY)
        ent = train_mofa(df_mofa, K=MOFA_K, R_decay=MOFA_R_DECAY, seed=MOFA_SEED)
        ent.save(outfile=str(model_path))
        log.info("Model saved to %s", model_path)
    else:
        log.info("Loading existing MOFA model from %s", model_path)

    m = mfx.mofa_model(str(model_path))

    # Attach metadata to model.
    m.metadata = m.metadata.join(
        sample_info.set_index("sample_id")[["sex", "nan_count"]],
        on="sample",
        how="left",
    )
    if "group" not in m.metadata.columns:
        m.metadata = m.metadata.join(
            sample_info.set_index("sample_id")["group"],
            on="sample",
            how="left",
        )

    log.info("Model: %d cells, %d factors, views=%s", m.shape[0], m.nfactors, m.views)

    # ── 7. Append tissue suffix to feature names (required by mofax) ──────
    for tissue_view in m.views:
        feats = [str(f) for f in m.features[tissue_view]]
        feats = [f if f.endswith(tissue_view) else f + tissue_view for f in feats]
        m.features[tissue_view] = np.array(feats, dtype=object)

    # ── 8. Save factor scores and weights ─────────────────────────────────
    factors_matrix = m.get_factors(df=True)
    factors_matrix.to_csv(mofa_out / "factors_matrix.csv")
    factors_matrix.to_excel(mofa_out / "factors_matrix.xlsx")

    weights = m.get_weights(df=True)
    weights["tissue"] = weights.index.to_series().apply(
        lambda x: next((v for v in m.views if x.endswith(v)), "unknown")
    )
    weights.index = weights.index.to_series().apply(
        lambda x: next((x.replace(v, "") for v in m.views if x.endswith(v)), x)
    )
    weights.to_csv(mofa_out / "factor_weights_per_feature.csv")
    weights.to_excel(mofa_out / "factor_weights_per_feature.xlsx")
    log.info("Factor scores and weights saved.")

    # ── 9. Variance explained ─────────────────────────────────────────────
    r2 = m.get_variance_explained(
        factors=range(m.nfactors),
        views=range(m.nviews),
    )
    r2_mean = (
        r2.groupby(["View", "Factor"])
        .agg({"R2": "mean"})
        .unstack(level=0)["R2"]
    )
    r2_mean.to_csv(mofa_out / "variance_explained_by_factor_and_view.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(r2_mean, cmap="Blues", cbar_kws={"label": "R²"})
    plt.title("R² by Factor and View (mean across groups)")
    plt.tight_layout()
    plt.savefig(mofa_out / "variance_explained_heatmap.png", dpi=150)
    plt.close()

    # ── 10. Variance explained by sex group (for figures) ─────────────────
    r2_sex = m.calculate_variance_explained(
        factors=[f"Factor{i+1}" for i in range(m.nfactors)],
        group_label="sex",
        per_factor=True,
    )
    r2_sex.to_csv(mofa_out / "variance_explained_by_factor_sex_and_view.csv", index=False)

    # ── 11. Factor-level statistical tests (Mann-Whitney U) ───────────────
    factors = [f"Factor{i+1}" for i in range(m.nfactors)]
    results_rows = []
    for sex in ["female", "male"]:
        for factor in factors:
            fvals = m.fetch_values(factor, unique=True)
            for i in range(len(GROUPS)):
                for j in range(i + 1, len(GROUPS)):
                    g1_vals = fvals[
                        (m.metadata["group"] == GROUPS[i]) & (m.metadata["sex"] == sex)
                    ]
                    g2_vals = fvals[
                        (m.metadata["group"] == GROUPS[j]) & (m.metadata["sex"] == sex)
                    ]
                    if len(g1_vals) < 2 or len(g2_vals) < 2:
                        continue
                    stat, p = mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")
                    results_rows.append({
                        "Factor":      factor,
                        "Group1":      GROUPS[i],
                        "Group2":      GROUPS[j],
                        "Sex":         sex,
                        "Statistic":   float(stat),
                        "P-value":     float(p),
                        "Comparison":  f"{GROUPS[i]} vs {GROUPS[j]}",
                    })

    mwu_df = pd.DataFrame(results_rows)
    mwu_df.to_csv(mofa_out / "mann_whitney_u_test_results_all_factors.csv", index=False)
    log.info("Mann-Whitney U results saved  n_tests=%d", len(mwu_df))

    # ── 12. GSEA on factor weights per view ───────────────────────────────
    gsea_mofa_dir = MOFA_DIR / "gsea"
    log.info("Running GSEA on MOFA factor weights …")

    weights_full = m.get_weights(df=True)

    for view in m.views:
        view_weights = weights_full[weights_full.index.str.endswith(view)].copy()
        view_weights.index = view_weights.index.str.replace(view, "", regex=False)
        background = set(view_weights.index.tolist())

        for factor in factors:
            if factor not in view_weights.columns:
                continue
            w = view_weights[factor].dropna()
            if w.empty:
                continue
            _run_factor_gsea(
                weights=w,
                factor=factor,
                view=view,
                out_dir=gsea_mofa_dir / factor / view,
                background_genes=background,
            )

    log.info("MOFA pipeline complete.")
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
        runpy.run_module("repro_paper.run_mofa", run_name="__main__", alter_sys=True)
    else:
        # Already running as module: just call the entry point.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        run_mofa_pipeline()
