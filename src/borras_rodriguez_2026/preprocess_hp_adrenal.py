"""
preprocess_hp_adrenal.py — Load, clean, impute, and save hippocampus and adrenal data.

Faithfully follows the original data_analysis_hp_sr.py + data_treatment_hp_sr.py pipeline:

  1. Compute per-sample nan_count from RAW, pre-imputation intensities
     (mirrors data_analysis_hp_sr.py / utils.treat_hp_sr_data) and save it.
     This must happen before imputation — by the time the main pipeline
     below has run, there are no missing intensities left to count.
  2. Load raw TSV.
  3. Filter rows where >50% of quantity values are NaN.
  4. Log2 normalisation (counts per million).
  5. Remove rows with NaN gene name or multi-gene entries (containing ';').
  6. Classify missing values as MNAR, MAR, or ambiguous.
  7. Impute MNAR with down-shifted Gaussian; MAR/ambiguous with KNN.
  8. Resolve duplicated genes:
       - Fragment/Isoform → sum intensities.
       - Other duplicates  → keep row with highest PG.Cscore.
  9. Convert imputed data to long format and merge with metadata.
  10. Save processed long-format CSV (all samples — sample exclusion, e.g.
      SAMPLES_TO_EXCLUDE, is applied downstream by consumers, not here,
      matching the original pipeline where data_treatment_hp_sr.py's saved
      output is unfiltered and stat_analysis.py/integration.py exclude
      specific samples later).
  11. PCA diagnostic plots.

Run directly:
    python -m repro_paper.preprocess_hp_adrenal
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from proteomics_analysis.plotting import plot_pca

from .config import (
    ADRENAL_RAW_FILE,
    GROUP_CATEGORIES,
    HP_RAW_FILE,
    IMPUTATION_SEED,
    METADATA_FILE,
    PROCESSED_DATA_DIR,
    SEX_CATEGORIES,
    GROUP_COLORS,
    SEX_COLORS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1-2: Load and filter
# ---------------------------------------------------------------------------

def _load_and_filter(path) -> tuple[pd.DataFrame, list[str]]:
    """Load TSV and drop rows where >50% of quantity columns are NaN."""
    df = pd.read_csv(path, sep="\t")
    log.info("Loaded %s  shape=%s", path, df.shape)

    quantity_cols = [col for col in df.columns if "Quantity" in col]
    log.info("Quantity columns found: %d", len(quantity_cols))

    # Keep rows where more than 50% of quantity values are non-NaN.
    df = df[df[quantity_cols].notna().sum(axis=1) > (len(quantity_cols) / 2)]
    log.info("Shape after >50%% NaN row filter: %s", df.shape)

    return df, quantity_cols


# ---------------------------------------------------------------------------
# Step 3: Log2 normalisation
# ---------------------------------------------------------------------------

def _log2_normalise(df: pd.DataFrame, quantity_cols: list[str]) -> pd.DataFrame:
    """Normalise each quantity column to counts per million then log2-transform."""
    df = df.copy()
    for col in quantity_cols:
        total = df[col].sum()
        df[col] = np.log2(df[col] / total * 1e6)
    return df


# ---------------------------------------------------------------------------
# Step 4: Gene-name cleaning
# ---------------------------------------------------------------------------

def _clean_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN gene names or multi-gene entries (containing ';')."""
    df = df[df["PG.Genes"].notna()]
    df = df[~df["PG.Genes"].str.contains(";")]
    log.info("Shape after gene-name cleaning: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 5: Missingness classification
# ---------------------------------------------------------------------------

def _classify_missingness(
    df: pd.DataFrame,
    quantity_cols: list[str],
) -> pd.DataFrame:
    """Return a DataFrame of the same shape as df[quantity_cols] with values
    'MNAR', 'MAR', 'ambiguous', or NaN (for observed values)."""
    detected_fraction = df[quantity_cols].notna().mean(axis=1)
    missingness = pd.DataFrame(index=df.index, columns=quantity_cols, dtype="object")

    for index, row in df.iterrows():
        prot_det_frac = detected_fraction.loc[index]
        for col in quantity_cols:
            if pd.isna(row[col]):
                nr_col = col.replace("Quantity", "NrOfStrippedSequencesIdentified")
                if nr_col in df.columns and row[nr_col] == 0:
                    missingness.at[index, col] = "MNAR"
                elif prot_det_frac >= 0.5:
                    missingness.at[index, col] = "MAR"
                else:
                    missingness.at[index, col] = "ambiguous"

    return missingness


# ---------------------------------------------------------------------------
# Step 6: Imputation
# ---------------------------------------------------------------------------

def _impute(
    df: pd.DataFrame,
    quantity_cols: list[str],
    missingness: pd.DataFrame,
) -> pd.DataFrame:
    """MNAR → down-shifted Gaussian; MAR/ambiguous → KNN (k=10)."""
    df_imputed = df.copy()

    # Peptide-wise variance for the MNAR imputation spread.
    peptide_variances = df[quantity_cols].var(axis=1)
    median_variance   = peptide_variances.median()
    log.info("Median peptide-wise variance: %.4f", median_variance)

    # MNAR: draw from a Gaussian centred at the 0.25% quantile of the column.
    # Seeded (unlike the original data_treatment_hp_sr.py) so preprocessing
    # is reproducible across runs; a local Generator avoids mutating global
    # numpy random state.
    rng = np.random.default_rng(IMPUTATION_SEED)
    for index, row in df.iterrows():
        for col in quantity_cols:
            if pd.isna(row[col]) and missingness.at[index, col] == "MNAR":
                col_quantile = df[col].quantile(0.0025)
                imputed      = rng.normal(
                    loc=col_quantile, scale=median_variance ** 0.5
                )
                df_imputed.at[index, col] = imputed

    log.info(
        "NaN after MNAR imputation: %d",
        df_imputed[quantity_cols].isna().sum().sum(),
    )

    # MAR / ambiguous: KNN imputation.
    imputer = KNNImputer(n_neighbors=10)
    df_imputed[quantity_cols] = imputer.fit_transform(df_imputed[quantity_cols])

    log.info(
        "NaN after KNN imputation: %d",
        df_imputed[quantity_cols].isna().sum().sum(),
    )
    return df_imputed


# ---------------------------------------------------------------------------
# Step 7: Duplicate gene resolution
# ---------------------------------------------------------------------------

def _resolve_duplicates(
    df_imputed: pd.DataFrame,
    quantity_cols: list[str],
) -> pd.DataFrame:
    """Resolve duplicated gene names:
    - Fragment/Isoform entries → sum intensities across rows.
    - Other duplicates → keep the row with the highest PG.Cscore.
    """
    duplicated_genes = df_imputed[
        df_imputed.duplicated(subset=["PG.Genes"], keep=False)
    ]
    log.info("Duplicated gene rows: %d", len(duplicated_genes))

    # Identify fragment/isoform cases.
    genes_frag_iso = []
    for gene, group in duplicated_genes.groupby("PG.Genes"):
        descs = group["PG.ProteinDescriptions"].str.lower()
        if descs.str.contains("fragment").any() or descs.str.contains("isoform").any():
            genes_frag_iso.append(gene)

    log.info("Fragment/isoform duplicates: %d", len(genes_frag_iso))

    # Start with non-duplicated rows.
    df_out = df_imputed[
        ~df_imputed["PG.Genes"].isin(duplicated_genes["PG.Genes"])
    ].copy()

    other_cols = [c for c in df_imputed.columns if c not in quantity_cols]

    # Case 1: fragment/isoform → sum intensities.
    for gene in genes_frag_iso:
        group   = duplicated_genes[duplicated_genes["PG.Genes"] == gene]
        new_row = {col: ";".join(group[col].astype(str).unique()) for col in other_cols}
        for col in quantity_cols:
            numeric_vals = pd.to_numeric(group[col], errors="coerce")
            new_row[col] = (
                numeric_vals.sum(skipna=True)
                if numeric_vals.notna().any()
                else np.nan
            )
        df_out = pd.concat(
            [df_out, pd.DataFrame([new_row])], ignore_index=True
        )

    # Case 2: other duplicates → keep row with highest PG.Cscore.
    non_frag_iso = duplicated_genes[
        ~duplicated_genes["PG.Genes"].isin(genes_frag_iso)
    ]
    for gene in non_frag_iso["PG.Genes"].unique():
        group    = non_frag_iso[non_frag_iso["PG.Genes"] == gene]
        best_row = group.loc[group["PG.Cscore"].idxmax()]
        df_out   = pd.concat(
            [df_out, pd.DataFrame([best_row])], ignore_index=True
        )

    log.info(
        "Duplicated genes after resolution: %d",
        df_out["PG.Genes"].duplicated().sum(),
    )
    return df_out


# ---------------------------------------------------------------------------
# Step 8-9: Long format conversion and metadata merge
# ---------------------------------------------------------------------------

def _to_long_format(
    df_processed: pd.DataFrame,
    quantity_cols: list[str],
    metadata: pd.DataFrame,
    tissue: str,
) -> pd.DataFrame:
    """Convert wide format to long format and merge with metadata."""
    records = []
    for col in quantity_cols:
        parts     = col.split("_")
        info      = parts[1].strip()
        split     = info.split(".")
        sample_id = re.findall(r"\d+", split[0])
        if not sample_id:
            continue
        records.append(pd.DataFrame({
            "Protein_ID":   df_processed["PG.UniProtIds"].values,
            "ProteinNames": df_processed["PG.ProteinDescriptions"].values,
            "Assay":        df_processed["PG.Genes"].values,
            "sample_id":    sample_id[0],
            "Intensity":    df_processed[col].values,
            "Tissue":       tissue,
        }))

    df_long = pd.concat(records, ignore_index=True)

    # Pad sample IDs for merge.
    df_long["sample_id"]  = df_long["sample_id"].astype(str).str.zfill(2)
    metadata              = metadata.copy()
    metadata[tissue]      = metadata[tissue].astype(int).astype(str).str.zfill(2)
    metadata["sample_id"] = metadata["sample_id"].astype(str).str.zfill(2)

    merged = df_long.merge(
        metadata,
        left_on="sample_id",
        right_on=tissue,
        how="left",
    )

    # Replace facility sample ID with the canonical study sample ID.
    merged["sample_id"] = merged["sample_id_y"]

    # Drop merge helper columns, guarding against missing ones.
    cols_to_drop = ["sample_id_x", "sample_id_y"]
    for extra in ["hipocamp", "adrenal", "melsa"]:
        if extra in merged.columns:
            cols_to_drop.append(extra)
    merged = merged.drop(columns=cols_to_drop)

    # Select and reorder final columns.
    base_cols   = ["Protein_ID", "ProteinNames", "Assay",
                   "condition", "treatment", "sex", "group",
                   "sample_id", "Intensity"]
    extra_cols  = [c for c in ["ncompanions", "partner_group"]
                   if c in merged.columns]
    final_cols  = base_cols[:6] + extra_cols + base_cols[6:]
    final_cols  = [c for c in final_cols if c in merged.columns]
    df_final    = merged[final_cols]

    # Lowercase all column names.
    df_final.columns = df_final.columns.str.lower()

    return df_final


# ---------------------------------------------------------------------------
# Pre-imputation nan_count (faithfully reproduces data_analysis_hp_sr.py)
# ---------------------------------------------------------------------------

def _raw_nan_counts(raw_file, metadata: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """Count genuine missing Intensity values per sample from the RAW data.

    Mirrors data_analysis_hp_sr.py + utils.treat_hp_sr_data: only rows with
    a null PG.UniProtIds or PG.Genes are dropped — no >50% filter, no log2,
    no duplicate-gene resolution, no imputation. This is the value later
    consumed downstream as the GLM's `nan_count` covariate, so it must
    reflect real pre-imputation missingness rather than the fully-imputed
    intensity column (which by construction has no NaNs left to count).

    ``metadata`` must already have lowercase columns, matching what
    ``preprocess_tissue`` builds before calling this.
    """
    df = pd.read_csv(raw_file, sep="\t")
    df = df[~df["PG.UniProtIds"].isnull()]
    df = df[~df["PG.Genes"].isnull()]
    df_filt = df[[
        col for col in df.columns
        if "Quantity" in col
        or col in ["PG.Genes", "PG.ProteinNames", "PG.ProteinDescriptions", "PG.UniProtIds"]
    ]]
    quantity_cols = [c for c in df_filt.columns if "Quantity" in c]

    records = []
    for col in quantity_cols:
        parts     = col.split("_")
        info      = parts[1].strip()
        split     = info.split(".")
        sample_id = re.findall(r"\d+", split[0])
        if not sample_id:
            continue
        records.append(pd.DataFrame({
            "sample_id": sample_id[0],
            "Intensity": df_filt[col].values,
        }))
    long_df = pd.concat(records, ignore_index=True)

    long_df["sample_id"] = long_df["sample_id"].astype(str).str.zfill(2)
    metadata              = metadata.copy()
    metadata[tissue]      = metadata[tissue].astype(int).astype(str).str.zfill(2)
    metadata["sample_id"] = metadata["sample_id"].astype(str).str.zfill(2)

    merged = long_df.merge(metadata, left_on="sample_id", right_on=tissue, how="left")
    merged["sample_id"] = merged["sample_id_y"]

    nan_counts = (
        merged[merged["Intensity"].isna()]
        .groupby("sample_id")
        .size()
        .reset_index(name="nan_count")
    )
    return nan_counts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_tissue(tissue: str) -> pd.DataFrame:
    """Full preprocessing pipeline for hippocampus or adrenal.

    Parameters
    ----------
    tissue : str
        Either ``'hipocamp'`` or ``'adrenal'``.

    Returns
    -------
    DataFrame
        Processed long-format DataFrame. Also written to
        ``PROCESSED_DATA_DIR/<tissue>/``.
    """
    raw_file = HP_RAW_FILE if tissue == "hipocamp" else ADRENAL_RAW_FILE
    out_dir  = PROCESSED_DATA_DIR / tissue
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata (needed for nan counts and final merge) ────────────
    metadata = pd.read_csv(METADATA_FILE)
    metadata = metadata.rename(
        columns={"ID": "Sample_ID", "trearment": "treatment"}
    )
    metadata.columns = metadata.columns.str.lower()
    metadata["group"] = (
        metadata["condition"].str.lower() + "_" + metadata["treatment"].str.lower()
    )

    # ── 1. Pre-imputation nan_count ───────────────────────────────────────
    # Must be computed from the RAW data, before any imputation happens —
    # this is the GLM's nan_count covariate, and by the time imputation has
    # run there are no missing intensities left to count.
    log.info("Computing pre-imputation nan_count …")
    nan_counts = _raw_nan_counts(raw_file, metadata, tissue)
    nan_counts.to_csv(out_dir / f"sample_nan_counts_{tissue}.csv", index=False)
    log.info(
        "Nan counts saved — total NaNs across samples: %d",
        int(nan_counts["nan_count"].sum()) if not nan_counts.empty else 0,
    )

    # ── 2-3. Load and filter ─────────────────────────────────────────────
    df, quantity_cols = _load_and_filter(raw_file)

    # ── 4. Log2 normalise ────────────────────────────────────────────────
    df = _log2_normalise(df, quantity_cols)

    # ── 5. Clean gene names ──────────────────────────────────────────────
    df = _clean_genes(df)

    # ── 6. Classify missingness ──────────────────────────────────────────
    log.info("Classifying missingness …")
    missingness = _classify_missingness(df, quantity_cols)

    # Save missingness summary.
    missingness_summary = pd.DataFrame({
        "PG.Genes":        df["PG.Genes"].values,
        "Fraction_Missing": df[quantity_cols].isna().mean(axis=1).values,
    })
    missingness_summary.to_csv(
        out_dir / f"{tissue}_missingness_summary.csv", index=False
    )

    # ── 7. Impute ────────────────────────────────────────────────────────
    log.info("Imputing missing values …")
    df_imputed = _impute(df, quantity_cols, missingness)

    # ── 8. Resolve duplicate genes ───────────────────────────────────────
    df_processed = _resolve_duplicates(df_imputed, quantity_cols)

    # Save wide-format processed file.
    df_processed.to_csv(out_dir / f"log2_{tissue}_processed.csv", index=False)

    # ── 9. Long format + metadata merge (imputed data for GLM) ───────────
    df_final = _to_long_format(df_processed, quantity_cols, metadata, tissue)

    # Note: sample exclusion (SAMPLES_TO_EXCLUDE, e.g. sample 1884) is NOT
    # applied here. The original pipeline's data_treatment_hp_sr.py saves
    # every sample and excludes specific ones later, downstream, in
    # stat_analysis.py / integration.py — so this saved CSV matches that
    # and downstream consumers (run_glm_gsea.py, run_mofa.py) apply the
    # exclusion themselves.

    # ── Categoricals for GLM reference level ─────────────────────────────
    df_final["group"] = pd.Categorical(
        df_final["group"], categories=GROUP_CATEGORIES
    )
    df_final["sex"] = pd.Categorical(
        df_final["sex"].str.lower(), categories=SEX_CATEGORIES
    )

    # Ensure ctrl_vehicle is first (reference level).
    df_final = df_final.sort_values(
        by=["treatment", "condition"], ascending=[False, True]
    ).reset_index(drop=True)

    # ── 10. Save long-format CSV ──────────────────────────────────────────
    out_path = out_dir / f"log2_{tissue}_long_format_processed.csv"
    df_final.to_csv(out_path, index=False)
    log.info("Saved %s  shape=%s", out_path, df_final.shape)

    # ── 11. PCA diagnostic plots ──────────────────────────────────────────
    for color_by, color_dict in [("sex", SEX_COLORS), ("group", GROUP_COLORS)]:
        try:
            plot_pca(
                df_final,
                color_by=color_by,
                color_dict=color_dict,
                output_dir=str(out_dir),
                label=tissue,
                id_col="sample_id",
                assay_col="assay",
                value_col="intensity",
            )
        except Exception as exc:
            log.warning("PCA failed for %s/%s: %s", tissue, color_by, exc)

    return df_final


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    # Only bootstrap when running as a plain script (python preprocess_hp_adrenal.py).
    # If already running as a module (python -m repro_paper.preprocess_hp_adrenal),
    # __spec__ is set and we skip straight to the normal entry point.
    if __spec__ is None:
        _here = _Path(__file__).resolve()
        for _parent in _here.parents:
            _src = _parent / "src"
            if (_src / "repro_paper" / "__init__.py").exists():
                if str(_src) not in sys.path:
                    sys.path.insert(0, str(_src))
                break
        import runpy
        runpy.run_module("repro_paper.preprocess_hp_adrenal", run_name="__main__", alter_sys=True)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        for t in ("hipocamp", "adrenal"):
            log.info("=== Preprocessing %s ===", t)
            preprocess_tissue(t)
        log.info("Done.")