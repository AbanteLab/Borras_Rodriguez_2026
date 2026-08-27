"""
preprocess_spleen.py — Load, clean, and save spleen proteomics data.

Reads the raw spleen TSV, resolves duplicate genes, converts to long format,
maps treatment/condition labels, merges metadata, and writes:

    <PROCESSED_DATA_DIR>/spleen/spleen_long_format_processed.csv

Run directly:
    python -m repro_paper.preprocess_spleen
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from proteomics_analysis.plotting import plot_pca

from .config import (
    GROUP_CATEGORIES,
    GROUP_COLORS,
    METADATA_FILE,
    PROCESSED_DATA_DIR,
    SAMPLES_TO_EXCLUDE,
    SEX_CATEGORIES,
    SEX_COLORS,
    SPLEEN_CONDITION_MAP,
    SPLEEN_RAW_FILE,
    SPLEEN_TREATMENT_MAP,
)
from .utils import treat_spleen_df

log = logging.getLogger(__name__)


def preprocess_spleen() -> pd.DataFrame:
    """Full preprocessing pipeline for the spleen tissue.

    Returns
    -------
    DataFrame
        Processed long-format DataFrame ready for GLM fitting.  Also written
        to ``PROCESSED_DATA_DIR/spleen/``.
    """
    out_dir = PROCESSED_DATA_DIR / "spleen"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    df = pd.read_csv(SPLEEN_RAW_FILE, sep="\t")
    log.info("Loaded spleen raw file  shape=%s", df.shape)

    # ------------------------------------------------------------------
    # 2. Basic cleaning
    # ------------------------------------------------------------------
    print(df)
    df = df[df["Genes"].notna()]
    for bad in ("Protein argonaute-1", "Protein argonaute-2"):
        if "Protein Names" in df.columns:
            df = df[df["Protein Names"] != bad]

    # Resolve duplicated Uniprot entries.
    dup_uniprot = df[df.duplicated(subset=["Uniprot"], keep=False)]
    log.info("Duplicated Uniprot rows: %d", len(dup_uniprot))

    # Identify duplicated genes where one row is a fragment/isoform.
    dup_genes        = df[df.duplicated(subset=["Genes"], keep=False)]
    quantity_cols    = [
        c for c in df.columns
        if c not in ["Genes", "Protein Names", "Uniprot", "Peptides (max)"]
        and c != "- Log p-value"  # keep everything before this separator
    ]
    key_cols         = [c for c in df.columns if c not in quantity_cols]

    genes_frag_iso   = []
    for gene, group in dup_genes.groupby("Genes"):
        descs = group["Protein Names"].str.lower()
        if descs.str.contains("fragment").any() or descs.str.contains("isoform").any():
            genes_frag_iso.append(gene)

    df_no_dup = df[~df["Genes"].isin(dup_genes["Genes"])].copy()
    for gene in genes_frag_iso:
        group   = dup_genes[dup_genes["Genes"] == gene]
        new_row = {col: ";".join(group[col].astype(str).unique()) for col in key_cols}
        for col in quantity_cols:
            new_row[col] = group[col].sum()
        df_no_dup = pd.concat(
            [df_no_dup, pd.DataFrame([new_row])], ignore_index=True
        )

    # ------------------------------------------------------------------
    # 3. Convert to long format
    # ------------------------------------------------------------------
    df_long = treat_spleen_df(df_no_dup)

    # ------------------------------------------------------------------
    # 4. Fix known sample ID errors
    # ------------------------------------------------------------------
    df_long["sample_id"] = df_long["sample_id"].replace("1591", "1581")

    # ------------------------------------------------------------------
    # 5. Exclude bad samples
    # ------------------------------------------------------------------
    for sid in SAMPLES_TO_EXCLUDE:
        df_long = df_long[df_long["sample_id"] != sid]
        df_long = df_long[df_long["sample_id"] != str(sid)]

    # ------------------------------------------------------------------
    # 6. Standardise column names and label encodings
    # ------------------------------------------------------------------
    df_long.columns   = df_long.columns.str.lower()
    df_long["treatment"] = df_long["treatment"].replace(SPLEEN_TREATMENT_MAP)
    df_long["condition"] = df_long["condition"].replace(SPLEEN_CONDITION_MAP)
    df_long["sex"]       = df_long["sex"].str.lower()

    # ------------------------------------------------------------------
    # 7. Load metadata and validate sample overlap
    # ------------------------------------------------------------------
    metadata = pd.read_csv(METADATA_FILE)
    metadata.columns = metadata.columns.str.lower()

    meta_ids     = metadata["sample_id"].astype(str).str.zfill(2).unique()
    df_ids       = df_long["sample_id"].astype(str).str.zfill(2).unique()
    missing_meta = set(meta_ids) - set(df_ids)
    missing_df   = set(df_ids) - set(meta_ids)
    if missing_meta:
        log.warning("Samples in metadata not in spleen data: %s", missing_meta)
    if missing_df:
        log.warning("Samples in spleen data not in metadata: %s", missing_df)

    # ------------------------------------------------------------------
    # 8. Group column and categoricals
    # ------------------------------------------------------------------
    df_long["group"] = df_long["condition"] + "_" + df_long["treatment"]
    df_long["group"] = pd.Categorical(df_long["group"], categories=GROUP_CATEGORIES)
    df_long["sex"]   = pd.Categorical(df_long["sex"],   categories=SEX_CATEGORIES)

    # Scaled intensity (used internally for MOFA; raw intensity kept for GLM).
    df_long["scaled_intensity"] = df_long.groupby("assay")["intensity"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )

    # Ensure ctrl_vehicle is first (reference level for GLM).
    df_long = df_long.sort_values(
        by=["treatment", "condition"], ascending=[False, True]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    out_path = out_dir / "spleen_long_format_processed.csv"
    df_long.to_csv(out_path, index=False)
    log.info("Saved spleen to %s  shape=%s", out_path, df_long.shape)

    # ------------------------------------------------------------------
    # 10. PCA diagnostic plots
    # ------------------------------------------------------------------
    pca_dir = str(out_dir)
    for color_by, color_dict in [("sex", SEX_COLORS), ("group", GROUP_COLORS)]:
        try:
            plot_pca(
                df_long,
                color_by=color_by,
                color_dict=color_dict,
                output_dir=pca_dir,
                label="spleen",
                id_col="sample_id",
                assay_col="assay",
                value_col="intensity",
            )
        except Exception as exc:
            log.warning("PCA plot failed for spleen / %s: %s", color_by, exc)

    return df_long

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
        runpy.run_module("repro_paper.preprocess_spleen", run_name="__main__", alter_sys=True)
    else:
        # Already running as module: just call the entry point.
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        preprocess_spleen()
