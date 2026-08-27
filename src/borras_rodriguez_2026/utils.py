"""
utils.py — Paper-specific helper functions for repro_paper.

Contains only functions that are specific to this study's data format or
MOFA pipeline.  Generic functions (add_fdr_columns, fit_GLMs,
contrasts_delta_method, run_gsea, plot_pca, make_volcano_plots,
plot_rank_change) are imported from the proteomics_analysis package.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import tqdm
from mofapy2.run.entry_point import entry_point


# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------

def row_is_reviewed(uniprot_ids: str) -> bool:
    """Return True if any accession in a semicolon-separated string is reviewed
    (i.e. does not start with 'A0A', which flags TrEMBL/unreviewed entries)."""
    accessions = uniprot_ids.split(";")
    return any(not acc.startswith("A0A") for acc in accessions)


def treat_hp_sr_data(df: pd.DataFrame, metadata: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """Parse a hippocampus or adrenal TSV into long format and merge with metadata.

    The raw file has one intensity column per sample, with column names like
    ``Report_<sample>.<suffix>``.  The function extracts the numeric sample ID
    from each column name, pivots to long format, and left-joins with metadata
    to attach condition, treatment, and sex.

    Parameters
    ----------
    df : DataFrame
        Raw wide-format DataFrame (as loaded from the .tsv file).
    metadata : DataFrame
        Metadata table.  Must contain columns ``Sample_ID``, ``condition``,
        ``Treatment``, ``sex``, and a column named ``tissue`` holding the
        old (facility-side) sample IDs for this tissue.
    tissue : str
        Tissue name used as the merge key into ``metadata`` (e.g.
        ``"hipocamp"`` or ``"adrenal"``).

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns:
        ``Protein_ID``, ``ProteinNames``, ``assay``, ``condition``,
        ``treatment``, ``sex``, ``sample_id``, ``intensity``, ``tissue``.
    """

    quantity_cols = [
    c for c in df.columns
        if len(c.split("_")) >= 2 and re.search(r"\d+", c.split("_")[1].split(".")[0])
    ]
    
    records = []

    for col in quantity_cols:
        parts     = col.split("_")
        info      = parts[1].strip()
        split     = info.split(".")
        sample_id = re.findall(r"\d+", split[0])
        if not sample_id:
            continue
        if not sample_id:
            continue
        tmp = pd.DataFrame(
            {
                "Protein_ID":   df["PG.UniProtIds"].values,
                "ProteinNames": df["PG.ProteinDescriptions"].values,
                "assay":        df["PG.Genes"].values,
                "sample_id":    sample_id[0],
                "intensity":    df[col].values,
                "tissue":       tissue,
            }
        )
        records.append(tmp)

    final_df = pd.concat(records, ignore_index=True)
    final_df["sample_id"] = final_df["sample_id"].astype(str).str.zfill(2)

    metadata = metadata.copy()
    print(metadata.keys())
    metadata[tissue]      = metadata[tissue].astype(int).astype(str).str.zfill(2)
    metadata["sample_id"] = metadata["sample_id"].astype(str).str.zfill(2)

    merged = final_df.merge(
        metadata,
        left_on="sample_id",
        right_on=tissue,
        how="left",
    )
    merged["sample_id"] = merged["sample_id_y"]
    cols_to_drop = ["sample_id_x", "sample_id_y"]
    for extra in ["hipocamp", "adrenal", "melsa"]:
        if extra in merged.columns:
            cols_to_drop.append(extra)
    merged = merged.drop(columns=cols_to_drop)

    final_df = merged[
        ["Protein_ID", "ProteinNames", "assay", "condition", "treatment",
         "sex", "sample_id", "intensity", "tissue"]
    ]
    return final_df


def treat_spleen_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse spleen Excel/TSV data into long format.

    Columns up to ``'- Log p-value'`` are intensity columns; everything after
    is comparison statistics and is discarded.  Each intensity column name
    encodes ``<Condition>-<Treatment>_<Sex>_<SampleID>``.

    Parameters
    ----------
    df : DataFrame
        Raw spleen DataFrame (duplicates already resolved by the caller).

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns:
        ``protein_id``, ``protein_names``, ``assay``, ``peptides``,
        ``condition``, ``treatment``, ``sex``, ``sample_id``,
        ``intensity``, ``tissue``.
    """
    split_col_index = df.columns.tolist().index("- Log p-value")
    data_df = df.iloc[:, :split_col_index]

    records = []
    for col in data_df.columns[4:]:
        parts     = col.split("-")
        cond      = parts[0].strip()
        parts_2   = parts[1].split("_")
        treatment = parts_2[0].strip()
        sex       = parts_2[1].strip()
        sample_id = parts_2[2].strip()

        tmp = pd.DataFrame(
            {
                "protein_id":    data_df.iloc[:, 0].values,
                "protein_names": data_df.iloc[:, 1].values,
                "assay":         data_df.iloc[:, 2].values,
                "peptides":      data_df.iloc[:, 3].values,
                "condition":     cond,
                "treatment":     treatment,
                "sex":           sex,
                "sample_id":     sample_id,
                "intensity":     data_df[col].values,
                "tissue":        "Spleen",
            }
        )
        records.append(tmp)

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# MOFA-specific helpers
# ---------------------------------------------------------------------------

def regress_out_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``intensity_no_sex`` column by regressing sex out of intensity.

    Fits ``intensity ~ C(sex)`` per assay using OLS and stores the residuals
    in a new column.  The original ``intensity`` column is left untouched.

    Parameters
    ----------
    df : DataFrame
        Long-format DataFrame with ``assay``, ``sex``, and ``intensity``
        columns.  Modified in-place AND returned.

    Returns
    -------
    DataFrame
        Same DataFrame with ``intensity_no_sex`` column added.
    """
    df["intensity_no_sex"] = np.nan
    for assay in tqdm.tqdm(df["assay"].unique(), desc="Regressing out sex"):
        mask  = df["assay"] == assay
        sub   = df.loc[mask].copy()
        model = smf.ols("intensity ~ C(sex)", data=sub).fit()
        df.loc[mask, "intensity_no_sex"] = model.resid.values
    return df


def get_top_variable_genes(
    df: pd.DataFrame,
    n: dict[str, int],
) -> dict[str, list[str]]:
    """Select the top-N most variable genes per tissue/view.

    Parameters
    ----------
    df : DataFrame
        Long-format MOFA input DataFrame with columns ``view``, ``feature``,
        and ``value``.
    n : dict
        Mapping of ``view_name -> number_of_top_genes`` to retain.

    Returns
    -------
    dict
        ``view_name -> list of feature names`` (sorted by descending variance).
    """
    top_genes_per_tissue: dict[str, list[str]] = {}
    for tissue, k in n.items():
        df_tissue = df[df["view"] == tissue]
        gene_variances = df_tissue.groupby("feature")["value"].var()
        top_genes_per_tissue[tissue] = (
            gene_variances.sort_values(ascending=False).head(k).index.tolist()
        )
    return top_genes_per_tissue


def compute_n_genes_for_variance_threshold(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> dict[str, int]:
    """Compute the number of top-variable features that explain ``threshold``
    fraction of total variance per view.

    Parameters
    ----------
    df : DataFrame
        Long-format DataFrame with ``view``, ``feature``, ``value`` columns.
    threshold : float
        Fraction of cumulative variance to capture (e.g. 0.90 for 90%).

    Returns
    -------
    dict
        ``view_name -> n_features`` needed to reach ``threshold``.
    """
    result: dict[str, int] = {}
    for view in df["view"].unique():
        df_view = df[df["view"] == view]
        gene_variances = (
            df_view.groupby("feature")["value"]
            .var()
            .sort_values(ascending=False)
        )
        total    = gene_variances.sum()
        cumsum   = gene_variances.cumsum()
        n_needed = int((cumsum <= threshold * total).sum()) + 1
        result[view] = n_needed
    return result


def train_mofa(
    df: pd.DataFrame,
    K: int = 10,
    R_decay: float = 0.01,
    seed: int = 42,
) -> entry_point:
    """Train a MOFA+ model on the supplied long-format DataFrame.

    Parameters
    ----------
    df : DataFrame
        Long-format MOFA input with columns ``feature``, ``sample``,
        ``view``, and ``value``.
    K : int
        Number of latent factors.
    R_decay : float
        ``dropR2`` convergence threshold (passed to mofapy2).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mofapy2 entry_point
        Trained model object.  Call ``.save(outfile=...)`` on the returned
        object to persist it as an HDF5 file.
    """
    ent = entry_point()
    ent.set_data_df(df, likelihoods=["gaussian", "gaussian", "gaussian"])
    ent.set_model_options(
        factors=K,
        spikeslab_weights=True,
        ard_weights=True,
    )
    ent.set_train_options(
        convergence_mode="fast",
        dropR2=R_decay,
        gpu_mode=True,
        seed=seed,
    )
    ent.build()
    ent.run()
    return ent


def build_sample_info(
    df_spl: pd.DataFrame,
    df_hp: pd.DataFrame,
    df_sr: pd.DataFrame,
    nan_count_hp: pd.DataFrame,
    nan_count_sr: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sex and missingness metadata across three tissues into one table.

    Parameters
    ----------
    df_spl, df_hp, df_sr : DataFrame
        Long-format processed DataFrames for spleen, hippocampus, and adrenal.
        Each must have ``sample_id`` and ``sex`` columns.
    nan_count_hp, nan_count_sr : DataFrame
        Per-sample missingness counts with columns ``sample_id`` and
        ``nan_count``.

    Returns
    -------
    DataFrame
        One row per sample with columns ``sex``, ``nan_count``,
        ``sample_id``, ``group``.
    """
    def _sex_series(df: pd.DataFrame) -> pd.Series:
        s = (
            df[["sample_id", "sex"]]
            .drop_duplicates()
            .set_index("sample_id")["sex"]
            .str.lower()
        )
        s.index = s.index.astype(str)
        return s

    sex_spl = _sex_series(df_spl)
    sex_hp  = _sex_series(df_hp)
    sex_sr  = _sex_series(df_sr)

    # Combine: prefer spleen, fall back to hp, then sr.
    sex_all = (
        sex_spl.to_frame("sex_spl")
        .join(sex_hp.rename("sex_hp"), how="outer")
        .join(sex_sr.rename("sex_sr"), how="outer")
    )
    sex_all["sex"] = sex_all["sex_spl"].combine_first(sex_all["sex_hp"]).combine_first(sex_all["sex_sr"])
    sex_all = sex_all["sex"]
    sex_all.index.name = "sample"

    # Missingness: sum hp + sr counts (spleen-only samples get 0).
    nan_hp = nan_count_hp.set_index("sample_id")["nan_count"]
    nan_sr = nan_count_sr.set_index("sample_id")["nan_count"]
    nan_hp.index = nan_hp.index.astype(str)
    nan_sr.index = nan_sr.index.astype(str)
    nan_all = nan_hp.add(nan_sr, fill_value=0).rename("nan_count")
    nan_all.index.name = "sample"

    sample_info = sex_all.to_frame().join(nan_all, how="outer")
    sample_info["nan_count"] = sample_info["nan_count"].fillna(0).astype(int)
    sample_info["sample_id"] = sample_info.index.astype(str)

    # Resolve group label: prefer spleen, then hp, then sr.
    def _group_series(df: pd.DataFrame) -> pd.Series:
        s = (
            df.rename(columns={"sample_id": "sample"})
            .astype({"sample": str})
            .set_index("sample")["group"]
            .groupby("sample")
            .first()
        )
        return s

    grp_spl = _group_series(df_spl)
    grp_hp  = _group_series(df_hp)
    grp_sr  = _group_series(df_sr)

    grp_all = (
        grp_spl.to_frame("g_spl")
        .join(grp_hp.rename("g_hp"), how="outer")
        .join(grp_sr.rename("g_sr"), how="outer")
    )
    grp_all["group"] = (
        grp_all["g_spl"].combine_first(grp_all["g_hp"]).combine_first(grp_all["g_sr"])
    )
    sample_info = sample_info.join(grp_all["group"], how="left")

    return sample_info
