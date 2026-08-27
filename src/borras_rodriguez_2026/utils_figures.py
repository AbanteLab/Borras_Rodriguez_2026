"""
utils_figures.py — Paper-specific helpers for generating Figures 4 and 5.

All functions here are specific to this study's layout of results files and
the visual style of the two paper figures.  Generic plotting utilities
(plot_pca, make_volcano_plots, plot_rank_change) are imported from
proteomics_analysis.plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import mofax as mfx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from .config import (
    FIGURES_DIR,
    GLM_FAMILY_NAME,
    GLM_LINK_NAME,
    GLM_RESULTS_DIR,
    GSEA_RESULTS_DIR,
    MOFA_DIR,
    MOFA_K,
    MOFA_R_DECAY,
    MOFA_SEED,
    MOFA_VAR_EXPL_THRESHOLD,
    RESULTS_DIR,
)
from .pathway_classes import PATHWAY_CLASSES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pathway class assignment
# ---------------------------------------------------------------------------

def assign_pathway_class(term: str) -> str:
    """Map a GSEA term name to one of the PATHWAY_CLASSES buckets.

    Returns ``"Other"`` when no keyword matches.
    """
    term_upper = term.upper().replace(" ", "_").replace("-", "_")
    for cls, keywords in PATHWAY_CLASSES.items():
        for kw in keywords:
            if kw in term_upper:
                return cls
    return "Other"


def define_pathway_class_palette(class_names) -> dict[str, tuple]:
    """Assign a distinct matplotlib color to each pathway class."""
    cmap    = plt.get_cmap("tab20")
    classes = list(class_names) + ["Other"]
    return {cls: cmap(i / max(len(classes) - 1, 1)) for i, cls in enumerate(classes)}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_paper_paths() -> dict[str, Path]:
    """Return the canonical path dict used by figure-generation scripts."""
    return {
        "results_dir":  RESULTS_DIR,
        "figures_dir":  FIGURES_DIR,
        "glm_dir":      GLM_RESULTS_DIR,
        "gsea_dir":     GSEA_RESULTS_DIR,
        "mofa_dir":     MOFA_DIR,
    }


def get_fig5_paths() -> dict[str, Path]:
    return get_paper_paths()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_fig4_glm_results(
    results_dir: Path,
    tissues: list[str],
    family_name: str = GLM_FAMILY_NAME,
    link_name: str = GLM_LINK_NAME,
) -> dict[str, pd.DataFrame]:
    """Load contrasts DataFrames for all tissues.

    Returns
    -------
    dict
        ``tissue -> contrasts DataFrame``
    """
    all_contrasts: dict[str, pd.DataFrame] = {}
    for tissue in tissues:
        path = (
            GLM_RESULTS_DIR
            / tissue
            / f"{tissue}_glm_contrasts_per_protein_with_fdr.csv"
        )
        if not path.exists():
            log.warning("GLM contrasts not found: %s", path)
            continue
        df = pd.read_csv(path, index_col=0)
        df["tissue"] = tissue
        all_contrasts[tissue] = df
    return all_contrasts


def load_fig4_gsea_results(
    results_dir: Path,
    tissues: list[str],
    collections: list[str],
    family_name: str = GLM_FAMILY_NAME,
    link_name: str = GLM_LINK_NAME,
) -> pd.DataFrame:
    """Concatenate all per-tissue, per-contrast GSEA combined-collection
    result files into a single DataFrame.

    Returns
    -------
    DataFrame with columns from the GSEA result files plus ``tissue`` and
    ``contrast``.
    """
    frames: list[pd.DataFrame] = []
    for tissue in tissues:
        gsea_base = GSEA_RESULTS_DIR / tissue / "contrasts"
        if not gsea_base.exists():
            log.warning("GSEA contrasts dir not found: %s", gsea_base)
            continue
        for contrast_dir in sorted(gsea_base.iterdir()):
            if not contrast_dir.is_dir():
                continue
            contrast = contrast_dir.name
            csv = contrast_dir / f"{contrast}_gsea_combined_collections_full_results.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv, index_col=0)
            df["tissue"]   = tissue
            df["contrast"] = contrast
            frames.append(df)
    if not frames:
        log.warning("No GSEA contrast results found.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_gsea_results(
    mofa_dir: Path,
    factors: list[str],
    collections: list[str],
    tissue_order: list[str],
) -> pd.DataFrame:
    """Load GSEA results for MOFA factors across all tissue views."""
    frames: list[pd.DataFrame] = []
    gsea_base = mofa_dir / "gsea"
    for factor in factors:
        for view in tissue_order:
            csv = gsea_base / factor / view / f"{factor}_{view}_gsea_combined_collections_full_results.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv, index_col=0)
            df["factor"] = factor
            df["view"]   = view
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_mofa_model(
    mofa_dir: Path,
    chosen_k: int = MOFA_K,
    seed: int = MOFA_SEED,
    r_decay: float = MOFA_R_DECAY,
    var_expl_by_n_genes: float = MOFA_VAR_EXPL_THRESHOLD,
) -> mfx.mofa_model:
    """Load a trained MOFA model from the expected HDF5 path."""
    model_path = (
        mofa_dir
        / f"mofa_model_K{chosen_k}_seed{seed}_Rdecay{r_decay}"
          f"_varexpl{var_expl_by_n_genes}.hdf5"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"MOFA model not found: {model_path}\n"
            "Run run_mofa.py first."
        )
    return mfx.mofa_model(str(model_path))


def attach_sample_metadata(
    mofa_model: mfx.mofa_model,
    sample_info_path: Path,
) -> mfx.mofa_model:
    """Join sex and group from sample_info.csv into model.metadata."""
    si = pd.read_csv(sample_info_path).set_index("sample_id")
    si.index = si.index.astype(str)
    for col in ["sex", "group", "nan_count"]:
        if col in si.columns and col not in mofa_model.metadata.columns:
            mofa_model.metadata = mofa_model.metadata.join(
                si[col], on="sample", how="left"
            )
    return mofa_model


# ---------------------------------------------------------------------------
# Figure 4 plotting
# ---------------------------------------------------------------------------

def plot_fig4_top_gene_barplots(
    all_contrasts: dict[str, pd.DataFrame],
    gsea_results_all: pd.DataFrame,
    class_color_map: dict[str, tuple],
    figures_dir: Path,
    ntop_genes_per_coef: int = 20,
) -> None:
    """Save bar plots showing top proteins per contrast, colored by pathway class."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    for tissue, contrasts_df in all_contrasts.items():
        contrast_cols = sorted(
            {c.rsplit("_", 1)[0] for c in contrasts_df.columns if c.endswith("_coef")}
        )
        for contrast in contrast_cols:
            coef_col = f"{contrast}_coef"
            fdr_col  = f"{contrast}_fdr"
            if coef_col not in contrasts_df.columns:
                continue

            top = (
                contrasts_df[[coef_col]]
                .dropna()
                .reindex(contrasts_df[coef_col].abs().sort_values(ascending=False).index)
                .head(ntop_genes_per_coef)
            )
            if top.empty:
                continue

            # Assign pathway class to each gene if GSEA results are available.
            if not gsea_results_all.empty and "Term" in gsea_results_all.columns:
                tissue_gsea = gsea_results_all[
                    (gsea_results_all["tissue"] == tissue)
                    & (gsea_results_all["contrast"] == contrast)
                ]
                # Use the most significant pathway per gene as a proxy.
                gene_class = {
                    gene: assign_pathway_class(
                        tissue_gsea.sort_values("FDR q-val")
                        .iloc[0]["Term"]
                    )
                    if not tissue_gsea.empty else "Other"
                    for gene in top.index
                }
            else:
                gene_class = {gene: "Other" for gene in top.index}

            colors = [
                class_color_map.get(gene_class.get(g, "Other"), (0.7, 0.7, 0.7))
                for g in top.index
            ]

            fig, ax = plt.subplots(figsize=(8, max(4, ntop_genes_per_coef * 0.35)))
            ax.barh(
                top.index[::-1],
                top[coef_col].values[::-1],
                color=colors[::-1],
            )
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Coefficient")
            ax.set_title(f"{tissue} — {contrast}\nTop {ntop_genes_per_coef} proteins")
            ax.set_ylabel("Protein")

            # Legend for pathway classes that appear.
            used_classes = sorted(set(gene_class.values()))
            patches = [
                mpatches.Patch(color=class_color_map.get(c, (0.7, 0.7, 0.7)), label=c)
                for c in used_classes
            ]
            ax.legend(
                handles=patches, loc="lower right", fontsize=7, frameon=False
            )

            plt.tight_layout()
            out = figures_dir / f"fig4_top_genes_{tissue}_{contrast}.pdf"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved %s", out)


def plot_fig4_sankey(
    gsea_results_all: pd.DataFrame,
    figures_dir: Path,
    fdr_cutoff: float = 0.25,
    top_n: int = 20,
) -> None:
    """Save a dot-plot summary of top GSEA terms per tissue colored by
    pathway class (used in lieu of a full Sankey for static PDF output)."""
    if gsea_results_all.empty:
        log.warning("No GSEA results to plot for Sankey/dotplot.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    sig = gsea_results_all[gsea_results_all["FDR q-val"] < fdr_cutoff].copy()
    if sig.empty:
        log.warning("No significant GSEA terms at FDR < %.2f for Sankey plot.", fdr_cutoff)
        return

    sig["NES_abs"] = sig["NES"].abs()
    top_terms = (
        sig.sort_values("NES_abs", ascending=False)
        .drop_duplicates("Term")
        .head(top_n)
    )

    tissues  = sig["tissue"].unique() if "tissue" in sig.columns else ["all"]
    n_panels = len(tissues)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, max(5, top_n * 0.4)), sharey=False
    )
    if n_panels == 1:
        axes = [axes]

    for ax, tissue in zip(axes, tissues):
        sub = sig[sig["tissue"] == tissue] if "tissue" in sig.columns else sig
        sub = (
            sub.sort_values("NES_abs", ascending=False)
            .drop_duplicates("Term")
            .head(top_n)
        )
        if sub.empty:
            ax.set_title(tissue)
            continue
        colors = [
            class_color_map.get(c, (0.7, 0.7, 0.7))
            for c in sub["Pathway_Class"]
        ] if "Pathway_Class" in sub.columns and "class_color_map" in dir() else ["steelblue"] * len(sub)

        ax.barh(sub["Term"][::-1], sub["NES"].values[::-1])
        ax.axvline(0, color="black", lw=0.7)
        ax.set_xlabel("NES")
        ax.set_title(tissue)

    plt.tight_layout()
    out = figures_dir / "fig4_gsea_sankey_summary.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 5 plotting
# ---------------------------------------------------------------------------

def calculate_variance_explained_by_group(
    mofa_model: mfx.mofa_model,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Compute R² of each factor grouped by sex using model's built-in method."""
    r2 = mofa_model.calculate_variance_explained(
        factors=list(factors.columns),
        group_label="sex",
        per_factor=True,
    )
    return r2


def plot_variance_explained_heatmap(
    r2: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save heatmap of variance explained per factor and view."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if r2.empty:
        log.warning("Empty R² DataFrame, skipping variance-explained heatmap.")
        return

    r2_pivot = r2.pivot_table(index="Factor", columns="View", values="R2", aggfunc="mean")
    factor_order = r2_pivot.mean(axis=1).sort_values().index
    r2_pivot = r2_pivot.loc[factor_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(r2_pivot, cmap="Blues", cbar_kws={"label": "R²"}, ax=ax)
    ax.set_title("Variance Explained by Factor and View")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_factor_values_by_group(
    mofa_model: mfx.mofa_model,
    factors_to_plot: list[str],
    out_path: Path,
    group_colors: dict[str, tuple],
) -> None:
    """Violin + strip plot of selected factor values split by group and sex."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    group_order = ["ctrl_vehicle", "cus_vehicle", "cus_dox"]
    sex_order   = ["male", "female"]

    n = len(factors_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n), squeeze=False)

    for ax, factor in zip(axes[:, 0], factors_to_plot):
        fvals = mofa_model.fetch_values(factor, unique=True)
        df_plot = pd.DataFrame(
            {
                factor:  fvals[factor],
                "Group": mofa_model.metadata["group"],
                "Sex":   mofa_model.metadata["sex"],
            }
        )
        df_plot["Group"] = pd.Categorical(df_plot["Group"], categories=group_order, ordered=True)

        palette = {g: group_colors.get(g, (0.5, 0.5, 0.5)) for g in group_order}
        sns.violinplot(
            x="Sex", y=factor, hue="Group", data=df_plot,
            order=sex_order, palette=palette, ax=ax, inner=None,
        )
        sns.stripplot(
            x="Sex", y=factor, hue="Group", data=df_plot,
            order=sex_order, palette=palette, ax=ax, dodge=True,
            color="black", alpha=0.5, size=3,
        )
        ax.set_title(factor)
        handles, labels = ax.get_legend_handles_labels()
        n_groups = len(group_order)
        ax.legend(handles[:n_groups], labels[:n_groups], title="Group", frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def compute_factor_pca(mofa_model: mfx.mofa_model):
    """Run PCA on the factor score matrix.

    Returns
    -------
    pca_df : DataFrame  (samples × ['PC1', 'PC2'])
    pca    : fitted sklearn PCA object
    """
    factors_matrix = mofa_model.get_factors(df=True)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(factors_matrix.fillna(0).values)
    pca_df = pd.DataFrame(coords, index=factors_matrix.index, columns=["PC1", "PC2"])
    pca_df = pca_df.join(mofa_model.metadata[["group", "sex"]], how="left")
    return pca_df, pca


def plot_factor_pca(
    pca_df: pd.DataFrame,
    pca,
    out_path: Path,
    group_colors: dict[str, tuple],
) -> None:
    """Scatter PCA of factor scores, colored by group."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    for grp, sub in pca_df.groupby("group"):
        color = group_colors.get(grp, (0.5, 0.5, 0.5))
        ax.scatter(sub["PC1"], sub["PC2"], label=grp, color=color, s=40, alpha=0.85)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title("PCA of MOFA factor scores — colored by group")
    ax.legend(title="Group", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_factor_pca_by_sex(pca_df: pd.DataFrame, pca) -> None:
    """Show (but do not save) PCA of factor scores colored by sex."""
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"male": "tab:blue", "female": "tab:orange"}
    for sex, sub in pca_df.groupby("sex"):
        ax.scatter(sub["PC1"], sub["PC2"], label=sex,
                   color=palette.get(sex, "gray"), s=40, alpha=0.85)
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title("PCA of MOFA factors — colored by sex")
    ax.legend(title="Sex", frameon=False)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def prepare_top_pathway_heatmap(
    gsea_results: pd.DataFrame,
    factor: str,
    tissue_order: list[str],
    top_n: int = 10,
    class_color_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Pivot GSEA NES values into a heatmap-ready matrix.

    Returns
    -------
    data_plot : DataFrame  (terms × views)
    col_colors : Series   (view -> color string)
    legend_handles : list of Patch objects for the pathway-class legend
    """
    sub = gsea_results[gsea_results["factor"] == factor].copy()
    if sub.empty:
        return pd.DataFrame(), pd.Series(dtype=str), []

    # Top N terms by absolute NES across all views.
    top_terms = (
        sub.groupby("Term")["NES"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    sub = sub[sub["Term"].isin(top_terms)]

    data_plot = sub.pivot_table(index="Term", columns="view", values="NES", aggfunc="mean")
    data_plot = data_plot.reindex(columns=tissue_order)

    # Column colors (by tissue).
    tissue_palette = dict(zip(tissue_order, sns.color_palette("Set2", len(tissue_order))))
    col_colors     = pd.Series(
        [tissue_palette.get(v, "gray") for v in data_plot.columns],
        index=data_plot.columns,
    )

    # Legend for pathway classes.
    legend_handles = []
    if class_color_map:
        for cls, color in class_color_map.items():
            legend_handles.append(
                mpatches.Patch(color=color, label=cls)
            )

    return data_plot, col_colors, legend_handles


def plot_top_pathway_heatmap(
    data_plot: pd.DataFrame,
    col_colors: pd.Series,
    legend_handles: list,
    out_path: Path,
) -> None:
    """Save clustermap heatmap of top pathway NES values."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if data_plot.empty:
        log.warning("Empty data for top-pathway heatmap, skipping.")
        return

    g = sns.clustermap(
        data_plot.fillna(0),
        cmap="RdBu_r",
        center=0,
        col_colors=col_colors,
        figsize=(max(6, len(data_plot.columns) * 1.5), max(5, len(data_plot) * 0.5)),
        yticklabels=True,
        xticklabels=True,
        row_cluster=True,
        col_cluster=False,
    )
    g.ax_heatmap.set_xlabel("Tissue View")
    g.ax_heatmap.set_ylabel("Pathway")
    g.fig.suptitle("Top pathways — NES", y=1.02)

    if legend_handles:
        g.ax_heatmap.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
            frameon=False,
            fontsize=7,
            title="Pathway class",
        )

    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    log.info("Saved %s", out_path)
