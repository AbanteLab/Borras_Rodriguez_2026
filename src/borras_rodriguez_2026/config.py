"""
config.py — All study-specific parameters for the repro_paper pipeline.

Edit the paths in the PATHS section to match your environment.
All other sections control analysis parameters.
"""

from pathlib import Path
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

# Root directory where raw proteomics data files and metadata live.
DATA_DIR = Path("/pool01/data/private/giralt_lab/proteomics_repro_fall_2025")

# Root directory where all pipeline outputs are written.
RESULTS_DIR = Path("/pool01/projects/giralt_lab/proteomics_repro_fall_2025/results_new_repos")

# Directory containing MSigDB .gmt gene-set files.
GMT_DIR = Path("/pool01/code/projects/puigdellivol_lab/proteomics_ad_fall_25/gene_sets")

# ---------------------------------------------------------------------------
# DERIVED PATHS  (do not edit unless the layout changes)
# ---------------------------------------------------------------------------

# Processed long-format CSVs are written alongside the raw data files.
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

GLM_RESULTS_DIR    = RESULTS_DIR / "glm_results"
GSEA_RESULTS_DIR   = RESULTS_DIR / "gsea_results"
MOFA_DIR           = RESULTS_DIR / "mofa"
FIGURES_DIR        = RESULTS_DIR / "paper_figures"

# ---------------------------------------------------------------------------
# RAW DATA FILE NAMES
# ---------------------------------------------------------------------------

HP_RAW_FILE      = DATA_DIR / "hipocamp.tsv"
ADRENAL_RAW_FILE = DATA_DIR / "adrenal.tsv"
SPLEEN_RAW_FILE  = DATA_DIR / "spleen.tsv"
METADATA_FILE    = DATA_DIR / "metadata.csv"

# ---------------------------------------------------------------------------
# SAMPLE EXCLUSIONS
# ---------------------------------------------------------------------------

# Sample IDs to drop across all tissues.
SAMPLES_TO_EXCLUDE = [1884]

# Spleen-only: samples that exist in spleen but not in hp/adrenal are noted
# during preprocessing; set REMOVE_SPLEEN_ONLY_SAMPLES = True to drop them
# from the MOFA input (they are always kept for per-tissue GLM/GSEA).
REMOVE_SPLEEN_ONLY_SAMPLES = False

# ---------------------------------------------------------------------------
# GROUP / CATEGORY ENCODING
# ---------------------------------------------------------------------------

# Canonical mapping applied to raw treatment/condition labels in the spleen
# data (which arrives with abbreviated codes).
SPLEEN_TREATMENT_MAP = {"VEH": "vehicle", "DOX": "dox"}
SPLEEN_CONDITION_MAP = {"CNT": "ctrl", "CUMS": "cus"}

# Ordered categories used to fix the reference level for the GLM.
# "ctrl_vehicle" is the reference (first element).
GROUP_CATEGORIES = ["ctrl_vehicle", "cus_dox", "cus_vehicle"]
SEX_CATEGORIES   = ["female", "male"]   # "female" is reference

# ---------------------------------------------------------------------------
# GLM
# ---------------------------------------------------------------------------

GLM_RESPONSE_VAR = "intensity"

# Patsy formula — identical for all three tissues.
GLM_FORMULA = "intensity ~ C(group) * C(sex) + nan_count"

# Family and link function.  Change the strings to switch model.
# Supported family strings : "Gaussian", "Gamma", "Poisson", "NegativeBinomial"
# Supported link strings   : "identity", "log", "logit", "sqrt"
GLM_FAMILY_NAME = "Gaussian"
GLM_LINK_NAME   = "identity"

def get_glm_family():
    """Return a statsmodels family instance from the config strings."""
    _families = {
        "Gaussian":        sm.families.Gaussian,
        "Gamma":           sm.families.Gamma,
        "Poisson":         sm.families.Poisson,
        "NegativeBinomial": sm.families.NegativeBinomial,
    }
    _links = {
        "identity": sm.families.links.identity(),
        "log":      sm.families.links.log(),
        "logit":    sm.families.links.logit(),
        "sqrt":     sm.families.links.sqrt(),
    }
    family_cls  = _families[GLM_FAMILY_NAME]
    link_obj    = _links[GLM_LINK_NAME]
    return family_cls(link=link_obj)

# ---------------------------------------------------------------------------
# CONTRASTS
# ---------------------------------------------------------------------------

CONTRASTS = {
    "cus_vehicle_vs_ctrl_vehicle_male": {
        "coefs": {
            "C(group)[T.cus_vehicle]":            1,
            "C(group)[T.cus_vehicle]:C(sex)[T.male]": 1,
        }
    },
    "cus_dox_vs_ctrl_vehicle_male": {
        "coefs": {
            "C(group)[T.cus_dox]":              1,
            "C(group)[T.cus_dox]:C(sex)[T.male]":   1,
        }
    },
    "cus_dox_vs_cus_vehicle_male": {
        "coefs": {
            "C(group)[T.cus_dox]":                    1,
            "C(group)[T.cus_dox]:C(sex)[T.male]":        1,
            "C(group)[T.cus_vehicle]":               -1,
            "C(group)[T.cus_vehicle]:C(sex)[T.male]":   -1,
        }
    },
    "cus_dox_vs_cus_vehicle_female": {
        "coefs": {
            "C(group)[T.cus_dox]":     1,
            "C(group)[T.cus_vehicle]": -1,
        }
    },
    "cus_vehicle_vs_ctrl_vehicle_female": {
        "coefs": {
            "C(group)[T.cus_vehicle]": 1,
        }
    },
    "cus_dox_vs_ctrl_vehicle_female": {
        "coefs": {
            "C(group)[T.cus_dox]": 1,
        }
    },
}

# ---------------------------------------------------------------------------
# GSEA
# ---------------------------------------------------------------------------

# Gene-set collections to test.  Each name is formatted into GMT_FILENAME_TEMPLATE.
GSEA_COLLECTIONS = [
    "m2.cgp",
    "m2.cp.biocarta",
    "m2.cp.reactome",
    "m2.cp.wikipathways",
    "m5.go.bp",
    "m5.go.cc",
    "m5.go.mf",
    "m7.all",
    "m8.all",
    "mh.all",
]

# Filename pattern for .gmt files inside GMT_DIR.
GMT_FILENAME_TEMPLATE = "{collection}.v2025.1.Mm.symbols.gmt"

GSEA_FDR_CUTOFF     = 0.25   # pathway significance threshold
GSEA_PERMUTATIONS   = 10_000
GSEA_MIN_SIZE       = 2
GSEA_MAX_SIZE       = 50
GSEA_SEED           = 42
GSEA_MIN_OVERLAP    = 2      # min background genes a set must contain

# Collections used for MOFA-factor GSEA (subset of above).
MOFA_GSEA_COLLECTIONS = [
    "m2.cgp",
    "m2.cp.biocarta",
    "m2.cp.reactome",
    "m2.cp.wikipathways",
    "m7.all",
    "m8.all",
    "mh.all",
]

# ---------------------------------------------------------------------------
# MOFA
# ---------------------------------------------------------------------------

MOFA_K          = 11     # number of latent factors
MOFA_SEED       = 0
MOFA_R_DECAY    = 0.01   # dropR2 convergence threshold

# Fraction of per-tissue variance captured by the top-variable genes fed to MOFA.
MOFA_VAR_EXPL_THRESHOLD = 0.90

# Tissue label used as the 'view' column in the MOFA-format dataframe.
# Keys must match the 'tissue' values assigned during preprocessing.
TISSUE_VIEW_NAMES = {"hipocamp": "hp", "adrenal": "sr", "spleen": "Spleen"}

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

GROUP_COLORS = {
    "ctrl_vehicle": (0.831, 0.831, 0.831),
    "cus_dox":      (1.000, 0.501, 0.006),
    "cus_vehicle":  (1.000, 0.753, 0.502),
}

SEX_COLORS = {
    "male":   "tab:blue",
    "female": "tab:orange",
}

VOLCANO_FDR_THRESHOLD  = 0.05
VOLCANO_COEF_THRESHOLD = 0.5
