"""
pathway_classes.py — Keyword-based pathway class definitions for this paper.

Used by utils_figures.py to assign a biological class label to each GSEA term.
This is entirely study-specific and lives only in repro_paper.
"""

PATHWAY_CLASSES: dict[str, list[str]] = {
    "Immune": [
        "IMMUNE", "INTERFERON", "INFLAMM", "INFLAM", "INFLAMMATION",
        "INFLAMMATORY", "CYTOKINE", "CHEMOKINE", "LEUKOCYTE", "LYMPHOCYTE",
        "MACROPHAGE", "MICROGLIA", "MONOCYTE", "NEUTROPHIL", "T_CELL",
        "B_CELL", "NK_CELL", "ANTIGEN", "MHC", "COMPLEMENT", "ALLOGRAFT",
        "GRAFT", "JAK_STAT", "NFKB", "TNFA", "IL1", "IL2", "IL6", "IL10",
        "IL17", "IFNG", "IFNA",
    ],
    "Aging_Senescence": [
        "AGING", "AGEING", "SENESC", "LONGEVITY", "LIFESPAN",
        "CELLULAR_SENESCENCE", "REPLICATIVE_SENESCENCE", "TELOMERE",
        "DNA_DAMAGE", "DNA_REPAIR", "P53", "SIRT", "FOXO", "AUTOPHAG",
        "MTOR", "OXIDATIVE_STRESS", "PROTEOSTASIS", "UNFOLDED_PROTEIN_RESPONSE",
        "UPR", "GENOME_INSTABILITY", "EPIGENETIC",
    ],
    "Mitochondrial_Metabolism": [
        "OXIDATIVE_PHOSPHORYLATION", "RESPIRATORY_CHAIN", "ELECTRON_TRANSPORT",
        "MITOCHONDR", "ATP_SYNTHESIS", "TCA_CYCLE", "CITRIC_ACID", "GLYCOLYSIS",
        "FATTY_ACID", "LIPID_METABOLISM", "CHOLESTEROL", "PEROXISOME",
        "HEME_METABOLISM",
    ],
    "Stress_Response": [
        "HYPOXIA", "REACTIVE_OXYGEN", "ROS", "HEAT_SHOCK",
        "UNFOLDED_PROTEIN_RESPONSE", "UPR", "ER_STRESS", "XENOBIOTIC",
        "DETOXIFICATION", "STRESS_RESPONSE", "DNA_DAMAGE_RESPONSE",
        "OXIDATIVE_STRESS",
    ],
    "Synaptic_Neuronal": [
        "SYNAP", "NEURON", "AXON", "DENDRITE", "NEUROTRANSMITTER",
        "GLUTAMATE", "GABA", "DOPAMINE", "SEROTONIN", "LONG_TERM_POTENTIATION",
        "LONG_TERM_DEPRESSION", "VESICLE", "CALCIUM_SIGNALING",
    ],
    "Neurogenesis_Plasticity": [
        "NEUROGENESIS", "NEURON_DIFFERENTIATION", "AXON_GUIDANCE",
        "DENDRITE_DEVELOPMENT", "SYNAPTOGENESIS", "PLASTICITY", "STEM_CELL",
        "PROGENITOR",
    ],
    "Glial_Activation": [
        "MICROGLIA", "ASTROCYTE", "OLIGODENDROCYTE", "GLIAL", "MYELIN",
        "NEUROINFLAMMATION",
    ],
    "Cell_Death": [
        "APOPTOSIS", "NECROPTOSIS", "FERROPTOSIS", "PYROPTOSIS",
        "CELL_DEATH", "CASPASE",
    ],
    "Cell_Cycle_Proliferation": [
        "CELL_CYCLE", "G2M", "E2F", "MYC", "MITOTIC", "PROLIFERATION",
        "CHECKPOINT",
    ],
    "ECM_Adhesion": [
        "EXTRACELLULAR_MATRIX", "ECM", "COLLAGEN", "INTEGRIN", "ADHESION",
        "ANGIOGENESIS", "EPITHELIAL_MESENCHYMAL_TRANSITION", "EMT",
    ],
    "Development_Organogenesis": [
        "ORGANOGENESIS", "MORPHOGENESIS", "EMBRYONIC_DEVELOPMENT",
        "PATTERN_SPECIFICATION", "TISSUE_DEVELOPMENT", "ORGAN_DEVELOPMENT",
        "CELL_FATE", "DIFFERENTIATION", "WNT", "HEDGEHOG", "NOTCH", "BMP",
        "TGF_BETA", "DEVELOPMENTAL_PROCESS", "ANATOMICAL_STRUCTURE_DEVELOPMENT",
    ],
}
