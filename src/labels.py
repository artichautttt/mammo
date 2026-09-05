"""Shared label vocabularies and BI-RADS helpers.

This module centralises the constants that both the data-preparation scripts and
the model code rely on, so the CSV columns, task names and severity ordering stay
consistent across the whole pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# --- BI-RADS ---------------------------------------------------------------

#: Ordering used to resolve conflicting BI-RADS sub-categories. Higher value =
#: more severe. ``pick_most_severe_subcategory`` keeps the entry with the highest
#: rank.
BIRADS_SEVERITY_ORDER: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "4A": 4,
    "4B": 5,
    "4C": 6,
    "5": 7,
    "6": 8,
}

#: The eight BI-RADS classes the model predicts (the plain "4" is dropped during
#: filtering because it is too ambiguous without a sub-category).
BIRADS_CLASSES: List[str] = ["0", "1", "2", "3", "4A", "4B", "4C", "5"]

#: Standard clinical recommendation per BI-RADS class (French).
BIRADS_RECOMMENDATION: Dict[str, str] = {
    "0": "Bilan complémentaire nécessaire (clichés additionnels ou échographie).",
    "1": "Examen normal. Poursuite du dépistage selon l'âge.",
    "2": "Findings bénins. Poursuite du dépistage de routine.",
    "3": "Probablement bénin. Surveillance rapprochée à 6 mois recommandée.",
    "4A": "Faible suspicion de malignité. Biopsie à discuter.",
    "4B": "Suspicion intermédiaire de malignité. Biopsie recommandée.",
    "4C": "Forte suspicion de malignité. Biopsie recommandée.",
    "5": "Très évocateur de malignité. Prise en charge histologique et thérapeutique.",
    "6": "Malignité prouvée par biopsie. Prise en charge oncologique.",
}

# --- MVKL manifestations -------------------------------------------------------

#: The 35 binary manifestations annotated in the MVKL dataset. Order is
#: significant: it defines the column order of the manifestation vector.
MVKL_MANIFESTATIONS: List[str] = [
    "mass",
    "spiculated_margin",
    "circumscribed_margin",
    "obscured_margin",
    "microlobulated_margin",
    "indistinct_margin",
    "round_shape",
    "oval_shape",
    "irregular_shape",
    "lobulated_shape",
    "architectural_distortion",
    "asymmetry",
    "focal_asymmetry",
    "global_asymmetry",
    "developing_asymmetry",
    "calcification",
    "amorphous_calcification",
    "coarse_heterogeneous_calcification",
    "fine_pleomorphic_calcification",
    "fine_linear_branching_calcification",
    "punctate_calcification",
    "round_calcification",
    "dystrophic_calcification",
    "milk_of_calcium_calcification",
    "vascular_calcification",
    "skin_calcification",
    "clustered_distribution",
    "linear_distribution",
    "segmental_distribution",
    "regional_distribution",
    "diffuse_distribution",
    "grouped_distribution",
    "skin_retraction",
    "nipple_retraction",
    "trabecular_thickening",
]
assert len(MVKL_MANIFESTATIONS) == 35, "expected exactly 35 MVKL manifestations"

# --- CBIS-DDSM morphological descriptors -------------------------------------

CBIS_MASS_SHAPE = [
    "ROUND", "OVAL", "IRREGULAR", "LOBULATED", "ARCHITECTURAL_DISTORTION",
    "ASYMMETRIC_BREAST_TISSUE", "FOCAL_ASYMMETRIC_DENSITY",
]
CBIS_MASS_MARGINS = [
    "CIRCUMSCRIBED", "ILL_DEFINED", "SPICULATED", "OBSCURED", "MICROLOBULATED",
]
CBIS_CALC_TYPE = [
    "AMORPHOUS", "PLEOMORPHIC", "FINE_LINEAR_BRANCHING", "PUNCTATE",
    "COARSE", "VASCULAR", "DYSTROPHIC", "LUCENT_CENTERED", "ROUND_AND_REGULAR",
    "MILK_OF_CALCIUM", "SKIN", "LARGE_RODLIKE", "EGGSHELL",
]
CBIS_CALC_DISTRIBUTION = [
    "CLUSTERED", "LINEAR", "SEGMENTAL", "REGIONAL", "DIFFUSELY_SCATTERED",
    "GROUPED",
]

#: Breast density (ACR a-d, encoded 1..4).
ACR_DENSITY_CLASSES = ["1", "2", "3", "4"]

# --- Multi-task head configuration -----------------------------------------

#: Maps a task name to its number of output classes. Consumed by
#: ``DualViewCLIP`` to build one ``nn.Linear`` head per task, and by the loss.
TASK_NUM_CLASSES: Dict[str, int] = {
    "birads": len(BIRADS_CLASSES),
    "acr_density": len(ACR_DENSITY_CLASSES),
    "mass_shape": len(CBIS_MASS_SHAPE),
    "mass_margins": len(CBIS_MASS_MARGINS),
    "calc_type": len(CBIS_CALC_TYPE),
    "calc_distribution": len(CBIS_CALC_DISTRIBUTION),
    # each manifestation is a binary head
    **{f"mf_{name}": 2 for name in MVKL_MANIFESTATIONS},
}

#: Label used for "annotation absent in this source" — ignored by the loss.
IGNORE_INDEX = -100


def birads_rank(subcategory: Optional[str]) -> int:
    """Return the severity rank of a BI-RADS (sub-)category.

    Unknown / missing values get ``-1`` so they never win a "most severe"
    comparison.
    """
    if subcategory is None:
        return -1
    return BIRADS_SEVERITY_ORDER.get(str(subcategory).strip().upper(), -1)


def pick_most_severe_subcategory(values: List[Optional[str]]) -> Optional[str]:
    """Pick the most severe BI-RADS sub-category among ``values``.

    Used when several annotations disagree on the sub-category of the same case.
    Returns ``None`` if no value is recognised.
    """
    best: Optional[str] = None
    best_rank = -1
    for value in values:
        rank = birads_rank(value)
        if rank > best_rank:
            best_rank = rank
            best = None if value is None else str(value).strip().upper()
    return best
