"""Assemblage du rapport mammographique : traduction des prédictions en langage
naturel, appel à l'API Groq (LLaMA) pour la synthèse, export PDF.

PDF nommé ``rapport_birads_{NomPatient}.pdf``.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .labels import BIRADS_RECOMMENDATION
from .two_view_model import Prediction, load_model, predict

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# --- Traduction des manifestations en langage naturel (FR) ------------------

MANIFESTATION_FR: Dict[str, str] = {
    "mass": "masse",
    "spiculated_margin": "contours spiculés",
    "circumscribed_margin": "contours circonscrits",
    "obscured_margin": "contours masqués",
    "microlobulated_margin": "contours microlobulés",
    "indistinct_margin": "contours indistincts",
    "round_shape": "forme ronde",
    "oval_shape": "forme ovale",
    "irregular_shape": "forme irrégulière",
    "lobulated_shape": "forme lobulée",
    "architectural_distortion": "distorsion architecturale",
    "asymmetry": "asymétrie",
    "focal_asymmetry": "asymétrie focale",
    "global_asymmetry": "asymétrie globale",
    "developing_asymmetry": "asymétrie évolutive",
    "calcification": "calcifications",
    "amorphous_calcification": "calcifications amorphes",
    "coarse_heterogeneous_calcification": "calcifications grossières hétérogènes",
    "fine_pleomorphic_calcification": "calcifications fines pléomorphes",
    "fine_linear_branching_calcification": "calcifications fines linéaires branchées",
    "punctate_calcification": "calcifications punctiformes",
    "round_calcification": "calcifications rondes",
    "dystrophic_calcification": "calcifications dystrophiques",
    "milk_of_calcium_calcification": "calcifications de type lait calcique",
    "vascular_calcification": "calcifications vasculaires",
    "skin_calcification": "calcifications cutanées",
    "clustered_distribution": "distribution en amas",
    "linear_distribution": "distribution linéaire",
    "segmental_distribution": "distribution segmentaire",
    "regional_distribution": "distribution régionale",
    "diffuse_distribution": "distribution diffuse",
    "grouped_distribution": "distribution groupée",
    "skin_retraction": "rétraction cutanée",
    "nipple_retraction": "rétraction mamelonnaire",
    "trabecular_thickening": "épaississement trabéculaire",
}

ACR_DENSITY_FR = {
    "1": "ACR a — seins presque entièrement graisseux",
    "2": "ACR b — densités fibroglandulaires éparses",
    "3": "ACR c — seins denses de façon hétérogène",
    "4": "ACR d — seins extrêmement denses",
}


@dataclass
class PatientInfo:
    """Informations administratives minimales du patient."""

    name: str
    age: Optional[int] = None
    exam_date: str = date.today().isoformat()
    history: Optional[str] = None

    @property
    def initials(self) -> str:
        return "".join(part[0].upper() for part in self.name.split() if part)


def manifestations_to_sentences(prediction: Prediction) -> List[str]:
    """Traduit les manifestations actives (déjà seuillées) en phrases FR."""
    sentences = []
    for name in prediction.active_manifestations:
        label = MANIFESTATION_FR.get(name, name.replace("_", " "))
        prob = prediction.manifestations.get(name, 0.0)
        sentences.append(f"Présence de {label} (probabilité {prob:.0%}).")
    if not sentences:
        sentences.append("Aucune manifestation significative retenue au seuil courant.")
    return sentences


def clinical_recommendation(birads_class: str) -> str:
    """Recommandation clinique standard associée au score BI-RADS."""
    return BIRADS_RECOMMENDATION.get(birads_class, "Corrélation clinique recommandée.")


# --- Synthèse LLM (Groq / LLaMA) ------------------------------------------

def _fallback_narrative(
    patient: PatientInfo, prediction: Prediction, similar_summaries: List[str]
) -> str:
    """Texte de repli si l'API Groq n'est pas disponible."""
    findings = " ".join(manifestations_to_sentences(prediction))
    return (
        f"Mammographie bilatérale analysée pour {patient.name}. "
        f"Classification BI-RADS retenue : {prediction.birads_class}. "
        f"{findings} "
        f"Densité mammaire estimée : {ACR_DENSITY_FR.get(prediction.acr_density_class, 'non déterminée')}. "
        f"{clinical_recommendation(prediction.birads_class)} "
        "Cette interprétation automatisée doit être confirmée par un radiologue."
    )


def synthesize_with_groq(
    patient: PatientInfo,
    prediction: Prediction,
    similar_summaries: List[str],
    model_name: Optional[str] = None,
) -> str:
    """Fusionne prédictions + cas similaires en un paragraphe médical fluide via
    l'API Groq. Retombe sur ``_fallback_narrative`` en l'absence de clé / lib."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_narrative(patient, prediction, similar_summaries)
    try:
        from groq import Groq
    except Exception:
        return _fallback_narrative(patient, prediction, similar_summaries)

    model_name = model_name or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    findings = "\n".join(f"- {s}" for s in manifestations_to_sentences(prediction))
    similar_block = "\n".join(f"- {s}" for s in similar_summaries) or "- (aucun)"

    prompt = f"""Tu es radiologue. Rédige en français un unique paragraphe de compte rendu
mammographique, factuel et sobre, à partir des éléments suivants. Termine par un
rappel explicite qu'un radiologue doit confirmer l'interprétation.

Patient : {patient.name}, {patient.age or 'âge non précisé'} ans.
Antécédents : {patient.history or 'aucun renseigné'}.
BI-RADS prédit : {prediction.birads_class}.
Densité mammaire : {ACR_DENSITY_FR.get(prediction.acr_density_class, 'non déterminée')}.
Caractéristiques retenues :
{findings}
Cas antérieurs similaires :
{similar_block}
Recommandation standard : {clinical_recommendation(prediction.birads_class)}
"""
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=450,
    )
    return completion.choices[0].message.content.strip()


# --- Export PDF -----------------------------------------------------------

_BORDEAUX = "#6b1f3a"
_VIOLET = "#4a2c63"

_BIRADS_COLORS = {
    "0": "#8a8a8a", "1": "#2e7d32", "2": "#2e7d32", "3": "#f9a825",
    "4A": "#ef6c00", "4B": "#e65100", "4C": "#c62828", "5": "#b71c1c",
}


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name) or "Patient"


def export_pdf(
    patient: PatientInfo,
    prediction: Prediction,
    narrative: str,
    similar_summaries: List[str],
    out_dir: str | Path = ".",
) -> Path:
    """Écrit ``rapport_birads_{NomPatient}.pdf`` et retourne son chemin."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rapport_birads_{_safe_filename(patient.name)}.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H", parent=styles["Heading2"], textColor=colors.HexColor(_BORDEAUX)))
    styles.add(ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=colors.grey))

    story = []
    # -- en-tête
    story.append(Paragraph("Rapport de diagnostic mammographique par IA", styles["Title"]))
    story.append(Paragraph(
        f"Patient (initiales) : <b>{patient.initials}</b> — "
        f"Document confidentiel — Édité le {date.today().isoformat()} — "
        f"Examen du {patient.exam_date}",
        styles["Small"],
    ))
    story.append(Spacer(1, 8 * mm))

    # -- informations patient
    story.append(Paragraph("Informations patient", styles["H"]))
    info = [
        ["Nom", patient.name],
        ["Âge", str(patient.age) if patient.age else "—"],
        ["Date d'examen", patient.exam_date],
        ["Antécédents / densité", patient.history or "—"],
    ]
    t = Table(info, colWidths=[45 * mm, 120 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2eef3"))]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))

    # -- BI-RADS global (badge coloré)
    badge_color = colors.HexColor(_BIRADS_COLORS.get(prediction.birads_class, "#6b1f3a"))
    badge = Table([[f"BI-RADS  {prediction.birads_class}"]], colWidths=[60 * mm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 16),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(Paragraph("BI-RADS global", styles["H"]))
    story.append(badge)
    story.append(Paragraph(clinical_recommendation(prediction.birads_class), styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    # -- description
    story.append(Paragraph("Description", styles["H"]))
    story.append(Paragraph(narrative, styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    # -- tableau des caractéristiques retenues
    story.append(Paragraph("Caractéristiques retenues", styles["H"]))
    rows = [["Groupe", "Catégorie", "Caractéristique", "Probabilité", "Statut"]]
    for name, prob in sorted(prediction.manifestations.items(), key=lambda kv: -kv[1]):
        if prob < 0.10:
            continue
        active = name in prediction.active_manifestations
        group = "Calcifications" if "calcification" in name or "distribution" in name else "Masse / architecture"
        rows.append([
            group, name.replace("_", " "),
            MANIFESTATION_FR.get(name, name),
            f"{prob:.0%}", "retenu" if active else "écarté",
        ])
    feat_table = Table(rows, colWidths=[35 * mm, 35 * mm, 55 * mm, 20 * mm, 20 * mm])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_VIOLET)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 6 * mm))

    # -- probabilités BI-RADS
    story.append(Paragraph("Probabilités BI-RADS", styles["H"]))
    prob_rows = [["Classe", "Confiance"]] + [
        [cls, f"{p:.1%}"] for cls, p in sorted(prediction.birads_probs.items(), key=lambda kv: -kv[1])
    ]
    story.append(Table(prob_rows, colWidths=[40 * mm, 40 * mm], style=TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
    ])))
    story.append(Spacer(1, 6 * mm))

    # -- densité mammaire ACR
    story.append(Paragraph("Densité mammaire (ACR)", styles["H"]))
    story.append(Paragraph(
        ACR_DENSITY_FR.get(prediction.acr_density_class, "Non déterminée"), styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    # -- conclusion
    story.append(Paragraph("Conclusion", styles["H"]))
    story.append(Paragraph(
        f"Score BI-RADS {prediction.birads_class}. "
        f"{clinical_recommendation(prediction.birads_class)} "
        "<b>Cette analyse est générée automatiquement : l'interprétation doit "
        "être confirmée par un radiologue qualifié avant toute décision "
        "clinique.</b>",
        styles["Normal"],
    ))

    SimpleDocTemplate(str(out_path), pagesize=A4).build(story)
    return out_path


# --- Orchestration -------------------------------------------------------------

def generate_report(
    model_path: str,
    left_mlo: str,
    right_cc: str,
    patient: PatientInfo,
    bank_csv: Optional[str] = None,
    image_root: Optional[str] = None,
    out_dir: str = ".",
) -> Path:
    """Pipeline complet : prédiction → cas similaires → synthèse LLM → PDF."""
    model = load_model(model_path)
    prediction = predict(model, left_mlo, right_cc)

    similar_summaries: List[str] = []
    if bank_csv:
        from .similarity_search import SimilarityIndex

        index = SimilarityIndex.build(bank_csv, model, image_root)
        for case in index.query(prediction.embedding, k=5):
            similar_summaries.append(
                f"{case.exam_id} ({case.dataset}) — BI-RADS {case.birads_subcategory}, "
                f"{case.pathology} — similarité {case.similarity:.2f}"
            )

    narrative = synthesize_with_groq(patient, prediction, similar_summaries)
    return export_pdf(patient, prediction, narrative, similar_summaries, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--left", required=True, help="Image MLO")
    parser.add_argument("--right", required=True, help="Image CC")
    parser.add_argument("--patient", required=True)
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--exam-date", default=date.today().isoformat())
    parser.add_argument("--history", default=None)
    parser.add_argument("--bank", default=None, help="CSV des cas d'entraînement")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    patient = PatientInfo(
        name=args.patient, age=args.age, exam_date=args.exam_date, history=args.history
    )
    path = generate_report(
        args.model, args.left, args.right, patient,
        bank_csv=args.bank, image_root=args.image_root, out_dir=args.out_dir,
    )
    print(f"Rapport généré → {path}")


if __name__ == "__main__":
    main()
