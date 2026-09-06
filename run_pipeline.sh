#!/usr/bin/env bash
# Exécute les 5 étapes de préparation des données de bout en bout.
# Prérequis : data/raw/MVKL et data/raw/CBIS-DDSM peuplés (non versionnés).
set -euo pipefail

RAW_MVKL="${RAW_MVKL:-data/raw/MVKL}"
RAW_CBIS="${RAW_CBIS:-data/raw/CBIS-DDSM}"
PROC="data/processed"
mkdir -p "$PROC"

echo "== Étape 1 : détection des vues MVKL =="
python scripts/detect_mvkl_views.py --raw "$RAW_MVKL" --out "$PROC/mvkl_labeled_views.csv"

echo "== Étape 2a : indexation CBIS-DDSM =="
python scripts/prepare_dataset.py --cbis "$RAW_CBIS" --out "$PROC/cbis_indexed.csv"

echo "== Étape 2b : cas bi-vues CBIS-DDSM =="
python scripts/build_dual_view_cases.py --indexed "$PROC/cbis_indexed.csv" --out "$PROC/dual_view_cases.csv"

echo "== Étape 3 : fusion MVKL + CBIS-DDSM =="
python scripts/build_combined_dual_view_dataset.py \
    --mvkl "$PROC/mvkl_labeled_views.csv" \
    --cbis "$PROC/dual_view_cases.csv" \
    --out  "$PROC/combined_dual_view_cases.csv"

echo "== Étape 4 : filtrage (attendu : 1930 cas) =="
python scripts/filter_combined_dataset.py \
    --in  "$PROC/combined_dual_view_cases.csv" \
    --out "$PROC/combined_dual_view_cases_filtered.csv"

echo "== Étape 5 : dataset hybride augmenté + pondéré =="
python scripts/build_hybrid_augmented_weighted_dataset.py \
    --in  "$PROC/combined_dual_view_cases_filtered.csv" \
    --out "$PROC/combined_hybrid_augmented_weighted.csv"

echo "Pipeline terminé → $PROC/combined_hybrid_augmented_weighted.csv"
