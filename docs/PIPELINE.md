# Détail du pipeline de données

| Étape | Script | Entrée | Sortie | Points clés |
|------:|--------|--------|--------|-------------|
| 1 | `detect_mvkl_views.py` | PNG MVKL bruts | `mvkl_labeled_views.csv` | template matching `TM_CCOEFF_NORMED` seuil **0.38** + OCR Tesseract ; région de label = **30 %** supérieurs, flou `(3,3)`, seuil binaire **115** ; latéralité par composantes connexes ; `pick_birads_for_breast` (priorité `same_breast` > `bilateral` > `unknown_context`, exclut `no_birads_for_breast`) |
| 2a | `prepare_dataset.py` | 4 CSV CBIS-DDSM | `cbis_indexed.csv` | normalisation colonnes ; `SeriesUID` = avant-dernier segment du chemin DICOM ; indexation des JPEG |
| 2b | `build_dual_view_cases.py` | `cbis_indexed.csv` | `dual_view_cases.csv` | groupement `(patient_id, left_or_right_breast, abnormality_id)` ; appariement CC+MLO (paires incomplètes rejetées) ; description synthétique « seen on CC and MLO views » |
| 3 | `build_combined_dual_view_dataset.py` | `mvkl_labeled_views.csv` + `dual_view_cases.csv` | `combined_dual_view_cases.csv` | colonnes absentes → `pd.NA` ; `pick_most_severe_subcategory` (ordre `{0:0,1:1,2:2,3:3,4:4,4A:4,4B:5,4C:6,5:7,6:8}`) |
| 4 | `filter_combined_dataset.py` | `combined_dual_view_cases.csv` | `combined_dual_view_cases_filtered.csv` | 5 critères ordonnés → **1930 cas** (852 CBIS-DDSM + 1078 MVKL) |
| 5 | `build_hybrid_augmented_weighted_dataset.py` | `..._filtered.csv` | `combined_hybrid_augmented_weighted.csv` | `DEFAULT_TARGET_PER_CLASS = 429`, `CLASS_TARGETS = {"1": 40}` ; params déterministes `random.Random(RANDOM_STATE + source_index*1009 + augmentation_index*9173)` ; **aucune copie image** (augmentation à la volée dans le DataLoader) ; pondération douce complémentaire |

## Distribution finale cible (BI-RADS)

| Classe | 0 | 1 | 2 | 3 | 4A | 4B | 4C | 5 |
|--------|---|---|---|---|----|----|----|---|
| Cible  | 429 | 40 | 429 | 429 | 429 | 429 | 429 | 429 |

## Captures d'écran de l'interface

Placer ici les captures des 5 vues Angular (`accueil.png`, `import.png`,
`resultats.png`, `export.png`, `rapport.png`) une fois l'application lancée.
