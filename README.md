# Diagnostic Mammographique par Intelligence Artificielle

Génération automatique de rapports mammographiques structurés (BI-RADS) à partir
de deux vues (CC + MLO) d'un même sein, à l'aide d'un modèle CLIP multi-tâches,
d'une recherche par similarité de cas, et d'une synthèse en langage naturel via
un LLM (API Groq / LLaMA).

> Projet de semestre — École Mohammadia d'Ingénieurs.
> Ce dépôt est une reconstruction fidèle de l'architecture décrite dans le
> rapport technique original (code source initial perdu).

---

## 1. Contexte

Le pipeline prend en entrée deux images mammographiques (incidence craniocaudale
`CC` et médiolatérale oblique `MLO`) et produit :

1. un score **BI-RADS** global (0, 1, 2, 3, 4A, 4B, 4C, 5, 6) ;
2. un vecteur de **manifestations** cliniques (35 manifestations binaires issues
   du jeu MVKL, + descripteurs morphologiques CBIS-DDSM : forme de masse, marges,
   type et distribution de calcifications) ;
3. une estimation de la **densité mammaire ACR** ;
4. les **5 cas les plus similaires** de la base d'entraînement (similarité
   cosinus sur l'embedding concaténé 1024-D) ;
5. un **rapport PDF** en français, rédigé par un LLM à partir des prédictions et
   des cas similaires, avec rappel explicite qu'un radiologue doit confirmer
   l'interprétation.

Deux jeux de données publics sont fusionnés :

| Source     | Annotations disponibles                                   | Manque                        |
|------------|----------------------------------------------------------|-------------------------------|
| **MVKL**   | vecteur de 35 manifestations binaires, BI-RADS textuel   | descripteurs morphologiques   |
| **CBIS-DDSM** | `mass_shape`, `mass_margins`, `calc_type`, `calc_distribution`, pathologie | vecteur de manifestations |

Les colonnes absentes d'une source restent à `pd.NA` et sont **ignorées dans la
fonction de perte** (`ignore_index = -100`).

---

## 2. Architecture

```
Images CC + MLO
      │
      ▼
detect_mvkl_views.py ──► mvkl_labeled_views.csv        (vue + latéralité + BI-RADS)
prepare_dataset.py    ──► cbis_indexed.csv
build_dual_view_cases.py ─► dual_view_cases.csv        (paires CC/MLO + description)
      │
      ▼
build_combined_dual_view_dataset.py ─► combined_dual_view_cases.csv
filter_combined_dataset.py          ─► combined_dual_view_cases_filtered.csv   (1930 cas)
build_hybrid_augmented_weighted_dataset.py ─► combined_hybrid_augmented_weighted.csv
      │
      ▼
train_report_ready.py ──► best_two_view_model.pt
      │
      ▼
two_view_model.py predict ──► manifestations + BI-RADS + embedding 1024-D
similarity_search.py       ──► top-k=5 cas proches
report_generator.py        ──► appel Groq (LLaMA) + rapport_birads_<Patient>.pdf
```

### Modèle `DualViewCLIP`

- Backbone **CLIP ViT-B/16 gelé** (`open_clip`), embedding 512-D.
- Fusion : `concat(cc, mlo)` (1024) → `Linear(1024, 512)` → `LayerNorm` → `GELU`
  → `Dropout(0.1)`.
- Têtes multi-tâches (`nn.ModuleDict`) : une tête `Linear` par tâche
  (BI-RADS, 35 manifestations, densité ACR, forme/marges/type/distribution).

---

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dépendances système :

- **Tesseract OCR** (binaire) — https://github.com/tesseract-ocr/tesseract
  Sous Windows, renseigner le chemin dans `.env` :
  `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
- Clé API Groq dans `.env` : `GROQ_API_KEY=...`

Copier `.env.example` vers `.env` et compléter.

---

## 4. Lancer le pipeline de bout en bout

Placer les données brutes dans `data/raw/MVKL/` et `data/raw/CBIS-DDSM/`
(non versionnées). Puis :

```bash
python scripts/detect_mvkl_views.py           --raw data/raw/MVKL           --out data/processed/mvkl_labeled_views.csv
python scripts/prepare_dataset.py             --cbis data/raw/CBIS-DDSM     --out data/processed/cbis_indexed.csv
python scripts/build_dual_view_cases.py       --indexed data/processed/cbis_indexed.csv --out data/processed/dual_view_cases.csv
python scripts/build_combined_dual_view_dataset.py --mvkl data/processed/mvkl_labeled_views.csv --cbis data/processed/dual_view_cases.csv --out data/processed/combined_dual_view_cases.csv
python scripts/filter_combined_dataset.py     --in data/processed/combined_dual_view_cases.csv --out data/processed/combined_dual_view_cases_filtered.csv
python scripts/build_hybrid_augmented_weighted_dataset.py --in data/processed/combined_dual_view_cases_filtered.csv --out data/processed/combined_hybrid_augmented_weighted.csv
```

Après l'étape 5 vous devez obtenir **1930 cas** (852 CBIS-DDSM + 1078 MVKL).

---

## 5. Entraînement

```bash
python src/train_report_ready.py       --dataset data/processed/combined_hybrid_augmented_weighted.csv       --epochs 18       --out best_two_view_model.pt
```

- `WeightedRandomSampler` sur la tâche BI-RADS (fréquence inverse normalisée,
  plafonnée à `max_sample_weight`).
- Perte multi-tâches = somme des cross-entropies par tâche (`ignore_index=-100`).
- 15 à 20 epochs max (au-delà : surapprentissage). Le meilleur checkpoint est
  conservé entre les epochs 12 et 15.

---

## 6. Prédiction

```bash
python src/two_view_model.py predict       --model best_two_view_model.pt       --left  path/vers/MLO.png       --right path/vers/CC.png
```

Génération du rapport complet + PDF :

```bash
python src/report_generator.py       --model best_two_view_model.pt       --left MLO.png --right CC.png       --patient "Jean Dupont" --age 57 --exam-date 2026-09-06       --bank data/processed/combined_hybrid_augmented_weighted.csv
# ⇒ rapport_birads_JeanDupont.pdf
```

API pour le frontend :

```bash
uvicorn src.api:app --reload --port 8000
```

---

## 7. Frontend Angular 17

```bash
cd frontend
npm install
npm start          # http://localhost:4200
```

Parcours linéaire guidé : **Accueil → Import → Résultats → Export → Rapport PDF**.
Charte graphique bordeaux/violet, orientée milieu médical.

Captures d'écran : voir `docs/`.

---

## 8. Tests

```bash
pytest -q
```

Les tests couvrent les fonctions de fusion / filtrage / sévérité BI-RADS qui ne
dépendent pas de vraies images.

---

## 9. Avertissement

Cet outil est un prototype académique. Aucune décision clinique ne doit être
prise sur la seule base de ses sorties : **l'interprétation d'un radiologue
qualifié est requise**.
