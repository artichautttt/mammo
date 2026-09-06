# Frontend — Diagnostic Mammographique par IA (Angular 17)

Application standalone Angular 17. Parcours linéaire guidé :

**Accueil → Import → Résultats → Export → Rapport PDF**

## Développement

```bash
npm install
npm start          # http://localhost:4200
```

L'application appelle l'API Python (`uvicorn src.api:app --port 8000`) définie
dans `src/app/core/analysis.service.ts` (`API_BASE`).

## Structure

| Fichier | Rôle |
|---|---|
| `src/app/app.routes.ts` | routes du parcours guidé |
| `src/app/core/analysis.service.ts` | état partagé + appels `/predict` et `/report` |
| `src/app/core/labels.fr.ts` | traductions FR (manifestations, ACR, recommandations) |
| `src/app/pages/home` | accueil, bouton « Nouvelle Analyse » |
| `src/app/pages/import` | deux zones drag-and-drop + formulaire patient |
| `src/app/pages/results` | 4 cartes, rapport global, probabilités BI-RADS, densité |
| `src/app/pages/export` | téléchargement automatique du PDF |
| `src/app/pages/report` | rendu écran mis en forme pour impression |

Charte graphique : bordeaux / violet (`src/styles.scss`).
