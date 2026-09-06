import { Routes } from '@angular/router';

/** Parcours linéaire guidé : Accueil → Import → Résultats → Export → Rapport PDF. */
export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'accueil' },
  {
    path: 'accueil',
    loadComponent: () =>
      import('./pages/home/home.component').then((m) => m.HomeComponent),
  },
  {
    path: 'import',
    loadComponent: () =>
      import('./pages/import/import.component').then((m) => m.ImportComponent),
  },
  {
    path: 'resultats',
    loadComponent: () =>
      import('./pages/results/results.component').then((m) => m.ResultsComponent),
  },
  {
    path: 'export',
    loadComponent: () =>
      import('./pages/export/export.component').then((m) => m.ExportComponent),
  },
  {
    path: 'rapport',
    loadComponent: () =>
      import('./pages/report/report.component').then((m) => m.ReportComponent),
  },
  { path: '**', redirectTo: 'accueil' },
];
