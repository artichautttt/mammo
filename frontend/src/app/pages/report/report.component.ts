import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { AnalysisService } from '../../core/analysis.service';
import {
  ACR_DENSITY_FR,
  BIRADS_COLORS,
  BIRADS_RECOMMENDATION_FR,
  MANIFESTATION_FR,
} from '../../core/labels.fr';

/**
 * Vue « Rapport PDF » — même contenu que la vue Résultats, mis en forme pour
 * l'impression / l'archivage (une seule colonne, styles print-friendly).
 */
@Component({
  selector: 'app-report',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <ng-container *ngIf="svc.result as r; else empty">
      <article class="card" style="max-width:720px; margin:0 auto;">
        <header style="border-bottom:2px solid var(--bordeaux); padding-bottom:10px; margin-bottom:16px;">
          <h1 style="color:var(--bordeaux); margin:0;">Rapport de diagnostic mammographique par IA</h1>
          <p class="muted">
            Patient (initiales) : <strong>{{ initials() }}</strong> —
            Document confidentiel — Édité le {{ today }} —
            Examen du {{ svc.patient.examDate }}
          </p>
        </header>

        <h3>Informations patient</h3>
        <p>
          Nom : {{ svc.patient.name }}<br />
          Âge : {{ svc.patient.age ?? '—' }}<br />
          Antécédents / densité : {{ svc.patient.history || '—' }}
        </p>

        <h3>BI-RADS global</h3>
        <span class="badge" [style.background]="color(r.birads_class)">BI-RADS {{ r.birads_class }}</span>
        <p>{{ reco(r.birads_class) }}</p>

        <h3>Description</h3>
        <p>{{ description() }}</p>

        <h3>Caractéristiques retenues</h3>
        <ul>
          <li *ngFor="let m of r.active_manifestations">
            {{ MANIFESTATION_FR[m] ?? m }} ({{ (r.manifestations[m] * 100) | number: '1.0-0' }}%)
          </li>
          <li *ngIf="!r.active_manifestations.length">Aucune manifestation retenue au seuil courant.</li>
        </ul>

        <h3>Probabilités BI-RADS</h3>
        <ul>
          <li *ngFor="let p of probs()">{{ p.cls }} : {{ (p.value * 100) | number: '1.1-1' }}%</li>
        </ul>

        <h3>Densité mammaire (ACR)</h3>
        <p>{{ density(r.acr_density_class) }}</p>

        <h3>Conclusion</h3>
        <p>Score BI-RADS {{ r.birads_class }}. {{ reco(r.birads_class) }}</p>
        <p class="disclaimer">
          Cette analyse est générée automatiquement : l'interprétation doit être
          confirmée par un radiologue qualifié avant toute décision clinique.
        </p>
      </article>

      <div style="text-align:center; margin-top:16px;" class="no-print">
        <button class="btn" (click)="print()">Imprimer</button>
        <a class="btn ghost" routerLink="/export">Retour à l'export</a>
      </div>
    </ng-container>

    <ng-template #empty>
      <p>Aucun rapport. <a routerLink="/import">Démarrer une analyse</a>.</p>
    </ng-template>
  `,
  styles: [
    `@media print { .no-print { display: none; } header { color: #000; } }`,
  ],
})
export class ReportComponent {
  today = new Date().toISOString().slice(0, 10);
  MANIFESTATION_FR = MANIFESTATION_FR;

  constructor(public svc: AnalysisService) {}

  color = (c: string) => BIRADS_COLORS[c] ?? 'var(--bordeaux)';
  reco = (c: string) => BIRADS_RECOMMENDATION_FR[c] ?? 'Corrélation clinique recommandée.';
  density = (c: string) => ACR_DENSITY_FR[c] ?? 'Non déterminée';

  initials(): string {
    return (this.svc.patient.name || '')
      .split(/\s+/)
      .filter(Boolean)
      .map((p) => p[0].toUpperCase())
      .join('');
  }

  description(): string {
    const r = this.svc.result!;
    const active = r.active_manifestations
      .map((m) => MANIFESTATION_FR[m] ?? m.replace(/_/g, ' '))
      .join(', ');
    return active
      ? `Caractéristiques retenues : ${active}. Classification BI-RADS ${r.birads_class}.`
      : `Aucune manifestation significative au seuil courant. Classification BI-RADS ${r.birads_class}.`;
  }

  probs(): { cls: string; value: number }[] {
    const r = this.svc.result!;
    return Object.entries(r.birads_probs)
      .map(([cls, value]) => ({ cls, value }))
      .sort((a, b) => b.value - a.value);
  }

  print(): void {
    window.print();
  }
}
