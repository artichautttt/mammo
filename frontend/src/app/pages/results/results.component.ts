import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { AnalysisService } from '../../core/analysis.service';
import { FeatureRow } from '../../core/analysis.models';
import {
  ACR_DENSITY_FR,
  BIRADS_COLORS,
  BIRADS_RECOMMENDATION_FR,
  MANIFESTATION_FR,
  manifestationGroup,
} from '../../core/labels.fr';

@Component({
  selector: 'app-results',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <ng-container *ngIf="svc.result as r; else empty">
      <div style="display:flex; align-items:baseline; gap:12px;">
        <h2 style="color:var(--bordeaux); margin:0;">Rapport d'analyse</h2>
        <span class="muted">{{ svc.patient.examDate }}</span>
      </div>

      <div class="grid-4" style="margin:16px 0;">
        <div class="card"><div class="muted">Patient</div><strong>{{ svc.patient.name }}</strong></div>
        <div class="card"><div class="muted">Âge</div><strong>{{ svc.patient.age ?? '—' }}</strong></div>
        <div class="card"><div class="muted">Date d'examen</div><strong>{{ svc.patient.examDate }}</strong></div>
        <div class="card">
          <div class="muted">BI-RADS global</div>
          <span class="badge" [style.background]="biradsColor(r.birads_class)">{{ r.birads_class }}</span>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-top:0;">Rapport global</h3>
        <p>{{ description() }}</p>
        <table>
          <thead>
            <tr><th>Groupe</th><th>Catégorie</th><th>Caractéristique</th><th>Probabilité</th><th>Statut</th></tr>
          </thead>
          <tbody>
            <tr *ngFor="let row of features()">
              <td>{{ row.group }}</td>
              <td>{{ row.category }}</td>
              <td>{{ row.feature }}</td>
              <td>{{ (row.probability * 100) | number: '1.0-0' }}%</td>
              <td [style.color]="row.retained ? 'var(--ok)' : 'var(--muted)'">
                {{ row.retained ? 'retenu' : 'écarté' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card" style="margin-top:16px;">
        <h3 style="margin-top:0;">Probabilités BI-RADS</h3>
        <table>
          <thead><tr><th>Classe</th><th>Confiance</th></tr></thead>
          <tbody>
            <tr *ngFor="let p of biradsProbs()">
              <td>{{ p.cls }}</td>
              <td>{{ (p.value * 100) | number: '1.1-1' }}%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card" style="margin-top:16px;">
        <h3 style="margin-top:0;">Densité mammaire</h3>
        <p>{{ densityLabel(r.acr_density_class) }}</p>
      </div>

      <div class="card" style="margin-top:16px;">
        <h3 style="margin-top:0;">Conclusion</h3>
        <p>Score BI-RADS {{ r.birads_class }}. {{ recommendation(r.birads_class) }}</p>
        <p class="disclaimer">
          Analyse générée automatiquement : l'interprétation doit être confirmée
          par un radiologue qualifié avant toute décision clinique.
        </p>
      </div>

      <div style="margin-top:20px;">
        <a class="btn" routerLink="/export">Aller à l'export</a>
      </div>
    </ng-container>

    <ng-template #empty>
      <p>Aucune analyse disponible. <a routerLink="/import">Démarrer une analyse</a>.</p>
    </ng-template>
  `,
})
export class ResultsComponent {
  constructor(public svc: AnalysisService, private router: Router) {}

  biradsColor = (c: string) => BIRADS_COLORS[c] ?? 'var(--bordeaux)';
  recommendation = (c: string) => BIRADS_RECOMMENDATION_FR[c] ?? 'Corrélation clinique recommandée.';
  densityLabel = (c: string) => ACR_DENSITY_FR[c] ?? 'Non déterminée';

  description(): string {
    const r = this.svc.result;
    if (!r) return '';
    const active = r.active_manifestations
      .map((m) => MANIFESTATION_FR[m] ?? m.replace(/_/g, ' '))
      .join(', ');
    return active
      ? `Caractéristiques retenues : ${active}. Classification BI-RADS ${r.birads_class}.`
      : `Aucune manifestation significative au seuil courant. Classification BI-RADS ${r.birads_class}.`;
  }

  features(): FeatureRow[] {
    const r = this.svc.result;
    if (!r) return [];
    return Object.entries(r.manifestations)
      .filter(([, p]) => p >= 0.1)
      .sort((a, b) => b[1] - a[1])
      .map(([name, p]) => ({
        group: manifestationGroup(name),
        category: name.replace(/_/g, ' '),
        feature: MANIFESTATION_FR[name] ?? name,
        probability: p,
        retained: r.active_manifestations.includes(name),
      }));
  }

  biradsProbs(): { cls: string; value: number }[] {
    const r = this.svc.result;
    if (!r) return [];
    return Object.entries(r.birads_probs)
      .map(([cls, value]) => ({ cls, value }))
      .sort((a, b) => b.value - a.value);
  }
}
