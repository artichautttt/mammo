import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { AnalysisService } from '../../core/analysis.service';

@Component({
  selector: 'app-export',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <h2 style="color:var(--bordeaux);">Export du rapport</h2>

    <div class="card" *ngIf="svc.result; else empty">
      <p>
        Le rapport PDF (<code>rapport_birads_{{ safeName() }}.pdf</code>) est
        généré côté serveur puis téléchargé automatiquement.
      </p>
      <button class="btn" [disabled]="loading" (click)="exportPdf()">
        {{ loading ? 'Préparation…' : 'Exporter le rapport' }}
      </button>

      <div class="disclaimer" *ngIf="downloadUrl">
        Téléchargement prêt —
        <a [href]="downloadUrl" [download]="fileName" target="_blank">Ouvrir un fichier</a>
      </div>

      <p class="muted" *ngIf="error" style="color:var(--danger);">{{ error }}</p>

      <div style="margin-top:16px;">
        <a class="btn ghost" routerLink="/rapport">Voir le rapport à l'écran</a>
      </div>
    </div>

    <ng-template #empty>
      <p>Aucune analyse à exporter. <a routerLink="/import">Démarrer une analyse</a>.</p>
    </ng-template>
  `,
})
export class ExportComponent {
  loading = false;
  error = '';
  downloadUrl: string | null = null;
  fileName = 'rapport.pdf';

  constructor(public svc: AnalysisService) {}

  safeName(): string {
    return (this.svc.patient.name || 'Patient').replace(/[^A-Za-z0-9]+/g, '');
  }

  exportPdf(): void {
    this.loading = true;
    this.error = '';
    this.fileName = `rapport_birads_${this.safeName()}.pdf`;
    this.svc.downloadReport().subscribe({
      next: (blob) => {
        this.loading = false;
        this.downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = this.downloadUrl;
        a.download = this.fileName;
        a.click();
      },
      error: (err) => {
        this.loading = false;
        this.error = `Échec de l'export : ${err?.message ?? err}`;
      },
    });
  }
}
