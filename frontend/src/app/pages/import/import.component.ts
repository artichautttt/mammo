import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AnalysisService } from '../../core/analysis.service';

@Component({
  selector: 'app-import',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <h2 style="color:var(--bordeaux);">Import des clichés</h2>

    <div class="grid-2">
      <div
        class="dropzone"
        [class.filled]="svc.mloPreview"
        (dragover)="$event.preventDefault()"
        (drop)="onDrop($event, 'mlo')"
        (click)="mloInput.click()"
      >
        <strong>Incidence MLO</strong>
        <p class="muted" *ngIf="!svc.mloPreview">Glissez-déposez ou cliquez</p>
        <img *ngIf="svc.mloPreview" [src]="svc.mloPreview" alt="aperçu MLO" style="max-height:180px; margin-top:10px;" />
        <input #mloInput type="file" accept="image/*" hidden (change)="onPick($event, 'mlo')" />
      </div>

      <div
        class="dropzone"
        [class.filled]="svc.ccPreview"
        (dragover)="$event.preventDefault()"
        (drop)="onDrop($event, 'cc')"
        (click)="ccInput.click()"
      >
        <strong>Incidence CC</strong>
        <p class="muted" *ngIf="!svc.ccPreview">Glissez-déposez ou cliquez</p>
        <img *ngIf="svc.ccPreview" [src]="svc.ccPreview" alt="aperçu CC" style="max-height:180px; margin-top:10px;" />
        <input #ccInput type="file" accept="image/*" hidden (change)="onPick($event, 'cc')" />
      </div>
    </div>

    <div class="card" style="margin-top:20px;">
      <h3 style="margin-top:0;">Informations patient</h3>
      <div class="grid-2">
        <label>Nom
          <input type="text" [(ngModel)]="svc.patient.name" style="width:100%;" />
        </label>
        <label>Âge
          <input type="number" [(ngModel)]="svc.patient.age" style="width:100%;" />
        </label>
        <label>Date d'examen
          <input type="date" [(ngModel)]="svc.patient.examDate" style="width:100%;" />
        </label>
        <label>Antécédents / densité (optionnel)
          <input type="text" [(ngModel)]="svc.patient.history" style="width:100%;" />
        </label>
      </div>
    </div>

    <p class="muted" *ngIf="error" style="color:var(--danger);">{{ error }}</p>

    <div style="margin-top:20px; display:flex; gap:12px;">
      <button class="btn" [disabled]="!svc.ready || loading" (click)="generate()">
        {{ loading ? 'Analyse en cours…' : 'Générer le rapport' }}
      </button>
      <button class="btn ghost" (click)="svc.reset()">Réinitialiser</button>
    </div>
  `,
})
export class ImportComponent {
  loading = false;
  error = '';

  constructor(public svc: AnalysisService, private router: Router) {}

  private setFile(file: File, kind: 'mlo' | 'cc'): void {
    const reader = new FileReader();
    reader.onload = () => {
      if (kind === 'mlo') {
        this.svc.mloFile = file;
        this.svc.mloPreview = reader.result as string;
      } else {
        this.svc.ccFile = file;
        this.svc.ccPreview = reader.result as string;
      }
    };
    reader.readAsDataURL(file);
  }

  onPick(event: Event, kind: 'mlo' | 'cc'): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) this.setFile(file, kind);
  }

  onDrop(event: DragEvent, kind: 'mlo' | 'cc'): void {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) this.setFile(file, kind);
  }

  generate(): void {
    this.loading = true;
    this.error = '';
    this.svc.predict().subscribe({
      next: () => {
        this.loading = false;
        this.router.navigate(['/resultats']);
      },
      error: (err) => {
        this.loading = false;
        this.error = `Échec de l'analyse : ${err?.message ?? err}`;
      },
    });
  }
}
