import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { PatientForm, PredictionResult } from './analysis.models';

const API_BASE = 'http://localhost:8000';

/**
 * État partagé du parcours d'analyse + appels à l'API d'inférence Python.
 * Conserve les images importées et le dernier résultat pour que les vues
 * Résultats / Export / Rapport y accèdent sans re-uploader.
 */
@Injectable({ providedIn: 'root' })
export class AnalysisService {
  mloFile: File | null = null;
  ccFile: File | null = null;
  mloPreview: string | null = null;
  ccPreview: string | null = null;

  patient: PatientForm = { name: '', examDate: new Date().toISOString().slice(0, 10) };
  result: PredictionResult | null = null;

  constructor(private http: HttpClient) {}

  get ready(): boolean {
    return !!this.mloFile && !!this.ccFile && this.patient.name.trim().length > 0;
  }

  reset(): void {
    this.mloFile = this.ccFile = null;
    this.mloPreview = this.ccPreview = null;
    this.result = null;
    this.patient = { name: '', examDate: new Date().toISOString().slice(0, 10) };
  }

  private buildForm(): FormData {
    const form = new FormData();
    form.append('mlo', this.mloFile as File);
    form.append('cc', this.ccFile as File);
    form.append('patient', this.patient.name);
    if (this.patient.age != null) form.append('age', String(this.patient.age));
    form.append('exam_date', this.patient.examDate);
    if (this.patient.history) form.append('history', this.patient.history);
    return form;
  }

  /** Lance la prédiction et mémorise le résultat. */
  predict(): Observable<PredictionResult> {
    return this.http
      .post<PredictionResult>(`${API_BASE}/predict`, this.buildForm())
      .pipe(tap((res) => (this.result = res)));
  }

  /** Récupère le PDF généré côté backend (ReportLab). */
  downloadReport(): Observable<Blob> {
    return this.http.post(`${API_BASE}/report`, this.buildForm(), {
      responseType: 'blob',
    });
  }
}
