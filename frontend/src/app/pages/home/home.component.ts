import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [RouterLink],
  template: `
    <section class="card" style="text-align:center; padding:48px 24px;">
      <h1 style="color:var(--bordeaux); margin-bottom:8px;">
        Diagnostic Mammographique par Intelligence Artificielle
      </h1>
      <p class="muted" style="max-width:560px; margin:0 auto 28px;">
        Génération automatique d'un rapport BI-RADS structuré à partir de deux
        incidences (MLO et CC) d'un même sein.
      </p>

      <svg width="220" height="140" viewBox="0 0 220 140" style="margin-bottom:28px;">
        <rect width="220" height="140" rx="10" fill="#f2eef3" />
        <path d="M30 120 Q45 40 80 55 Q95 70 60 120 Z" fill="var(--violet-light)" />
        <path d="M130 120 Q150 30 190 60 Q200 90 165 120 Z" fill="var(--bordeaux)" opacity="0.8" />
        <text x="45" y="135" font-size="11" fill="var(--muted)">MLO</text>
        <text x="150" y="135" font-size="11" fill="var(--muted)">CC</text>
      </svg>

      <div>
        <a class="btn" routerLink="/import">Nouvelle Analyse</a>
      </div>

      <p class="disclaimer" style="margin-top:32px;">
        Outil d'aide au diagnostic à visée académique. L'interprétation d'un
        radiologue qualifié reste indispensable.
      </p>
    </section>
  `,
})
export class HomeComponent {}
