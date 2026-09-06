import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <header
      style="background: linear-gradient(90deg, var(--bordeaux), var(--violet)); color:#fff;"
    >
      <div class="container" style="padding:14px 24px; display:flex; align-items:center; gap:16px;">
        <strong>Diagnostic Mammographique par IA</strong>
        <nav style="margin-left:auto; display:flex; gap:14px; font-size:14px;">
          <a routerLink="/accueil" routerLinkActive="active" style="color:#fff;">Accueil</a>
          <a routerLink="/import" routerLinkActive="active" style="color:#fff;">Import</a>
          <a routerLink="/resultats" routerLinkActive="active" style="color:#fff;">Résultats</a>
          <a routerLink="/export" routerLinkActive="active" style="color:#fff;">Export</a>
        </nav>
      </div>
    </header>
    <main class="container">
      <router-outlet></router-outlet>
    </main>
  `,
})
export class AppComponent {}
