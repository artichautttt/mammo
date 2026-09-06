/** Traductions FR des manifestations et des classes de densité ACR. */
export const MANIFESTATION_FR: Record<string, string> = {
  mass: 'masse',
  spiculated_margin: 'contours spiculés',
  circumscribed_margin: 'contours circonscrits',
  obscured_margin: 'contours masqués',
  microlobulated_margin: 'contours microlobulés',
  indistinct_margin: 'contours indistincts',
  round_shape: 'forme ronde',
  oval_shape: 'forme ovale',
  irregular_shape: 'forme irrégulière',
  lobulated_shape: 'forme lobulée',
  architectural_distortion: 'distorsion architecturale',
  asymmetry: 'asymétrie',
  focal_asymmetry: 'asymétrie focale',
  global_asymmetry: 'asymétrie globale',
  developing_asymmetry: 'asymétrie évolutive',
  calcification: 'calcifications',
  amorphous_calcification: 'calcifications amorphes',
  coarse_heterogeneous_calcification: 'calcifications grossières hétérogènes',
  fine_pleomorphic_calcification: 'calcifications fines pléomorphes',
  fine_linear_branching_calcification: 'calcifications fines linéaires branchées',
  punctate_calcification: 'calcifications punctiformes',
  round_calcification: 'calcifications rondes',
  dystrophic_calcification: 'calcifications dystrophiques',
  milk_of_calcium_calcification: 'calcifications de type lait calcique',
  vascular_calcification: 'calcifications vasculaires',
  skin_calcification: 'calcifications cutanées',
  clustered_distribution: 'distribution en amas',
  linear_distribution: 'distribution linéaire',
  segmental_distribution: 'distribution segmentaire',
  regional_distribution: 'distribution régionale',
  diffuse_distribution: 'distribution diffuse',
  grouped_distribution: 'distribution groupée',
  skin_retraction: 'rétraction cutanée',
  nipple_retraction: 'rétraction mamelonnaire',
  trabecular_thickening: 'épaississement trabéculaire',
};

export const ACR_DENSITY_FR: Record<string, string> = {
  '1': 'ACR a — seins presque entièrement graisseux',
  '2': 'ACR b — densités fibroglandulaires éparses',
  '3': 'ACR c — seins denses de façon hétérogène',
  '4': 'ACR d — seins extrêmement denses',
};

export const BIRADS_RECOMMENDATION_FR: Record<string, string> = {
  '0': 'Bilan complémentaire nécessaire (clichés additionnels ou échographie).',
  '1': 'Examen normal. Poursuite du dépistage selon l’âge.',
  '2': 'Findings bénins. Poursuite du dépistage de routine.',
  '3': 'Probablement bénin. Surveillance rapprochée à 6 mois recommandée.',
  '4A': 'Faible suspicion de malignité. Biopsie à discuter.',
  '4B': 'Suspicion intermédiaire de malignité. Biopsie recommandée.',
  '4C': 'Forte suspicion de malignité. Biopsie recommandée.',
  '5': 'Très évocateur de malignité. Prise en charge histologique.',
};

export const BIRADS_COLORS: Record<string, string> = {
  '0': '#8a8a8a', '1': '#2e7d32', '2': '#2e7d32', '3': '#f9a825',
  '4A': '#ef6c00', '4B': '#e65100', '4C': '#c62828', '5': '#b71c1c',
};

export function manifestationGroup(name: string): string {
  return name.includes('calcification') || name.includes('distribution')
    ? 'Calcifications'
    : 'Masse / architecture';
}
