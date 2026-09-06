export interface PatientForm {
  name: string;
  age?: number;
  examDate: string;
  history?: string;
}

export interface PredictionResult {
  birads_class: string;
  birads_probs: Record<string, number>;
  acr_density_class: string;
  acr_density_probs: Record<string, number>;
  manifestations: Record<string, number>;
  active_manifestations: string[];
  morphology: Record<string, string>;
}

export interface FeatureRow {
  group: string;
  category: string;
  feature: string;
  probability: number;
  retained: boolean;
}
