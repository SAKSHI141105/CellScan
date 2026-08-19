// Vite's dev proxy forwards /api/* to the FastAPI backend on :8000 (see
// vite.config.ts), so relative paths work in both dev and a same-origin
// production deploy without an env var to manage.
const BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

export interface TopContributor {
  feature: string;
  value: number;
  shap_value: number;
}

export interface TabularPredictResult {
  predicted_class: "Benign" | "Malignant";
  probability_malignant: number;
  top_contributors: TopContributor[];
  model_source: string;
  explanation: string;
}

export interface TabularDefaults {
  feature_groups: Record<string, string[]>;
  values: Record<string, number>;
}

export const tabularApi = {
  defaults: () => fetch(`${BASE}/tabular/defaults`).then((r) => handle<TabularDefaults>(r)),

  predict: (features: Record<string, number>) =>
    fetch(`${BASE}/tabular/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    }).then((r) => handle<TabularPredictResult>(r)),

  predictBatch: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/tabular/predict-batch`, { method: "POST", body: form }).then((r) =>
      handle<{ rows: Record<string, unknown>[]; n_rows: number; n_malignant: number; n_benign: number }>(r)
    );
  },
};

export interface ImagePredictResult {
  predicted_class: "Benign" | "Malignant";
  probability_malignant: number;
  preprocessed_png_base64: string;
  gradcam_png_base64: string;
  model_key: string;
  filename: string;
}

export const imageApi = {
  status: () => fetch(`${BASE}/image/status`).then((r) => handle<{ available: boolean; model_key: string | null }>(r)),

  predict: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/image/predict`, { method: "POST", body: form }).then((r) => handle<ImagePredictResult>(r));
  },

  predictBatch: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return fetch(`${BASE}/image/predict-batch`, { method: "POST", body: form }).then((r) =>
      handle<{ results: ImagePredictResult[]; n_malignant: number; n_benign: number }>(r)
    );
  },
};

export interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface RocCurve {
  name: string;
  auc: number;
  points: { fpr: number; tpr: number }[];
}

export const reportsApi = {
  tabularComparison: () => fetch(`${BASE}/reports/tabular-comparison`).then((r) => handle<{ models: ModelMetrics[] }>(r)),
  smoteComparison: () => fetch(`${BASE}/reports/smote-comparison`).then((r) => handle<{ rows: Record<string, unknown>[] }>(r)),
  clusteringSummary: () => fetch(`${BASE}/reports/clustering-summary`).then((r) => handle<{ methods: Record<string, unknown>[] }>(r)),
  rocCurves: () => fetch(`${BASE}/reports/roc-curves`).then((r) => handle<{ curves: RocCurve[] }>(r)),

  downloadCsv: (payload: { predicted_class: string; probability_malignant: number; source: string; top_contributors?: TopContributor[] }) =>
    fetch(`${BASE}/report/csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.blob()),

  downloadPdf: (payload: { predicted_class: string; probability_malignant: number; source: string; top_contributors?: TopContributor[] }) =>
    fetch(`${BASE}/report/pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.blob()),
};

export interface ClusterPoint {
  x: number;
  y: number;
  trueDiagnosis: "Benign" | "Malignant";
  kmeansCluster: string;
}

export const clustersApi = {
  projection: (method: "pca" | "tsne") =>
    fetch(`${BASE}/clusters/projection?method=${method}`).then((r) => handle<{ points: ClusterPoint[] }>(r)),
};

export function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
