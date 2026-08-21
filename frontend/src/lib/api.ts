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

// FastAPI's HTTPException usually carries a plain string in `detail`, but a
// few endpoints (image inference) raise a structured {error, details} object
// so the UI can show the actual failure reason instead of a generic "failed"
// — this normalizes either shape into one readable string.
function extractErrorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "error" in detail) {
      const { error, details } = detail as { error: string; details?: string };
      return details ? `${error}: ${details}` : error;
    }
  }
  return fallback;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, extractErrorMessage(body, res.statusText));
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

  explainLime: (features: Record<string, number>) =>
    fetch(`${BASE}/tabular/explain-lime`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    }).then((r) => handle<{ weights: { feature: string; weight: number }[] }>(r)),
};

export interface TextureFeatures {
  glcm_contrast: number;
  glcm_homogeneity: number;
  glcm_energy: number;
  glcm_correlation: number;
  glcm_dissimilarity: number;
  glcm_ASM: number;
  canny_edge_density: number;
  sobel_mean: number;
  sobel_std: number;
}

export interface PixelHistogram {
  bin_centers: number[];
  counts: number[];
  mean_intensity: number;
  std_intensity: number;
}

export interface ImagePredictResult {
  predicted_class: "Benign" | "Malignant";
  probability_malignant: number;
  preprocessed_png_base64: string;
  gradcam_png_base64: string;
  model_key: string;
  texture_features: TextureFeatures;
  pixel_histogram: PixelHistogram;
  is_demo: boolean;
  filename: string;
}

export const imageApi = {
  status: () =>
    fetch(`${BASE}/image/status`).then((r) => handle<{ available: boolean; model_key: string | null; is_demo: boolean }>(r)),

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

export interface MammographyExplanation {
  predicted_probability: number;
  estimated_lesion_area_fraction: number;
  attention_concentration: number;
  summary: string;
}

export interface MammographyPredictResult {
  predicted_class: "Benign" | "Malignant";
  probability_malignant: number;
  preprocessed_png_base64: string;
  gradcam_png_base64: string;
  model_key: string;
  explanation: MammographyExplanation;
  texture_features: TextureFeatures;
  pixel_histogram: PixelHistogram;
  is_demo: boolean;
  filename: string;
}

export const mammographyApi = {
  status: () => fetch(`${BASE}/mammography/status`).then((r) => handle<{ available: boolean; is_demo: boolean }>(r)),

  predictBatch: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return fetch(`${BASE}/mammography/predict-batch`, { method: "POST", body: form }).then((r) =>
      handle<{ results: MammographyPredictResult[]; n_malignant: number; n_benign: number }>(r)
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

export interface ConfusionMatrix {
  name: string;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export const reportsApi = {
  tabularComparison: () => fetch(`${BASE}/reports/tabular-comparison`).then((r) => handle<{ models: ModelMetrics[] }>(r)),
  smoteComparison: () => fetch(`${BASE}/reports/smote-comparison`).then((r) => handle<{ rows: Record<string, unknown>[] }>(r)),
  clusteringSummary: () => fetch(`${BASE}/reports/clustering-summary`).then((r) => handle<{ methods: Record<string, unknown>[] }>(r)),
  rocCurves: () => fetch(`${BASE}/reports/roc-curves`).then((r) => handle<{ curves: RocCurve[] }>(r)),
  confusionMatrices: () => fetch(`${BASE}/reports/confusion-matrices`).then((r) => handle<{ matrices: ConfusionMatrix[] }>(r)),

  downloadCsv: (payload: ReportPayload) =>
    fetch(`${BASE}/report/csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.blob()),

  downloadPdf: (payload: ReportPayload) =>
    fetch(`${BASE}/report/pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.blob()),
};

export interface ReportPayload {
  predicted_class: string;
  probability_malignant: number;
  source: string;
  top_contributors?: TopContributor[];
  gradcam_png_base64?: string;
  is_demo?: boolean;
}

export interface ClusterPoint {
  x: number;
  y: number;
  z?: number;
  trueDiagnosis: "Benign" | "Malignant";
  kmeansCluster: string;
}

export const clustersApi = {
  projection: (method: "pca" | "tsne" | "umap", dimensions: 2 | 3 = 2) =>
    fetch(`${BASE}/clusters/projection?method=${method}&dimensions=${dimensions}`).then((r) =>
      handle<{ points: ClusterPoint[] }>(r)
    ),
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
