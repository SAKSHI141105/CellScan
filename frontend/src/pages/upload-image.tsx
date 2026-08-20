import { useEffect, useMemo, useState } from "react";
import { Dropzone } from "@/components/ui/dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { StagedLoader } from "@/components/ui/staged-loader";
import { RiskGauge } from "@/components/ui/risk-gauge";
import { TextureFeatureChart } from "@/components/texture-feature-chart";
import { DisclaimerBanner } from "@/components/layout/disclaimer-banner";
import { ApiError, imageApi, reportsApi, triggerDownload, type ImagePredictResult } from "@/lib/api";
import { riskTier } from "@/lib/utils";
import { AlertTriangle, Download, FileDown, FileImage } from "lucide-react";

const STAGE_MESSAGES = [
  "Preprocessing image — grayscale, resize, CLAHE...",
  "Denoising and normalizing pixel intensities...",
  "Running CNN inference...",
  "Generating Grad-CAM heatmap overlay...",
];

/** A short, deterministic tag derived from the filename — not a real
 * specimen/accession number, just something stable to reference the upload
 * by in the metadata strip and the exported report. */
function specimenTag(filename: string): string {
  let hash = 0;
  for (const ch of filename) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return `SPEC-${hash.toString(36).toUpperCase().padStart(6, "0").slice(0, 6)}`;
}

function narrativeFor(predictedClass: string, probability: number, isDemo: boolean): string {
  if (isDemo) {
    return "This model has never been trained on histopathology data — it only carries ImageNet-pretrained weights. The prediction, heatmap, and probability above are not clinically meaningful; they exist to demonstrate the UI while a real model finishes training.";
  }
  const pct = (probability * 100).toFixed(1);
  if (predictedClass === "Malignant") {
    return `The model's attention, visualized via Grad-CAM, concentrates on the highlighted tissue regions in the heatmap. At ${pct}% predicted probability of malignancy, this sample falls in the ${riskTier(probability).label.toLowerCase()} band and would warrant closer review in a real triage workflow.`;
  }
  return `Grad-CAM attention is diffuse rather than concentrated on any single irregular structure, consistent with the model's ${pct}% predicted probability of malignancy — the ${riskTier(probability).label.toLowerCase()} band.`;
}

function SpecimenHeader({
  filename,
  timestamp,
  modelKey,
  isDemo,
}: {
  filename: string;
  timestamp: string;
  modelKey: string;
  isDemo: boolean;
}) {
  const fields = [
    { label: "Specimen ID", value: specimenTag(filename) },
    { label: "Source file", value: filename },
    { label: "Analyzed", value: timestamp },
    { label: "Specimen type", value: "Histopathology tissue patch" },
    { label: "Model", value: isDemo ? `${modelKey.replace(/_/g, " ")} (DEMO)` : modelKey.replace(/_/g, " ") },
  ];
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-2 rounded-lg border border-border bg-muted/50 px-4 py-3 sm:grid-cols-5">
      {fields.map((f) => (
        <div key={f.label} className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{f.label}</div>
          <div className={`truncate text-xs font-semibold ${f.label === "Model" && isDemo ? "text-risk-high-fg" : ""}`} title={f.value}>
            {f.value}
          </div>
        </div>
      ))}
    </div>
  );
}

function DemoModeBanner() {
  return (
    <div className="mt-4 flex items-start gap-2 rounded-lg border-2 border-risk-high-fg/40 bg-risk-high-bg/60 px-4 py-3 text-sm text-risk-high-fg">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>
        <strong>Demo mode.</strong> The active model has ImageNet-pretrained weights only — it has never seen a
        histopathology image and its predictions are not meaningful. This exists so you can exercise the upload →
        preprocess → Grad-CAM → report flow before a real model is trained. Run{" "}
        <code className="font-mono-num">python scripts/train_image.py</code> with a real dataset for genuine
        predictions.
      </span>
    </div>
  );
}

export function UploadImage() {
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);
  const [modelIsDemo, setModelIsDemo] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ImagePredictResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const [downloading, setDownloading] = useState<"csv" | "pdf" | null>(null);
  const [analyzedAt, setAnalyzedAt] = useState<string>("");

  useEffect(() => {
    imageApi.status().then((s) => {
      setModelAvailable(s.available);
      setModelIsDemo(s.is_demo);
    });
  }, []);

  const previewUrls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);

  function handleFiles(accepted: File[]) {
    setFiles(accepted);
    setResults(null);
    setSelected(0);
    setError(null);
  }

  async function runPrediction() {
    if (!files.length) return;
    setLoading(true);
    setError(null);
    try {
      const res = await imageApi.predictBatch(files);
      setResults(res.results);
      setAnalyzedAt(new Date().toLocaleString());
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  async function download(kind: "csv" | "pdf") {
    if (!results) return;
    const result = results[selected];
    setDownloading(kind);
    try {
      const payload = { ...result, source: result.filename, gradcam_png_base64: result.gradcam_png_base64, is_demo: result.is_demo };
      const blob = kind === "csv" ? await reportsApi.downloadCsv(payload) : await reportsApi.downloadPdf(payload);
      triggerDownload(blob, `cellscan_${specimenTag(result.filename)}_report.${kind}`);
    } finally {
      setDownloading(null);
    }
  }

  const current = results?.[selected];
  const currentPreviewUrl = previewUrls[selected];

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Histopathology Image Prediction</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Upload one or more stained tissue patches (BreakHis / IDC-style). CellScan converts each to grayscale,
        resizes to 224x224, applies CLAHE contrast enhancement and denoising, then scores it with the trained
        CNN.
      </p>

      {modelAvailable === false && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-risk-mid-fg/25 bg-risk-mid-bg/50 px-4 py-3 text-sm text-risk-mid-fg">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            No trained image model or demo weights found. Run{" "}
            <code className="font-mono-num">python scripts/generate_demo_weights.py</code> to test this page
            immediately with an untrained model, or <code className="font-mono-num">python scripts/train_image.py</code>{" "}
            after downloading the histopathology dataset for real predictions (see README).
          </span>
        </div>
      )}
      {modelAvailable === true && modelIsDemo && <DemoModeBanner />}

      <div className="mt-6">
        <Dropzone
          onFiles={handleFiles}
          accept={{ "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif"] }}
          multiple
          hint="One or more PNG/JPG/TIF tissue patches"
        />
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap gap-3">
            {files.map((f, i) => (
              <figure key={f.name + i} className="w-24">
                <img src={previewUrls[i]} className="aspect-square w-24 rounded-lg border border-border object-cover" />
                <figcaption className="mt-1 truncate text-[10px] text-muted-foreground" title={f.name}>
                  {f.name}
                </figcaption>
              </figure>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">{files.length} file(s) ready</span>
            <Button onClick={runPrediction} loading={loading}>
              Run prediction
            </Button>
          </div>
        </div>
      )}

      {error && <p className="mt-3 text-sm text-risk-high-fg">{error}</p>}
      {loading && (
        <Card className="mt-4">
          <CardContent className="pt-5">
            <StagedLoader stages={STAGE_MESSAGES} />
          </CardContent>
        </Card>
      )}

      {results && (
        <div className="mt-6 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Batch results</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {results.length} image(s) scored —{" "}
              <strong>{results.filter((r) => r.predicted_class === "Malignant").length}</strong> flagged malignant.
              {results.some((r) => r.is_demo) && (
                <span className="ml-2 font-semibold text-risk-high-fg">(demo model — not real predictions)</span>
              )}
            </CardContent>
          </Card>

          <Table>
            <THead>
              <TR>
                <TH>Filename</TH>
                <TH>Predicted</TH>
                <TH>Probability</TH>
                <TH>Risk</TH>
              </TR>
            </THead>
            <TBody>
              {results.map((r, i) => {
                const { tier, label } = riskTier(r.probability_malignant);
                return (
                  <TR
                    key={r.filename + i}
                    onClick={() => setSelected(i)}
                    className={`cursor-pointer hover:bg-muted/60 ${selected === i ? "bg-muted" : ""}`}
                  >
                    <TD className="font-sans">{r.filename}</TD>
                    <TD>{r.predicted_class}</TD>
                    <TD>{(r.probability_malignant * 100).toFixed(1)}%</TD>
                    <TD className="flex flex-wrap gap-1.5">
                      <Badge tier={tier}>{label}</Badge>
                      {r.is_demo && <Badge tier="high">DEMO</Badge>}
                    </TD>
                  </TR>
                );
              })}
            </TBody>
          </Table>

          {current && (
            <div className="space-y-4">
              <SpecimenHeader filename={current.filename} timestamp={analyzedAt} modelKey={current.model_key} isDemo={current.is_demo} />

              <div className="grid gap-4 lg:grid-cols-2">
                {/* left column — original alongside the Grad-CAM overlay */}
                <Card>
                  <CardHeader>
                    <CardTitle>Original vs. Grad-CAM overlay</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <figure>
                        <img src={currentPreviewUrl} className="aspect-square w-full rounded-lg border border-border object-cover" />
                        <figcaption className="mt-1.5 text-xs text-muted-foreground">Original upload</figcaption>
                      </figure>
                      <figure>
                        <img
                          src={`data:image/png;base64,${current.gradcam_png_base64}`}
                          className="aspect-square w-full rounded-lg border border-border object-cover"
                        />
                        <figcaption className="mt-1.5 text-xs text-muted-foreground">Grad-CAM heatmap</figcaption>
                      </figure>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Warmer regions in the heatmap contributed more strongly to the predicted class.
                    </p>
                    <div className="flex items-center gap-2 border-t border-border pt-3">
                      <FileImage className="h-3.5 w-3.5 text-muted-foreground" />
                      <img
                        src={`data:image/png;base64,${current.preprocessed_png_base64}`}
                        className="h-12 w-12 rounded border border-border object-cover"
                      />
                      <span className="text-xs text-muted-foreground">Preprocessed input (grayscale + CLAHE + denoise)</span>
                    </div>
                  </CardContent>
                </Card>

                {/* right column — risk gauge, narrative, report actions */}
                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Risk assessment</CardTitle>
                    </CardHeader>
                    <CardContent className="flex flex-col items-center gap-2 pt-2">
                      <RiskGauge probability={current.probability_malignant} />
                      <p className="text-sm font-semibold">{current.predicted_class}</p>
                      {current.is_demo && <Badge tier="high">DEMO — not a real prediction</Badge>}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>Clinical summary</CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm leading-relaxed">
                      {narrativeFor(current.predicted_class, current.probability_malignant, current.is_demo)}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>GLCM texture features (this image)</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <TextureFeatureChart features={current.texture_features} />
                    </CardContent>
                  </Card>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" loading={downloading === "csv"} onClick={() => download("csv")}>
                      <Download className="h-3.5 w-3.5" /> CSV report
                    </Button>
                    <Button variant="outline" size="sm" loading={downloading === "pdf"} onClick={() => download("pdf")}>
                      <FileDown className="h-3.5 w-3.5" /> Download clinical report (PDF)
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <DisclaimerBanner />
    </div>
  );
}
