import { useEffect, useMemo, useState } from "react";
import { Dropzone } from "@/components/ui/dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { StagedLoader } from "@/components/ui/staged-loader";
import { RiskGauge } from "@/components/ui/risk-gauge";
import { TextureFeatureChart } from "@/components/texture-feature-chart";
import { PixelHistogramChart } from "@/components/pixel-histogram-chart";
import { ZoomableImage } from "@/components/zoomable-image";
import { DisclaimerBanner } from "@/components/layout/disclaimer-banner";
import { ApiError, mammographyApi, reportsApi, triggerDownload, type MammographyPredictResult } from "@/lib/api";
import { riskTier } from "@/lib/utils";
import { AlertTriangle, Download, FileDown } from "lucide-react";

const STAGE_MESSAGES = [
  "Detecting input format (DICOM / 16-bit TIF / RGB)...",
  "Converting to grayscale, resizing, CLAHE enhancement...",
  "Running lesion-guided ResNet50 inference...",
  "Generating Grad-CAM heatmap over the mammogram...",
];

function specimenTag(filename: string): string {
  let hash = 0;
  for (const ch of filename) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return `MAMMO-${hash.toString(36).toUpperCase().padStart(6, "0").slice(0, 6)}`;
}

function DemoModeBanner() {
  return (
    <div className="mt-4 flex items-start gap-2 rounded-lg border-2 border-risk-high-fg/40 bg-risk-high-bg/60 px-4 py-3 text-sm text-risk-high-fg">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>
        <strong>Demo mode.</strong> This lesion-guided model has ImageNet-pretrained weights only — it has never
        been trained on real mammography data (CBIS-DDSM/MIAS/INbreast/VinDr-Mammo). Predictions and the
        estimated-lesion-area/attention numbers below are not meaningful. See{" "}
        <code className="font-mono-num">research/mammography_generalization/README.md</code> for how to train a
        real checkpoint.
      </span>
    </div>
  );
}

export function Mammography() {
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);
  const [modelIsDemo, setModelIsDemo] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<MammographyPredictResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const [downloading, setDownloading] = useState<"csv" | "pdf" | null>(null);
  const [analyzedAt, setAnalyzedAt] = useState<string>("");

  useEffect(() => {
    mammographyApi.status().then((s) => {
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
      const res = await mammographyApi.predictBatch(files);
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
      const payload = {
        predicted_class: result.predicted_class,
        probability_malignant: result.probability_malignant,
        source: result.filename,
        gradcam_png_base64: result.gradcam_png_base64,
        is_demo: result.is_demo,
      };
      const blob = kind === "csv" ? await reportsApi.downloadCsv(payload) : await reportsApi.downloadPdf(payload);
      triggerDownload(blob, `cellscan_${specimenTag(result.filename)}_report.${kind}`);
    } finally {
      setDownloading(null);
    }
  }

  const current = results?.[selected];
  const currentFile = files[selected];
  const currentRawPreviewUrl = currentFile?.type.startsWith("image/") ? previewUrls[selected] : null;

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Mammography Lesion-Guided Prediction</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Upload a mammogram (DICOM, 16-bit TIF, or standard PNG/JPG — any of these is auto-detected and normalized).
        CellScan converts it to grayscale, resizes to 512x512, applies CLAHE and denoising, then scores it with a
        ResNet50 backbone trained with an auxiliary lesion-segmentation head so its features attend to mass/
        calcification patterns rather than scanner-specific artifacts.
      </p>

      {modelAvailable === false && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-risk-mid-fg/25 bg-risk-mid-bg/50 px-4 py-3 text-sm text-risk-mid-fg">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            No mammography checkpoint or demo weights found. Run{" "}
            <code className="font-mono-num">python scripts/generate_demo_weights.py --target mammography</code> to
            test this page immediately.
          </span>
        </div>
      )}
      {modelAvailable === true && modelIsDemo && <DemoModeBanner />}

      <div className="mt-6">
        <Dropzone
          onFiles={handleFiles}
          accept={{ "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif", ".tiff"], "application/dicom": [".dcm"] }}
          multiple
          hint="DICOM, 16-bit TIF, or PNG/JPG mammograms"
        />
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap gap-3">
            {files.map((f, i) => (
              <figure key={f.name + i} className="w-24">
                {f.type.startsWith("image/") ? (
                  <img src={previewUrls[i]} className="aspect-square w-24 rounded-lg border border-border object-cover" />
                ) : (
                  <div className="flex aspect-square w-24 items-center justify-center rounded-lg border border-border bg-muted text-[10px] text-muted-foreground">
                    DICOM
                  </div>
                )}
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
              {results.length} mammogram(s) scored —{" "}
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
              <div className="grid grid-cols-2 gap-x-6 gap-y-2 rounded-lg border border-border bg-muted/50 px-4 py-3 sm:grid-cols-4">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Specimen ID</div>
                  <div className="truncate text-xs font-semibold">{specimenTag(current.filename)}</div>
                </div>
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Analyzed</div>
                  <div className="truncate text-xs font-semibold">{analyzedAt}</div>
                </div>
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Modality</div>
                  <div className="truncate text-xs font-semibold">Mammography</div>
                </div>
                <div>
                  <div className={`truncate text-xs font-semibold ${current.is_demo ? "text-risk-high-fg" : ""}`}>
                    <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Model</div>
                    {current.model_key.replace(/_/g, " ")}
                    {current.is_demo ? " (DEMO)" : ""}
                  </div>
                </div>
              </div>

              {/* Top — enlarged, zoomable side-by-side scan inspection */}
              <div className={`grid gap-3 ${currentRawPreviewUrl ? "sm:grid-cols-3" : "sm:grid-cols-2"}`}>
                {currentRawPreviewUrl && <ZoomableImage src={currentRawPreviewUrl} label="Raw uploaded scan" />}
                <ZoomableImage
                  src={`data:image/png;base64,${current.preprocessed_png_base64}`}
                  label="Grayscale + CLAHE"
                  caption="processed input"
                />
                <ZoomableImage
                  src={`data:image/png;base64,${current.gradcam_png_base64}`}
                  label="Grad-CAM overlay"
                  caption="model attention"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Warmer regions in the heatmap contributed more strongly to the predicted class — for a
                lesion-guided model this should concentrate on mass/calcification regions rather than scanner
                artifacts.
                {!currentRawPreviewUrl && " (Raw scan preview unavailable for DICOM input — browsers can't decode it directly.)"}
              </p>

              {/* Middle — high-contrast diagnostic summary */}
              <Card>
                <CardContent className="grid gap-6 pt-5 sm:grid-cols-[auto_1fr] sm:items-center">
                  <div className="flex flex-col items-center gap-2">
                    <RiskGauge probability={current.probability_malignant} />
                    <Badge tier={riskTier(current.probability_malignant).tier} className="text-sm">
                      {current.predicted_class}
                    </Badge>
                    {current.is_demo && <Badge tier="high">DEMO — not a real prediction</Badge>}
                  </div>
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Clinical indicators
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-x-6 gap-y-1 text-sm">
                      <span>
                        Estimated lesion area{" "}
                        <strong className="font-mono-num">
                          {(current.explanation.estimated_lesion_area_fraction * 100).toFixed(1)}%
                        </strong>
                      </span>
                      <span>
                        Attention concentration{" "}
                        <strong className="font-mono-num">
                          {(current.explanation.attention_concentration * 100).toFixed(1)}%
                        </strong>
                      </span>
                    </div>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{current.explanation.summary}</p>
                    <div className="mt-4 flex gap-2">
                      <Button variant="outline" size="sm" loading={downloading === "csv"} onClick={() => download("csv")}>
                        <Download className="h-3.5 w-3.5" /> CSV report
                      </Button>
                      <Button variant="outline" size="sm" loading={downloading === "pdf"} onClick={() => download("pdf")}>
                        <FileDown className="h-3.5 w-3.5" /> Download clinical report (PDF)
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Bottom — per-image analytics */}
              <div className="grid gap-4 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Pixel intensity distribution (this scan)</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <PixelHistogramChart histogram={current.pixel_histogram} />
                    <p className="mt-2 text-[11px] text-muted-foreground">
                      Computed from this image's own post-CLAHE pixels. Not shown: a comparison against
                      benign/malignant reference distributions — this environment has no real trained-on dataset
                      to compute honest class-conditional baselines from.
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>GLCM texture features (this scan)</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <TextureFeatureChart features={current.texture_features} />
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </div>
      )}

      <DisclaimerBanner />
    </div>
  );
}
