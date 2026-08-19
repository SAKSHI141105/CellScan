import { useEffect, useMemo, useState } from "react";
import { Dropzone } from "@/components/ui/dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { PredictionResultCard } from "@/components/prediction-result-card";
import { DisclaimerBanner } from "@/components/layout/disclaimer-banner";
import { ApiError, imageApi, reportsApi, triggerDownload, type ImagePredictResult } from "@/lib/api";
import { riskTier } from "@/lib/utils";
import { AlertTriangle, Download, FileDown } from "lucide-react";

export function UploadImage() {
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ImagePredictResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const [downloading, setDownloading] = useState<"csv" | "pdf" | null>(null);

  useEffect(() => {
    imageApi.status().then((s) => setModelAvailable(s.available));
  }, []);

  const previewUrls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);

  function handleFiles(accepted: File[]) {
    setFiles(accepted);
    setResults(null);
    setSelected(0);
  }

  async function runPrediction() {
    if (!files.length) return;
    setLoading(true);
    setError(null);
    try {
      const res = await imageApi.predictBatch(files);
      setResults(res.results);
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
      const payload = { ...result, source: result.filename };
      const blob = kind === "csv" ? await reportsApi.downloadCsv(payload) : await reportsApi.downloadPdf(payload);
      triggerDownload(blob, `cellscan_image_report.${kind}`);
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
            No trained image model found yet. Run <code className="font-mono-num">python scripts/train_image.py</code>{" "}
            after downloading the histopathology dataset (see README) to enable predictions.
          </span>
        </div>
      )}

      <div className="mt-6">
        <Dropzone
          onFiles={handleFiles}
          accept={{ "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif"] }}
          multiple
          hint="One or more PNG/JPG/TIF tissue patches"
        />
      </div>

      {files.length > 0 && (
        <div className="mt-4 flex items-center gap-3">
          <span className="text-sm text-muted-foreground">{files.length} file(s) selected</span>
          <Button onClick={runPrediction} loading={loading}>
            Run prediction
          </Button>
        </div>
      )}

      {error && <p className="mt-3 text-sm text-risk-high-fg">{error}</p>}
      {loading && <Spinner className="mt-3" label="Running CNN inference..." />}

      {results && (
        <div className="mt-6 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Batch results</CardTitle>
            </CardHeader>
            <CardContent className="text-sm">
              {results.length} image(s) scored —{" "}
              <strong>{results.filter((r) => r.predicted_class === "Malignant").length}</strong> flagged malignant.
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
                    <TD>
                      <Badge tier={tier}>{label}</Badge>
                    </TD>
                  </TR>
                );
              })}
            </TBody>
          </Table>

          {current && (
            <div className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <figure>
                  <img src={currentPreviewUrl} className="w-full rounded-lg border border-border" />
                  <figcaption className="mt-1.5 text-xs text-muted-foreground">Original upload</figcaption>
                </figure>
                <figure>
                  <img
                    src={`data:image/png;base64,${current.preprocessed_png_base64}`}
                    className="w-full rounded-lg border border-border"
                  />
                  <figcaption className="mt-1.5 text-xs text-muted-foreground">
                    Preprocessed (grayscale + CLAHE + denoise)
                  </figcaption>
                </figure>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <div className="space-y-4">
                  <PredictionResultCard predictedClass={current.predicted_class} probability={current.probability_malignant} />
                  <p className="text-xs text-muted-foreground">Model: {current.model_key}</p>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" loading={downloading === "csv"} onClick={() => download("csv")}>
                      <Download className="h-3.5 w-3.5" /> CSV report
                    </Button>
                    <Button variant="outline" size="sm" loading={downloading === "pdf"} onClick={() => download("pdf")}>
                      <FileDown className="h-3.5 w-3.5" /> PDF report
                    </Button>
                  </div>
                </div>
                <Card>
                  <CardHeader>
                    <CardTitle>Grad-CAM — regions driving the prediction</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <img
                      src={`data:image/png;base64,${current.gradcam_png_base64}`}
                      className="w-full rounded-lg border border-border"
                    />
                    <p className="mt-2 text-xs text-muted-foreground">
                      Warmer regions contributed more strongly to the predicted class.
                    </p>
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
