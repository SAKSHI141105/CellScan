import { useEffect, useMemo, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { NumberField } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Dropzone } from "@/components/ui/dropzone";
import { Spinner } from "@/components/ui/spinner";
import { PredictionResultCard } from "@/components/prediction-result-card";
import { DisclaimerBanner } from "@/components/layout/disclaimer-banner";
import { ApiError, tabularApi, reportsApi, triggerDownload, type TabularPredictResult, type TopContributor } from "@/lib/api";
import { riskTier } from "@/lib/utils";
import { Download, FileDown } from "lucide-react";

function ShapBars({ contributors }: { contributors: TopContributor[] }) {
  const max = Math.max(...contributors.map((c) => Math.abs(c.shap_value)), 0.001);
  return (
    <div className="space-y-2">
      {contributors.map((c) => {
        const pct = (Math.abs(c.shap_value) / max) * 100;
        const positive = c.shap_value > 0;
        return (
          <div key={c.feature} className="flex items-center gap-3 text-xs">
            <span className="w-40 shrink-0 truncate text-muted-foreground">{c.feature.replace(/_/g, " ")}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className={`h-full rounded-full ${positive ? "bg-risk-high-fg" : "bg-risk-low-fg"}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="w-14 shrink-0 font-mono-num text-right">{c.shap_value.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}

function ResultPanel({ result, source }: { result: TabularPredictResult; source: string }) {
  const [downloading, setDownloading] = useState<"csv" | "pdf" | null>(null);

  async function download(kind: "csv" | "pdf") {
    setDownloading(kind);
    try {
      const payload = { ...result, source };
      const blob = kind === "csv" ? await reportsApi.downloadCsv(payload) : await reportsApi.downloadPdf(payload);
      triggerDownload(blob, `cellscan_report.${kind}`);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        <PredictionResultCard predictedClass={result.predicted_class} probability={result.probability_malignant} />
        <Card>
          <CardHeader>
            <CardTitle>Explanation</CardTitle>
          </CardHeader>
          <CardContent className="text-sm leading-relaxed">{result.explanation}</CardContent>
        </Card>
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
          <CardTitle>Top contributing features (SHAP)</CardTitle>
        </CardHeader>
        <CardContent>
          <ShapBars contributors={result.top_contributors} />
        </CardContent>
      </Card>
    </div>
  );
}

export function ClinicalData() {
  const [tab, setTab] = useState("manual");
  const [featureGroups, setFeatureGroups] = useState<Record<string, string[]>>({});
  const [values, setValues] = useState<Record<string, number>>({});
  const [loadingDefaults, setLoadingDefaults] = useState(true);

  const [manualResult, setManualResult] = useState<TabularPredictResult | null>(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState<string | null>(null);

  useEffect(() => {
    tabularApi.defaults().then((d) => {
      setFeatureGroups(d.feature_groups);
      setValues(d.values);
      setLoadingDefaults(false);
    });
  }, []);

  const allFeatures = useMemo(() => Object.values(featureGroups).flat(), [featureGroups]);

  async function runManualPrediction() {
    setManualLoading(true);
    setManualError(null);
    try {
      const result = await tabularApi.predict(values);
      setManualResult(result);
    } catch (e) {
      setManualError(e instanceof ApiError ? e.message : "Prediction failed");
    } finally {
      setManualLoading(false);
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Clinical Data Prediction</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Score a single patient by hand, or upload a CSV of many rows (same 30 WDBC-style columns) for batch
        predictions.
      </p>

      <Tabs value={tab} onValueChange={setTab} className="mt-6">
        <TabsList>
          <TabsTrigger value="manual">Manual entry</TabsTrigger>
          <TabsTrigger value="batch">Batch CSV upload</TabsTrigger>
        </TabsList>

        <TabsContent value="manual" className="pt-6">
          {loadingDefaults ? (
            <Spinner label="Loading reference values..." />
          ) : (
            <div className="space-y-6">
              {Object.entries(featureGroups).map(([group, cols]) => (
                <div key={group}>
                  <h3 className="mb-2 text-xs font-bold uppercase tracking-wide text-muted-foreground">
                    {group} features
                  </h3>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                    {cols.map((col) => (
                      <NumberField
                        key={col}
                        label={col}
                        value={values[col] ?? 0}
                        onChange={(v) => setValues((prev) => ({ ...prev, [col]: v }))}
                      />
                    ))}
                  </div>
                </div>
              ))}

              <Button onClick={runManualPrediction} loading={manualLoading}>
                Run prediction
              </Button>
              {manualError && <p className="text-sm text-risk-high-fg">{manualError}</p>}
              {manualResult && (
                <div className="pt-2">
                  <ResultPanel result={manualResult} source="manual clinical entry" />
                </div>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="batch" className="pt-6">
          <BatchUpload allFeatures={allFeatures} />
        </TabsContent>
      </Tabs>

      <DisclaimerBanner />
    </div>
  );
}

function BatchUpload({ allFeatures }: { allFeatures: string[] }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batch, setBatch] = useState<{ rows: Record<string, unknown>[]; n_malignant: number; n_benign: number } | null>(null);
  const [selectedRow, setSelectedRow] = useState<number>(0);
  const [rowResult, setRowResult] = useState<TabularPredictResult | null>(null);
  const [rowLoading, setRowLoading] = useState(false);

  async function handleFiles(files: File[]) {
    const file = files[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    setBatch(null);
    setRowResult(null);
    try {
      const result = await tabularApi.predictBatch(file);
      setBatch(result);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Batch prediction failed");
    } finally {
      setLoading(false);
    }
  }

  async function explainRow() {
    if (!batch) return;
    const row = batch.rows[selectedRow];
    const features = Object.fromEntries(allFeatures.map((f) => [f, Number(row[f])]));
    setRowLoading(true);
    try {
      const result = await tabularApi.predict(features);
      setRowResult(result);
    } finally {
      setRowLoading(false);
    }
  }

  function downloadResultsCsv() {
    if (!batch) return;
    const cols = Object.keys(batch.rows[0]);
    const csv = [cols.join(","), ...batch.rows.map((r) => cols.map((c) => r[c]).join(","))].join("\n");
    triggerDownload(new Blob([csv], { type: "text/csv" }), "cellscan_batch_results.csv");
  }

  return (
    <div className="space-y-6">
      <p className="text-xs text-muted-foreground">
        CSV needs the 30 WDBC feature columns (radius_mean, texture_mean, ... fractal_dimension_worst). Extra
        columns like id or diagnosis are ignored.
      </p>
      <Dropzone onFiles={handleFiles} accept={{ "text/csv": [".csv"] }} hint="CSV file, one row per patient" />

      {loading && <Spinner label="Scoring rows..." />}
      {error && <p className="text-sm text-risk-high-fg">{error}</p>}

      {batch && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Batch results</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm">
                {batch.rows.length} rows scored — <strong>{batch.n_malignant}</strong> flagged malignant,{" "}
                <strong>{batch.n_benign}</strong> benign.
              </p>
            </CardContent>
          </Card>

          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Results</h3>
            <Button variant="outline" size="sm" onClick={downloadResultsCsv}>
              <Download className="h-3.5 w-3.5" /> Download full results CSV
            </Button>
          </div>

          <Table>
            <THead>
              <TR>
                <TH>Row</TH>
                <TH>Predicted</TH>
                <TH>Probability</TH>
                <TH>Risk</TH>
              </TR>
            </THead>
            <TBody>
              {batch.rows.map((row, i) => {
                const prob = Number(row.probability_malignant);
                const { tier, label } = riskTier(prob);
                return (
                  <TR
                    key={i}
                    onClick={() => setSelectedRow(i)}
                    className={`cursor-pointer hover:bg-muted/60 ${selectedRow === i ? "bg-muted" : ""}`}
                  >
                    <TD>{i}</TD>
                    <TD>{String(row.predicted_class)}</TD>
                    <TD>{(prob * 100).toFixed(1)}%</TD>
                    <TD>
                      <Badge tier={tier}>{label}</Badge>
                    </TD>
                  </TR>
                );
              })}
            </TBody>
          </Table>

          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">Inspect row {selectedRow}</span>
            <Button size="sm" variant="secondary" onClick={explainRow} loading={rowLoading}>
              Explain this row
            </Button>
          </div>

          {rowResult && <ResultPanel result={rowResult} source={`batch row ${selectedRow}`} />}
        </div>
      )}
    </div>
  );
}
