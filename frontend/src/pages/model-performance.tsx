import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Spinner } from "@/components/ui/spinner";
import { reportsApi, type ConfusionMatrix, type ModelMetrics, type RocCurve } from "@/lib/api";

const CURVE_COLORS = ["#457b9d", "#2a9d8f", "#e76f51", "#8338ec", "#f4a261", "#1d3557", "#264653"];

function ConfusionMatrixGrid({ matrix }: { matrix: ConfusionMatrix }) {
  const total = matrix.tn + matrix.fp + matrix.fn + matrix.tp;
  const cells = [
    { label: "True Negative", value: matrix.tn, kind: "correct" as const },
    { label: "False Positive", value: matrix.fp, kind: "error" as const },
    { label: "False Negative", value: matrix.fn, kind: "critical" as const },
    { label: "True Positive", value: matrix.tp, kind: "correct" as const },
  ];
  const styles = {
    correct: "bg-risk-low-bg text-risk-low-fg border-risk-low-fg/20",
    error: "bg-risk-mid-bg text-risk-mid-fg border-risk-mid-fg/20",
    // false negatives (missed malignant cases) get the strongest visual weight —
    // that's the error that actually matters clinically, not a cosmetic choice
    critical: "bg-risk-high-bg text-risk-high-fg border-risk-high-fg/30 ring-1 ring-risk-high-fg/30",
  };

  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-xs">
        <div />
        <div className="pb-1 text-center font-semibold text-muted-foreground">Predicted Benign</div>
        <div className="pb-1 text-center font-semibold text-muted-foreground">Predicted Malignant</div>

        <div className="flex items-center justify-end pr-2 font-semibold text-muted-foreground">Actual Benign</div>
        {[cells[0], cells[1]].map((c) => (
          <div key={c.label} className={`rounded-lg border p-3 text-center ${styles[c.kind]}`}>
            <div className="text-xl font-bold font-mono-num">{c.value}</div>
            <div className="mt-0.5 text-[10px] uppercase tracking-wide opacity-80">
              {((c.value / total) * 100).toFixed(0)}% · {c.label}
            </div>
          </div>
        ))}

        <div className="flex items-center justify-end pr-2 font-semibold text-muted-foreground">Actual Malignant</div>
        {[cells[2], cells[3]].map((c) => (
          <div key={c.label} className={`rounded-lg border p-3 text-center ${styles[c.kind]}`}>
            <div className="text-xl font-bold font-mono-num">{c.value}</div>
            <div className="mt-0.5 text-[10px] uppercase tracking-wide opacity-80">
              {((c.value / total) * 100).toFixed(0)}% · {c.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ModelPerformance() {
  const [models, setModels] = useState<ModelMetrics[] | null>(null);
  const [curves, setCurves] = useState<RocCurve[] | null>(null);
  const [matrices, setMatrices] = useState<ConfusionMatrix[] | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("ensemble_voting");
  const [smote, setSmote] = useState<Record<string, unknown>[] | null>(null);
  const [clustering, setClustering] = useState<Record<string, unknown>[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      reportsApi.tabularComparison().then((r) => setModels(r.models)),
      reportsApi.rocCurves().then((r) => setCurves(r.curves)),
      reportsApi.confusionMatrices().then((r) => setMatrices(r.matrices)),
      reportsApi.smoteComparison().then((r) => setSmote(r.rows)),
      reportsApi.clusteringSummary().then((r) => setClustering(r.methods)),
    ]).finally(() => setLoading(false));
  }, []);

  const activeMatrix = matrices?.find((m) => m.name === selectedModel) ?? matrices?.[0];

  const notTrainedYet = !loading && models === null;

  // merge ROC points into one dataset keyed by fpr for a multi-line chart
  const rocData = curves
    ? Array.from({ length: 61 }, (_, i) => {
        const point: Record<string, number> = { fpr: i / 60 };
        curves.forEach((c) => {
          const nearest = c.points.reduce((a, b) => (Math.abs(b.fpr - point.fpr) < Math.abs(a.fpr - point.fpr) ? b : a));
          point[c.name] = nearest.tpr;
        });
        return point;
      })
    : [];

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Model Performance Comparison</h1>

      {loading && <Spinner className="mt-6" label="Loading results..." />}

      {notTrainedYet && (
        <Card className="mt-6">
          <CardContent className="pt-5 text-sm text-muted-foreground">
            No results yet — run <code className="font-mono-num">python scripts/train_tabular.py</code> (and
            optionally <code className="font-mono-num">python -m src.api.main</code> to serve them) to populate
            this page.
          </CardContent>
        </Card>
      )}

      {models && (
        <div className="mt-6 space-y-8">
          <div>
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted-foreground">
              Tabular models — held-out test set
            </h2>
            <Table>
              <THead>
                <TR>
                  <TH>Model</TH>
                  <TH>Accuracy</TH>
                  <TH>Precision</TH>
                  <TH>Recall</TH>
                  <TH>F1</TH>
                  <TH>ROC AUC</TH>
                </TR>
              </THead>
              <TBody>
                {models.map((m) => (
                  <TR key={m.name}>
                    <TD className="font-sans font-medium">{m.name.replace(/_/g, " ")}</TD>
                    <TD>{m.accuracy.toFixed(3)}</TD>
                    <TD>{m.precision.toFixed(3)}</TD>
                    <TD className="font-semibold text-accent">{m.recall.toFixed(3)}</TD>
                    <TD>{m.f1.toFixed(3)}</TD>
                    <TD>{m.roc_auc.toFixed(3)}</TD>
                  </TR>
                ))}
              </TBody>
            </Table>
            <p className="mt-2 text-xs text-muted-foreground">
              Sorted by recall — minimizing missed malignant cases is the priority metric here, not raw accuracy.
            </p>
          </div>

          {curves && curves.length > 0 && (
            <div>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted-foreground">ROC curves</h2>
              <Card>
                <CardContent className="pt-5">
                  <ResponsiveContainer width="100%" height={360}>
                    <LineChart data={rocData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                      <XAxis dataKey="fpr" tickFormatter={(v) => v.toFixed(1)} label={{ value: "False Positive Rate", position: "insideBottom", offset: -4, fontSize: 11 }} fontSize={11} />
                      <YAxis domain={[0, 1]} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 11 }} fontSize={11} />
                      <Tooltip formatter={(v) => Number(v).toFixed(3)} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} formatter={(v) => v.replace(/_/g, " ")} />
                      {curves.map((c, i) => (
                        <Line key={c.name} type="monotone" dataKey={c.name} stroke={CURVE_COLORS[i % CURVE_COLORS.length]} dot={false} strokeWidth={2} name={`${c.name} (AUC=${c.auc.toFixed(3)})`} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}

          {matrices && matrices.length > 0 && activeMatrix && (
            <div>
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wide text-muted-foreground">Confusion matrix</h2>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="h-8 rounded-md border border-border bg-background px-2 text-xs font-medium"
                >
                  {matrices.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>
              <Card>
                <CardContent className="pt-5">
                  <ConfusionMatrixGrid matrix={activeMatrix} />
                </CardContent>
              </Card>
              <p className="mt-2 text-xs text-muted-foreground">
                False negatives (highlighted) are missed malignant cases — the error this system is tuned to
                minimize, at the cost of tolerating more false positives.
              </p>
            </div>
          )}

          {smote && (
            <div>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted-foreground">
                SMOTE vs. baseline (Logistic Regression)
              </h2>
              <Table>
                <THead>
                  <TR>
                    <TH>Run</TH>
                    <TH>Accuracy</TH>
                    <TH>Precision</TH>
                    <TH>Recall</TH>
                    <TH>F1</TH>
                    <TH>ROC AUC</TH>
                  </TR>
                </THead>
                <TBody>
                  {smote.map((row) => (
                    <TR key={String(row.name)}>
                      <TD className="font-sans">{String(row.name).replace(/_/g, " ")}</TD>
                      <TD>{Number(row.accuracy).toFixed(3)}</TD>
                      <TD>{Number(row.precision).toFixed(3)}</TD>
                      <TD>{Number(row.recall).toFixed(3)}</TD>
                      <TD>{Number(row.f1).toFixed(3)}</TD>
                      <TD>{Number(row.roc_auc).toFixed(3)}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
            </div>
          )}

          {clustering && (
            <div>
              <h2 className="mb-3 text-sm font-bold uppercase tracking-wide text-muted-foreground">
                Clustering quality vs. true diagnosis labels
              </h2>
              <Table>
                <THead>
                  <TR>
                    <TH>Method</TH>
                    <TH>Silhouette</TH>
                    <TH>ARI</TH>
                    <TH>NMI</TH>
                  </TR>
                </THead>
                <TBody>
                  {clustering.map((row) => (
                    <TR key={String(row.name)}>
                      <TD className="font-sans">{String(row.name)}</TD>
                      <TD>{Number(row.silhouette).toFixed(3)}</TD>
                      <TD>{Number(row.ari).toFixed(3)}</TD>
                      <TD>{Number(row.nmi).toFixed(3)}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
              <p className="mt-2 text-xs text-muted-foreground">
                Silhouette measures cluster cohesion; ARI/NMI measure agreement with the true benign/malignant
                split.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
