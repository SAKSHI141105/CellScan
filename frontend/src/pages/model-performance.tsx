import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { Spinner } from "@/components/ui/spinner";
import { reportsApi, type ModelMetrics, type RocCurve } from "@/lib/api";

const CURVE_COLORS = ["#457b9d", "#2a9d8f", "#e76f51", "#8338ec", "#f4a261", "#1d3557", "#264653"];

export function ModelPerformance() {
  const [models, setModels] = useState<ModelMetrics[] | null>(null);
  const [curves, setCurves] = useState<RocCurve[] | null>(null);
  const [smote, setSmote] = useState<Record<string, unknown>[] | null>(null);
  const [clustering, setClustering] = useState<Record<string, unknown>[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([
      reportsApi.tabularComparison().then((r) => setModels(r.models)),
      reportsApi.rocCurves().then((r) => setCurves(r.curves)),
      reportsApi.smoteComparison().then((r) => setSmote(r.rows)),
      reportsApi.clusteringSummary().then((r) => setClustering(r.methods)),
    ]).finally(() => setLoading(false));
  }, []);

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
