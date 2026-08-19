import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ResponsiveContainer } from "recharts";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { clustersApi, type ClusterPoint } from "@/lib/api";

type Method = "pca" | "tsne";
type ColorBy = "trueDiagnosis" | "kmeansCluster";

const DIAGNOSIS_COLORS: Record<string, string> = { Benign: "#2a9d8f", Malignant: "#e76f51" };
const CLUSTER_COLORS = ["#457b9d", "#f4a261", "#8338ec", "#264653"];

function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-muted p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-semibold transition-colors",
            value === o.value ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ClusterExplorer() {
  const [method, setMethod] = useState<Method>("pca");
  const [colorBy, setColorBy] = useState<ColorBy>("trueDiagnosis");
  const [points, setPoints] = useState<ClusterPoint[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    clustersApi
      .projection(method)
      .then((r) => setPoints(r.points))
      .finally(() => setLoading(false));
  }, [method]);

  const groups = useMemo(() => {
    if (!points) return {};
    const g: Record<string, ClusterPoint[]> = {};
    for (const p of points) {
      const key = p[colorBy];
      (g[key] ??= []).push(p);
    }
    return g;
  }, [points, colorBy]);

  const colorFor = (key: string) => (colorBy === "trueDiagnosis" ? DIAGNOSIS_COLORS[key] : CLUSTER_COLORS[Number(key) % CLUSTER_COLORS.length]);

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Unsupervised Cluster Explorer</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        How well does the feature space separate benign from malignant without ever using the diagnosis label?
        PCA and t-SNE project the 30-dimensional feature space down to 2D so the clustering can be inspected
        visually.
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-4">
        <SegmentedControl
          value={method}
          onChange={setMethod}
          options={[
            { value: "pca", label: "PCA" },
            { value: "tsne", label: "t-SNE" },
          ]}
        />
        <SegmentedControl
          value={colorBy}
          onChange={setColorBy}
          options={[
            { value: "trueDiagnosis", label: "True diagnosis" },
            { value: "kmeansCluster", label: "KMeans cluster" },
          ]}
        />
      </div>

      <div className="mt-6 rounded-xl border border-border bg-card p-4">
        {loading || !points ? (
          <div className="flex h-[420px] items-center justify-center">
            <Spinner label={`Computing ${method.toUpperCase()} projection...`} />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis type="number" dataKey="x" name="dim 1" fontSize={11} />
              <YAxis type="number" dataKey="y" name="dim 2" fontSize={11} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {Object.entries(groups).map(([key, pts]) => (
                <Scatter key={key} name={key} data={pts} fill={colorFor(key)} opacity={0.75} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>

      <p className="mt-3 text-xs text-muted-foreground">
        If the KMeans-colored plot roughly mirrors the true-diagnosis-colored plot, the clustering is recovering
        the diagnosis structure without ever seeing the labels — see the Model Performance page for the ARI/NMI
        scores that quantify this.
      </p>
    </div>
  );
}
