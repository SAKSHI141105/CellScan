import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TextureFeatures } from "@/lib/api";

const DISPLAY_FEATURES: { key: keyof TextureFeatures; label: string }[] = [
  { key: "glcm_contrast", label: "GLCM Contrast" },
  { key: "glcm_homogeneity", label: "Homogeneity" },
  { key: "glcm_energy", label: "Energy" },
  { key: "glcm_correlation", label: "Correlation" },
  { key: "glcm_dissimilarity", label: "Dissimilarity" },
  { key: "canny_edge_density", label: "Edge Density" },
];

/** Bar chart of GLCM texture + edge features computed directly from this
 * specific uploaded image (src/feature_engineering/texture_features.py) —
 * not a canned example, a real per-image measurement. Features are on very
 * different natural scales (contrast can run into the hundreds, energy/
 * correlation live in [0,1]), so each bar is normalized against this same
 * image's own value set purely for a legible chart — the tooltip shows the
 * real unnormalized number.
 */
export function TextureFeatureChart({ features }: { features: TextureFeatures }) {
  const rows = DISPLAY_FEATURES.map(({ key, label }) => ({ label, raw: features[key] ?? 0 }));
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.raw)), 1e-6);
  const data = rows.map((r) => ({ ...r, normalized: r.raw / maxAbs }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} className="stroke-border" />
        <XAxis type="number" domain={[0, 1]} hide />
        <YAxis type="category" dataKey="label" width={100} fontSize={11} tickLine={false} axisLine={false} />
        <Tooltip
          formatter={(_value, _name, item) => [item.payload.raw.toFixed(4), "value"]}
          contentStyle={{ fontSize: 12, borderRadius: 8 }}
        />
        <Bar dataKey="normalized" fill="hsl(var(--accent))" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
