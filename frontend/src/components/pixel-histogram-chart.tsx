import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PixelHistogram } from "@/lib/api";

/** Pixel-intensity distribution of this specific scan, post grayscale/CLAHE/
 * denoise (src/feature_engineering/texture_features.py::pixel_intensity_histogram).
 * Deliberately a single-image histogram, not a "vs. benign/malignant baseline"
 * overlay — this repo has no real trained-on dataset to compute honest
 * class-conditional reference curves from, so we show only what's actually
 * measurable: this scan's own pixels.
 */
export function PixelHistogramChart({ histogram }: { histogram: PixelHistogram }) {
  const data = histogram.bin_centers.map((center, i) => ({ center, count: histogram.counts[i] }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border" />
          <XAxis
            dataKey="center"
            tickFormatter={(v: number) => v.toFixed(0)}
            fontSize={11}
            tickLine={false}
            axisLine={false}
            interval={5}
          />
          <YAxis fontSize={11} tickLine={false} axisLine={false} width={36} />
          <Tooltip
            formatter={(value) => [value, "pixels"]}
            labelFormatter={(label) => `intensity ~${Number(label).toFixed(0)}`}
            contentStyle={{ fontSize: 12, borderRadius: 8 }}
          />
          <Bar dataKey="count" fill="hsl(var(--primary))" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="mt-1 text-center text-[11px] text-muted-foreground">
        mean intensity {histogram.mean_intensity.toFixed(1)} · std {histogram.std_intensity.toFixed(1)}
      </p>
    </div>
  );
}
