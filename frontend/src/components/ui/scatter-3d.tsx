import { useMemo } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import { useTheme } from "@/contexts/theme-context";

const Plot = createPlotlyComponent(Plotly);

export interface Point3D {
  x: number;
  y: number;
  z: number;
  group: string;
}

const THEME_COLORS = {
  light: { paper: "#ffffff", font: "#0b1f2a", grid: "#dfe7ec" },
  dark: { paper: "#0d151c", font: "#ebf0f4", grid: "#26313a" },
};

// fixed palette so the same group always gets the same color across
// method/dimension switches — avoids the "wait, did colors just swap" confusion
const GROUP_COLORS: Record<string, string> = {
  Benign: "#2a9d8f",
  Malignant: "#e76f51",
  "0": "#457b9d",
  "1": "#f4a261",
  "2": "#8338ec",
  "3": "#264653",
};

export function Scatter3D({ points, height = 480 }: { points: Point3D[]; height?: number }) {
  const { resolvedTheme } = useTheme();
  const colors = THEME_COLORS[resolvedTheme];

  const traces = useMemo(() => {
    const groups = Array.from(new Set(points.map((p) => p.group)));
    return groups.map((group) => {
      const groupPoints = points.filter((p) => p.group === group);
      return {
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: group,
        x: groupPoints.map((p) => p.x),
        y: groupPoints.map((p) => p.y),
        z: groupPoints.map((p) => p.z),
        marker: { size: 4, opacity: 0.8, color: GROUP_COLORS[group] ?? "#94a3b8" },
      };
    });
  }, [points]);

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height,
        margin: { l: 0, r: 0, t: 10, b: 0 },
        paper_bgcolor: colors.paper,
        font: { color: colors.font, size: 11 },
        legend: { orientation: "h", y: -0.02 },
        scene: {
          xaxis: { title: { text: "dim 1" }, gridcolor: colors.grid, zerolinecolor: colors.grid },
          yaxis: { title: { text: "dim 2" }, gridcolor: colors.grid, zerolinecolor: colors.grid },
          zaxis: { title: { text: "dim 3" }, gridcolor: colors.grid, zerolinecolor: colors.grid },
        },
      }}
      config={{ displayModeBar: true, displaylogo: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}
