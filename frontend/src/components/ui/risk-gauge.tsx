import { useEffect, useState } from "react";
import { riskTier } from "@/lib/utils";

const TIER_STROKE: Record<string, string> = {
  low: "hsl(var(--risk-low-fg))",
  moderate: "hsl(var(--risk-mid-fg))",
  high: "hsl(var(--risk-high-fg))",
};

// semicircular arc gauge, 0-100% mapped left-to-right — deliberately not a
// full donut/speedometer, that reads as "generic BI dashboard" rather than
// the single-number risk read a clinician actually wants at a glance
export function RiskGauge({ probability, size = 180 }: { probability: number; size?: number }) {
  const { tier, label } = riskTier(probability);
  const [animatedPct, setAnimatedPct] = useState(0);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnimatedPct(probability * 100));
    return () => cancelAnimationFrame(raf);
  }, [probability]);

  const radius = 80;
  const circumference = Math.PI * radius; // half-circle arc length
  const offset = circumference * (1 - animatedPct / 100);
  const width = size;
  const height = size / 2 + 24;

  return (
    <div className="flex flex-col items-center">
      <svg width={width} height={height} viewBox="0 0 200 124" className="overflow-visible">
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth={16}
          strokeLinecap="round"
        />
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={TIER_STROKE[tier]}
          strokeWidth={16}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.7s ease-out, stroke 0.3s ease" }}
        />
        <text x="100" y="90" textAnchor="middle" className="font-mono-num" style={{ fontSize: 30, fontWeight: 700, fill: "hsl(var(--foreground))" }}>
          {probability * 100 < 0.05 ? "0.0" : (probability * 100).toFixed(1)}%
        </text>
      </svg>
      <span
        className="-mt-1 rounded-full px-3 py-1 text-xs font-semibold"
        style={{ color: TIER_STROKE[tier], backgroundColor: `hsl(var(--risk-${tier === "moderate" ? "mid" : tier}-bg))` }}
      >
        {label}
      </span>
    </div>
  );
}
