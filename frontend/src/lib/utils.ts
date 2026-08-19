import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function riskTier(probability: number): { label: string; tier: "low" | "moderate" | "high" } {
  if (probability < 0.35) return { label: "Low Risk", tier: "low" };
  if (probability < 0.65) return { label: "Moderate Risk", tier: "moderate" };
  return { label: "High Risk", tier: "high" };
}
