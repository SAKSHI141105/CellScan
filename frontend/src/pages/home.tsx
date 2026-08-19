import { Link } from "react-router-dom";
import { ArrowRight, Brain, FileSpreadsheet, Image as ImageIcon } from "lucide-react";
import { SpotlightCard } from "@/components/ui/spotlight-card";
import { DisclaimerBanner } from "@/components/layout/disclaimer-banner";

const FEATURE_CARDS = [
  {
    to: "/clinical-data",
    icon: FileSpreadsheet,
    title: "Clinical Data",
    body: "Enter the 30 diagnostic measurements, or upload a CSV of many patients for batch scoring with SHAP explanations.",
  },
  {
    to: "/upload-image",
    icon: ImageIcon,
    title: "Histopathology Image",
    body: "Upload tissue patches. CellScan preprocesses them automatically and returns a Grad-CAM heatmap alongside the prediction.",
  },
  {
    to: "/model-performance",
    icon: Brain,
    title: "Model Insight",
    body: "Compare every trained model's metrics, inspect ROC curves, and explore how unsupervised clustering separates the two classes.",
  },
];

export function Home() {
  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Multi-modal breast tissue classification, with explanations</h1>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        CellScan runs two independent pipelines over the Wisconsin Diagnostic Breast Cancer dataset and
        histopathology image patches — one on structured clinical measurements, one on tissue imagery — and pairs
        each prediction with a visual explanation of what drove it, rather than returning a bare label.
      </p>

      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        {FEATURE_CARDS.map(({ to, icon: Icon, title, body }) => (
          <Link key={to} to={to}>
            <SpotlightCard className="h-full transition-transform hover:-translate-y-0.5">
              <Icon className="h-5 w-5 text-accent" />
              <h3 className="mt-3 text-sm font-semibold">{title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">{body}</p>
              <div className="mt-3 flex items-center gap-1 text-xs font-medium text-accent">
                Open <ArrowRight className="h-3 w-3" />
              </div>
            </SpotlightCard>
          </Link>
        ))}
      </div>

      <DisclaimerBanner />
    </div>
  );
}
