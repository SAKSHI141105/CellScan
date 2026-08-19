import { Card, CardContent } from "@/components/ui/card";
import { ShieldAlert } from "lucide-react";

export function About() {
  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">About CellScan</h1>
      <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        CellScan is a research and portfolio project exploring breast tissue classification from two angles at
        once: structured clinical measurements (the Wisconsin Diagnostic Breast Cancer dataset) and
        histopathology image patches. It combines supervised classifiers, unsupervised clustering, and
        explainability tooling (SHAP, LIME, Grad-CAM) behind a single interface.
      </p>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
        It was built to demonstrate a full ML system end to end — data cleaning through a real API-backed
        frontend — not to be used as a medical device.
      </p>

      <Card className="mt-6 border-risk-mid-fg/25 bg-risk-mid-bg/40">
        <CardContent className="pt-5">
          <div className="mb-2 flex items-center gap-2 text-risk-mid-fg">
            <ShieldAlert className="h-4 w-4" />
            <h2 className="text-sm font-bold">Disclaimer</h2>
          </div>
          <div className="space-y-3 text-sm leading-relaxed text-risk-mid-fg/90">
            <p>
              <strong>This is not a diagnostic tool.</strong> CellScan is an educational and research prototype.
              It has not been validated on an independent clinical cohort, has not undergone any regulatory
              review, and carries no clearance from the FDA, CE, or any equivalent body.
            </p>
            <p>
              Predictions shown here must never be used to make, support, or influence an actual medical
              decision. Breast tissue diagnosis requires a licensed pathologist reviewing histology under a
              microscope, correlated with clinical and radiological findings — a single model score is not a
              substitute for that process, no matter how confident the displayed percentage looks.
            </p>
            <p>
              If you are looking at this because you or someone you know is concerned about a real diagnosis,
              please consult a qualified physician.
            </p>
          </div>
        </CardContent>
      </Card>

      <h2 className="mt-8 text-sm font-bold uppercase tracking-wide text-muted-foreground">What's under the hood</h2>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <Card>
          <CardContent className="pt-5">
            <h3 className="mb-2 text-sm font-semibold">Tabular pipeline</h3>
            <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-muted-foreground">
              <li>Wisconsin Diagnostic Breast Cancer dataset, 30 features</li>
              <li>Correlation-based feature pruning + Random Forest importance ranking</li>
              <li>SMOTE for class balancing</li>
              <li>Logistic Regression, Random Forest, XGBoost, SVM, MLP — each tuned via Grid/RandomizedSearchCV with stratified 5-fold CV</li>
              <li>Voting + stacking ensembles over the top-performing models</li>
              <li>SHAP (global + per-prediction) and LIME explanations</li>
            </ul>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-5">
            <h3 className="mb-2 text-sm font-semibold">Image pipeline</h3>
            <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-muted-foreground">
              <li>Histopathology patches (BreakHis / Kaggle IDC format)</li>
              <li>Grayscale, CLAHE, denoising, augmentation</li>
              <li>GLCM texture + Canny/Sobel edge features (classical route)</li>
              <li>Custom CNN + transfer learning (ResNet50 / EfficientNetB0 / VGG16)</li>
              <li>Grad-CAM for visual explanation</li>
              <li>Conv-autoencoder reconstruction error as an unsupervised anomaly signal</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <p className="mt-8 text-xs text-muted-foreground">
        See README.md in the project root for setup, folder structure, and how to reproduce the training runs.
      </p>
    </div>
  );
}
