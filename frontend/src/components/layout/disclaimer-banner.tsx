import { ShieldAlert } from "lucide-react";

export function DisclaimerBanner() {
  return (
    <div className="mt-8 flex items-start gap-3 rounded-xl border border-risk-mid-fg/25 bg-risk-mid-bg/60 px-4 py-3 text-sm text-risk-mid-fg">
      <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
      <p>
        <strong className="font-semibold">Research use only.</strong> CellScan is an educational project, not a
        diagnostic device. It has not been reviewed by any regulatory body and must not inform real clinical
        decisions. See the About page for the full disclaimer.
      </p>
    </div>
  );
}
