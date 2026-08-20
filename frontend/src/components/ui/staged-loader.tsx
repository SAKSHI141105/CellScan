import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

/**
 * Rotates through a fixed script of stage labels on a timer while some
 * actual async work runs in the background. It's cosmetic — the API call
 * doesn't report real progress — but a single static "Loading..." spinner
 * for a multi-second CNN + Grad-CAM pass reads as stuck, not slow.
 */
export function StagedLoader({ stages, stepMs = 900 }: { stages: string[]; stepMs?: number }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    setIndex(0);
    const id = setInterval(() => {
      setIndex((i) => Math.min(i + 1, stages.length - 1));
    }, stepMs);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stages.length]);

  return (
    <div className="flex items-center gap-2.5 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 shrink-0 animate-spin text-accent" />
      <span>{stages[index]}</span>
    </div>
  );
}
