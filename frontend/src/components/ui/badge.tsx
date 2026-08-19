import type { HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold", {
  variants: {
    tier: {
      low: "bg-risk-low-bg text-risk-low-fg",
      moderate: "bg-risk-mid-bg text-risk-mid-fg",
      high: "bg-risk-high-bg text-risk-high-fg",
      neutral: "bg-muted text-muted-foreground",
    },
  },
  defaultVariants: { tier: "neutral" },
});

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, tier, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tier }), className)} {...props} />;
}
