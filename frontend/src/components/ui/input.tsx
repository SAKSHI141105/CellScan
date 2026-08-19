import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-9 w-full rounded-md border border-border bg-background px-3 text-sm font-mono-num outline-none transition-shadow focus:ring-2 focus:ring-ring/40",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";

export function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-medium capitalize text-muted-foreground">{label.replace(/_/g, " ")}</span>
      <Input
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(e.target.valueAsNumber || 0)}
      />
    </label>
  );
}
