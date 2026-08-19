import { Monitor, Moon, Sun } from "lucide-react";
import { motion } from "framer-motion";
import { useTheme } from "@/contexts/theme-context";
import { cn } from "@/lib/utils";

const OPTIONS = [
  { value: "light" as const, icon: Sun, label: "Light" },
  { value: "system" as const, icon: Monitor, label: "System" },
  { value: "dark" as const, icon: Moon, label: "Dark" },
];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div className="relative flex items-center gap-0.5 rounded-full border border-sidebar-active/60 bg-sidebar-active/30 p-0.5">
      {OPTIONS.map(({ value, icon: Icon, label }) => {
        const active = theme === value;
        return (
          <button
            key={value}
            onClick={() => setTheme(value)}
            aria-label={label}
            className={cn(
              "relative z-10 flex h-7 w-7 items-center justify-center rounded-full transition-colors",
              active ? "text-sidebar" : "text-sidebar-foreground/70 hover:text-sidebar-foreground"
            )}
          >
            {active && (
              <motion.div
                layoutId="theme-toggle-pill"
                className="absolute inset-0 rounded-full bg-sidebar-foreground"
                transition={{ type: "spring", stiffness: 500, damping: 32 }}
              />
            )}
            <Icon className="relative z-10 h-3.5 w-3.5" />
          </button>
        );
      })}
    </div>
  );
}
