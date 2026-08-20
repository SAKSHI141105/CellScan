import { NavLink, useNavigate } from "react-router-dom";
import { Activity, FileSpreadsheet, Image as ImageIcon, Info, LayoutDashboard, LogOut, ScanLine, ScatterChart, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/auth-context";
import { ThemeToggle } from "./theme-toggle";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/clinical-data", label: "Clinical Data", icon: FileSpreadsheet },
  { to: "/upload-image", label: "Upload Image", icon: ImageIcon },
  { to: "/mammography", label: "Mammography", icon: ScanLine },
  { to: "/model-performance", label: "Model Performance", icon: Activity },
  { to: "/cluster-explorer", label: "Cluster Explorer", icon: ScatterChart },
  { to: "/about", label: "About", icon: Info },
];

export function Sidebar() {
  const { session, logout } = useAuth();
  const navigate = useNavigate();

  function handleSignOut() {
    logout();
    navigate("/login");
  }

  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col bg-sidebar px-3 py-5 text-sidebar-foreground">
      <div className="flex items-center gap-2.5 px-2 pb-6">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent to-primary text-sm font-bold text-white">
          CS
        </div>
        <div>
          <div className="text-sm font-bold leading-tight text-white">CellScan</div>
          <div className="text-[10px] uppercase tracking-wider text-sidebar-foreground/60">Decision Support Platform</div>
        </div>
      </div>

      {session && (
        <div className="mb-4 rounded-lg bg-sidebar-active/40 px-3 py-2.5">
          <div className="truncate text-xs font-semibold text-white" title={session.name}>
            {session.name}
          </div>
          <div className="text-[10px] text-sidebar-foreground/70">{session.role}</div>
          <div className="mt-1.5 flex items-center gap-1 text-[10px] text-accent">
            <ShieldCheck className="h-3 w-3" /> MFA verified
          </div>
        </div>
      )}

      <nav className="flex flex-1 flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive
                  ? "bg-sidebar-active text-white"
                  : "text-sidebar-foreground/75 hover:bg-sidebar-active/50 hover:text-white"
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      <button
        onClick={handleSignOut}
        className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-sidebar-foreground/75 transition-colors hover:bg-sidebar-active/50 hover:text-white"
      >
        <LogOut className="h-4 w-4 shrink-0" /> Sign out
      </button>

      <div className="flex items-center justify-between border-t border-sidebar-active/60 px-2 pt-4">
        <span className="text-[11px] text-sidebar-foreground/60">Theme</span>
        <ThemeToggle />
      </div>
    </aside>
  );
}
