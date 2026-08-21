import { useState } from "react";
import { AnimatePresence } from "framer-motion";
import { Menu } from "lucide-react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./sidebar";
import { PageTransition } from "./page-transition";

export function Layout() {
  const location = useLocation();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background md:flex-row">
      <Sidebar mobileOpen={mobileNavOpen} onCloseMobile={() => setMobileNavOpen(false)} />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* mobile-only top bar — the sidebar is an off-canvas drawer below md */}
        <header className="flex items-center gap-3 border-b border-border bg-card px-4 py-3 md:hidden">
          <button
            onClick={() => setMobileNavOpen(true)}
            className="rounded-md p-1.5 text-foreground hover:bg-muted"
            aria-label="Open menu"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-accent to-primary text-[10px] font-bold text-white">
              CS
            </div>
            <span className="text-sm font-bold">CellScan</span>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-10 lg:py-8">
          <div className="mx-auto max-w-5xl">
            <AnimatePresence mode="wait">
              <PageTransition key={location.pathname}>
                <Outlet />
              </PageTransition>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
}
