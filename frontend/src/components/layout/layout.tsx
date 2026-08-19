import { AnimatePresence } from "framer-motion";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./sidebar";
import { PageTransition } from "./page-transition";

export function Layout() {
  const location = useLocation();

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <main className="flex-1 overflow-y-auto px-10 py-8">
        <div className="mx-auto max-w-5xl">
          <AnimatePresence mode="wait">
            <PageTransition key={location.pathname}>
              <Outlet />
            </PageTransition>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
