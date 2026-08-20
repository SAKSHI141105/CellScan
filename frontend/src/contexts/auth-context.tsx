import { createContext, useContext, useState, type ReactNode } from "react";

/** Mock authentication only — no backend session, no real credential check.
 * Exists so the dashboard has a clinical-portal login flow to demo, not to
 * gate access to anything real. Never wire this to real PHI/credentials
 * without a genuine auth backend behind it. */
const STORAGE_KEY = "cellscan-auth";

export interface ClinicianSession {
  name: string;
  role: "Radiologist" | "Oncologist" | "Pathologist";
  hospitalNetwork: string;
  mfaVerified: boolean;
}

interface AuthContextValue {
  session: ClinicianSession | null;
  login: (session: ClinicianSession) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function readStoredSession(): ClinicianSession | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ClinicianSession) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<ClinicianSession | null>(readStoredSession);

  function login(next: ClinicianSession) {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setSession(next);
  }

  function logout() {
    sessionStorage.removeItem(STORAGE_KEY);
    setSession(null);
  }

  return <AuthContext.Provider value={{ session, login, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
