import { useRef, useState, type MouseEvent } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ShieldCheck, Stethoscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAuth, type ClinicianSession } from "@/contexts/auth-context";

const HOSPITAL_NETWORKS = ["St. Mary General — Radiology", "Northshore Oncology Center", "Research Sandbox (this build)"];
const ROLES: ClinicianSession["role"][] = ["Radiologist", "Oncologist", "Pathologist"];

/** Two slow-drifting gradient blurs behind the form — purely decorative,
 * kept to CSS/framer-motion transforms (no canvas/particle library) so it
 * costs nothing on low-end devices and doesn't fight prefers-reduced-motion
 * users too hard (it's a slow ambient drift, not a flashy loop). */
function AmbientBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <motion.div
        className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-primary/25 blur-3xl"
        animate={{ x: [0, 40, 0], y: [0, 30, 0] }}
        transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute -bottom-32 -right-24 h-[28rem] w-[28rem] rounded-full bg-accent/20 blur-3xl"
        animate={{ x: [0, -30, 0], y: [0, -40, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

/** Subtle 3D tilt on the login card, following the cursor — a common
 * "React Bits"-style micro-interaction, done here as a plain transform on
 * mousemove rather than pulling in a tilt library for one effect. */
function TiltCard({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState("perspective(900px) rotateX(0deg) rotateY(0deg)");

  function handleMouseMove(e: MouseEvent<HTMLDivElement>) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    setTransform(`perspective(900px) rotateX(${(-py * 6).toFixed(2)}deg) rotateY(${(px * 6).toFixed(2)}deg)`);
  }

  function handleMouseLeave() {
    setTransform("perspective(900px) rotateX(0deg) rotateY(0deg)");
  }

  return (
    <div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ transform, transition: "transform 0.15s ease-out" }}
    >
      {children}
    </div>
  );
}

export function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [name, setName] = useState("");
  const [role, setRole] = useState<ClinicianSession["role"]>("Radiologist");
  const [network, setNetwork] = useState(HOSPITAL_NETWORKS[2]);
  const [mfaCode, setMfaCode] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) {
      setError("Enter a clinician name to continue.");
      return;
    }
    if (mfaCode.trim().length !== 6) {
      setError("Enter the 6-digit verification code.");
      return;
    }
    setError(null);
    setSubmitting(true);
    // deliberate short delay so the MFA step reads as a real check rather
    // than an instant no-op — this is mock auth, see auth-context.tsx
    setTimeout(() => {
      login({ name: name.trim(), role, hospitalNetwork: network, mfaVerified: true });
      navigate("/");
    }, 450);
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background px-4 py-8">
      <AmbientBackground />

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative w-full max-w-md"
      >
        <div className="mb-8 flex flex-col items-center gap-3 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-primary text-lg font-bold text-white shadow-lg shadow-primary/20">
            CS
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">CellScan Decision Support Platform</h1>
            <p className="mt-1 text-xs text-muted-foreground">Clinical sign-in — research build</p>
          </div>
        </div>

        <TiltCard>
          <form onSubmit={handleSubmit} className="space-y-4 rounded-xl border border-border bg-card p-6 shadow-xl shadow-black/10">
            <div className="flex items-center gap-2 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-xs font-medium text-primary">
              <ShieldCheck className="h-4 w-4 shrink-0" />
              Mock authentication — for demonstration only, not a real credential check
            </div>

            <label className="block">
              <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Clinician name
              </span>
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Dr. Jane Okafor" className="font-sans" />
            </label>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted-foreground">Role</span>
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value as ClinicianSession["role"])}
                  className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                >
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Hospital network
                </span>
                <select
                  value={network}
                  onChange={(e) => setNetwork(e.target.value)}
                  className="h-9 w-full rounded-md border border-border bg-background px-3 text-sm"
                >
                  {HOSPITAL_NETWORKS.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label className="block">
              <span className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                MFA verification code
              </span>
              <Input
                value={mfaCode}
                onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="6-digit code (any digits work in this demo)"
                className="font-mono-num tracking-widest"
              />
            </label>

            {error && <p className="text-xs text-risk-high-fg">{error}</p>}

            <Button type="submit" className="w-full" loading={submitting}>
              <Stethoscope className="h-4 w-4" /> Sign in to CellScan
            </Button>
          </form>
        </TiltCard>

        <p className="mt-4 text-center text-[11px] text-muted-foreground">
          Research/educational build. Not a real clinical system — see the About page's disclaimer after signing in.
        </p>
      </motion.div>
    </div>
  );
}
