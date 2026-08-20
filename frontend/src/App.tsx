import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ThemeProvider } from "@/contexts/theme-context";
import { AuthProvider } from "@/contexts/auth-context";
import { Layout } from "@/components/layout/layout";
import { ProtectedRoute } from "@/components/layout/protected-route";
import { Login } from "@/pages/login";
import { Home } from "@/pages/home";
import { ClinicalData } from "@/pages/clinical-data";
import { UploadImage } from "@/pages/upload-image";
import { Mammography } from "@/pages/mammography";
import { ModelPerformance } from "@/pages/model-performance";
import { ClusterExplorer } from "@/pages/cluster-explorer";
import { About } from "@/pages/about";

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route element={<ProtectedRoute />}>
              <Route element={<Layout />}>
                <Route path="/" element={<Home />} />
                <Route path="/clinical-data" element={<ClinicalData />} />
                <Route path="/upload-image" element={<UploadImage />} />
                <Route path="/mammography" element={<Mammography />} />
                <Route path="/model-performance" element={<ModelPerformance />} />
                <Route path="/cluster-explorer" element={<ClusterExplorer />} />
                <Route path="/about" element={<About />} />
              </Route>
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
