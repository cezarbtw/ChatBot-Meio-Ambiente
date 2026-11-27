import { useCallback, useEffect, useState } from "react";
import { BarChart3, Leaf, Newspaper, Sparkles } from "lucide-react";

import { ApiError, apiClient } from "./api/client";
import type { AnalyticsResponse, HealthResponse } from "./types";
import { ClassifierView } from "./views/ClassifierView";
import { DashboardView } from "./views/DashboardView";
import { FeedView } from "./views/FeedView";

type TabId = "dashboard" | "classifier" | "feed";

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [analyticsSource, setAnalyticsSource] = useState("");
  const [loadingAnalytics, setLoadingAnalytics] = useState(true);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);

  const loadAnalyticsAndHealth = useCallback(async () => {
    setLoadingAnalytics(true);
    setAnalyticsError(null);
    try {
      const [healthData, analyticsData] = await Promise.all([
        apiClient.getHealth(),
        apiClient.getAnalytics({ source: analyticsSource || undefined }),
      ]);
      setHealth(healthData);
      setAnalytics(analyticsData);
    } catch (err) {
      setAnalyticsError(
        err instanceof ApiError
          ? err.message
          : "Não foi possível conectar ao backend FastAPI."
      );
    } finally {
      setLoadingAnalytics(false);
    }
  }, [analyticsSource]);

  useEffect(() => {
    void loadAnalyticsAndHealth();
  }, [loadAnalyticsAndHealth]);

  const tabs: Array<{ id: TabId; label: string; icon: typeof BarChart3 }> = [
    { id: "dashboard", label: "Dashboard Analítico", icon: BarChart3 },
    { id: "classifier", label: "Classificador IA", icon: Sparkles },
    { id: "feed", label: "Feed & Curadoria", icon: Newspaper },
  ];

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-700 text-white">
              <Leaf className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-ink sm:text-lg">
                EcoMonitor IA
              </h1>
              <p className="text-xs text-slate-500">
                Classificação de Notícias Ambientais · BERT + MLP · Telemetria TLS/HTTP
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-medium ${
                health?.model_loaded
                  ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                  : "border-amber-200 bg-amber-50 text-amber-800"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  health?.model_loaded ? "bg-emerald-600" : "bg-amber-500"
                }`}
              />
              {health?.model_loaded ? "API + Modelo Online" : "API Offline / Conectando"}
            </span>
          </div>
        </div>

        <nav
          aria-label="Navegação principal"
          className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-4 sm:px-6"
        >
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                aria-current={isActive ? "page" : undefined}
                className={`inline-flex shrink-0 items-center gap-2 border-b-2 px-3 py-2.5 text-xs font-semibold transition sm:text-sm ${
                  isActive
                    ? "border-emerald-700 text-emerald-800"
                    : "border-transparent text-slate-600 hover:text-ink"
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {activeTab === "dashboard" && (
          <DashboardView
            analytics={analytics}
            loading={loadingAnalytics}
            error={analyticsError}
            selectedSource={analyticsSource}
            onSourceChange={setAnalyticsSource}
            onRefresh={() => void loadAnalyticsAndHealth()}
          />
        )}

        {activeTab === "classifier" && (
          <ClassifierView onSavedToDb={() => void loadAnalyticsAndHealth()} />
        )}

        {activeTab === "feed" && (
          <FeedView
            availableSources={analytics?.sources_available ?? []}
            onDataMutated={() => void loadAnalyticsAndHealth()}
          />
        )}
      </main>
    </div>
  );
}

export default App;
