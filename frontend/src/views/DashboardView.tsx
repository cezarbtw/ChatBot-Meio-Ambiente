import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Globe,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";

import type { AnalyticsResponse } from "../types";

interface DashboardViewProps {
  analytics: AnalyticsResponse | null;
  loading: boolean;
  error: string | null;
  selectedSource: string;
  onSourceChange: (source: string) => void;
  onRefresh: () => void;
}

const KEYWORD_COLORS = ["#0284c7", "#15803d", "#d97706", "#dc2626", "#7c3aed"];

function formatDateLabel(dateStr: string): string {
  const parts = dateStr.split("-");
  if (parts.length !== 3) return dateStr;
  return `${parts[2]}/${parts[1]}`;
}

export function DashboardView({
  analytics,
  loading,
  error,
  selectedSource,
  onSourceChange,
  onRefresh,
}: DashboardViewProps) {
  const keywordChartData = useMemo(() => {
    if (!analytics) return [];
    const { percentages, keywords } = analytics.keyword_distribution;
    return (["Boa", "Irrelevante", "Ruim"] as const).map((label) => {
      const row: Record<string, string | number> = { label };
      for (const kw of keywords) {
        row[kw] = percentages[label]?.[kw] ?? 0;
      }
      return row;
    });
  }, [analytics]);

  const dailyData = useMemo(() => {
    if (!analytics) return [];
    return analytics.daily_series.map((pt) => ({
      ...pt,
      dateLabel: formatDateLabel(pt.date),
    }));
  }, [analytics]);

  if (loading && !analytics) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-600">
        Carregando métricas do monitor ambiental...
      </div>
    );
  }

  if (error && !analytics) {
    return (
      <div
        role="alert"
        className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800"
      >
        <p className="font-semibold">Não foi possível carregar o dashboard.</p>
        <p className="mt-1">{error}</p>
        <button
          type="button"
          onClick={onRefresh}
          className="mt-3 rounded-lg bg-rose-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-800"
        >
          Tentar novamente
        </button>
      </div>
    );
  }

  if (!analytics) return null;

  const { overview, protocol_stats, model_metrics, benchmark_summary } = analytics;
  const byLabel = overview.by_label;
  const accuracyPct = model_metrics?.holdout?.accuracy
    ? `${(model_metrics.holdout.accuracy * 100).toFixed(1)}%`
    : "—";
  const cvF1Pct = model_metrics?.cross_validation?.f1_macro_mean
    ? `${(model_metrics.cross_validation.f1_macro_mean * 100).toFixed(1)}%`
    : "—";

  return (
    <div className="space-y-6">
      {/* Barra de ações / filtro */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
        <div>
          <h2 className="text-base font-semibold text-ink">
            Visão Analítica do Monitoramento
          </h2>
          <p className="text-xs text-slate-500">
            Histórico persistido no SQLite, incidência de palavras-chave e telemetria TLS/HTTP.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="dashboard-source-filter" className="text-xs font-medium text-slate-600">
            Filtrar série por fonte:
          </label>
          <select
            id="dashboard-source-filter"
            value={selectedSource}
            onChange={(e) => onSourceChange(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs text-ink focus:border-emerald-600 focus:outline-none"
          >
            <option value="">Todas as fontes</option>
            {analytics.sources_available.map((src) => (
              <option key={src.name} value={src.name}>
                {src.name} ({src.source_type})
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-ink hover:bg-slate-50 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Atualizar
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Itens Monitorados</span>
            <Activity className="h-4 w-4 text-emerald-700" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-ink">
            {overview.total_items}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Boa: <strong className="text-emerald-700">{byLabel.Boa ?? 0}</strong> · Ruim:{" "}
            <strong className="text-rose-700">{byLabel.Ruim ?? 0}</strong> · Irrel.:{" "}
            <strong>{byLabel.Irrelevante ?? 0}</strong>
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Casos Ambíguos</span>
            <AlertTriangle className="h-4 w-4 text-amber-600" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-amber-700">
            {overview.ambiguous_items}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Sinalizados para revisão manual na curadoria
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Desempenho do Modelo</span>
            <Cpu className="h-4 w-4 text-emerald-700" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-ink">{accuracyPct}</p>
          <p className="mt-1 text-xs text-slate-500">
            Holdout · CV F1-Macro: <span className="font-mono font-semibold">{cvF1Pct}</span>
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex items-center justify-between text-xs font-medium text-slate-500">
            <span>Telemetria TLS/Rede</span>
            <ShieldCheck className="h-4 w-4 text-emerald-700" />
          </div>
          <p className="mt-2 font-mono text-2xl font-bold text-ink">
            {protocol_stats.total_records}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            TLS médio:{" "}
            <span className="font-mono">
              {protocol_stats.avg_tls_ms !== null ? `${protocol_stats.avg_tls_ms} ms` : "—"}
            </span>{" "}
            · TTFB:{" "}
            <span className="font-mono">
              {protocol_stats.avg_ttfb_ms !== null ? `${protocol_stats.avg_ttfb_ms} ms` : "—"}
            </span>
          </p>
        </div>
      </div>

      {/* Gráficos principais (equivalentes interativos das Imagens 2 e 3 do PDF) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-ink">
            Ocorrência Diária por Classificação
          </h3>
          <p className="mb-4 text-xs text-slate-500">
            Evolução temporal das notícias/publicações coletadas por data de publicação.
          </p>
          {dailyData.length === 0 ? (
            <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-slate-200 text-xs text-slate-500">
              Sem registros diários ainda. Execute uma coleta na aba Feed & Curadoria.
            </div>
          ) : (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dailyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="dateLabel" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Boa" fill="#15803d" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Ruim" fill="#dc2626" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Irrelevante" fill="#64748b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-ink">
            Distribuição Percentual das Palavras-Chave (%)
          </h3>
          <p className="mb-4 text-xs text-slate-500">
            Proporção de ocorrências de termos ambientais entre classes (Boa, Irrelevante, Ruim).
          </p>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={keywordChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {analytics.keyword_distribution.keywords.map((kw, idx) => (
                  <Bar
                    key={kw}
                    dataKey={kw}
                    fill={KEYWORD_COLORS[idx % KEYWORD_COLORS.length]}
                    radius={[3, 3, 0, 0]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Painel de Protocolos de Rede & Informações do Modelo */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2">
            <Globe className="h-4 w-4 text-emerald-700" />
            <h3 className="text-sm font-semibold text-ink">
              Protocolos de Rede Identificados no Tráfego
            </h3>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
            <div className="rounded-lg bg-slate-50 p-3">
              <p className="text-slate-500">Versão HTTP</p>
              <div className="mt-1 space-y-0.5 font-mono font-semibold text-ink">
                {Object.entries(protocol_stats.http_versions).map(([k, v]) => (
                  <div key={k}>
                    {k}: {v}
                  </div>
                ))}
                {Object.keys(protocol_stats.http_versions).length === 0 && <div>—</div>}
              </div>
            </div>
            <div className="rounded-lg bg-slate-50 p-3">
              <p className="text-slate-500">ALPN Negociado</p>
              <div className="mt-1 space-y-0.5 font-mono font-semibold text-ink">
                {Object.entries(protocol_stats.alpn_protocols).map(([k, v]) => (
                  <div key={k}>
                    {k}: {v}
                  </div>
                ))}
                {Object.keys(protocol_stats.alpn_protocols).length === 0 && <div>—</div>}
              </div>
            </div>
            <div className="rounded-lg bg-slate-50 p-3">
              <p className="text-slate-500">Versão TLS</p>
              <div className="mt-1 space-y-0.5 font-mono font-semibold text-ink">
                {Object.entries(protocol_stats.tls_versions).map(([k, v]) => (
                  <div key={k}>
                    {k}: {v}
                  </div>
                ))}
                {Object.keys(protocol_stats.tls_versions).length === 0 && <div>—</div>}
              </div>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-600">
            <span>
              DNS médio:{" "}
              <strong className="font-mono">
                {protocol_stats.avg_dns_ms !== null ? `${protocol_stats.avg_dns_ms} ms` : "—"}
              </strong>
            </span>
            <span>
              Erros DNS/TLS/HTTP:{" "}
              <strong className="font-mono">{protocol_stats.errors_count}</strong>
            </span>
            <span>
              Fontes ativas:{" "}
              <strong className="font-mono">{Object.keys(overview.by_source).length}</strong>
            </span>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center gap-2">
            <Cpu className="h-4 w-4 text-emerald-700" />
            <h3 className="text-sm font-semibold text-ink">
              Configuração Ativa de IA & Benchmark
            </h3>
          </div>
          <dl className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg bg-slate-50 p-2.5">
              <dt className="text-slate-500">Backbone Transformer</dt>
              <dd className="mt-0.5 truncate font-mono font-semibold text-ink">
                {model_metrics?.model_config?.bert_model_name ?? "—"}
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 p-2.5">
              <dt className="text-slate-500">Pooling / Pipeline</dt>
              <dd className="mt-0.5 font-mono font-semibold text-ink">
                {model_metrics?.model_config?.pooling ?? "—"} ·{" "}
                {model_metrics?.model_config?.use_scaler ? "StandardScaler" : "raw"} ·{" "}
                {model_metrics?.model_config?.classifier_type ?? "mlp"}
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 p-2.5">
              <dt className="text-slate-500">Viés Boa -&gt; Ruim</dt>
              <dd className="mt-0.5 font-mono font-semibold text-ink">
                {model_metrics?.holdout?.polarization_errors
                  ? `${model_metrics.holdout.polarization_errors.boa_classified_as_ruim_count} (${(
                      model_metrics.holdout.polarization_errors.boa_classified_as_ruim_rate * 100
                    ).toFixed(1)}%)`
                  : "—"}
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 p-2.5">
              <dt className="text-slate-500">Melhor no Benchmark</dt>
              <dd className="mt-0.5 truncate font-mono font-semibold text-ink">
                {benchmark_summary?.best_configuration?.name ?? "—"}
              </dd>
            </div>
          </dl>
        </section>
      </div>
    </div>
  );
}
