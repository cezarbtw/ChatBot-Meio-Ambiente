import { useCallback, useEffect, useState } from "react";
import {
  Download,
  ExternalLink,
  Play,
  Search,
  Shield,
} from "lucide-react";

import { ApiError, apiClient, type NewsQueryFilters } from "../api/client";
import { LabelBadge } from "../components/LabelBadge";
import type { CollectResponse, NewsItem, NewsLabel } from "../types";

interface FeedViewProps {
  availableSources: Array<{ name: string; source_type: string }>;
  onDataMutated?: () => void;
}

const PAGE_SIZE = 12;
const LABELS: NewsLabel[] = ["Boa", "Ruim", "Irrelevante"];

export function FeedView({ availableSources, onDataMutated }: FeedViewProps) {
  const [items, setItems] = useState<NewsItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState("");
  const [sourceTypeFilter, setSourceTypeFilter] = useState("");
  const [protocolFilter, setProtocolFilter] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [ambiguousOnly, setAmbiguousOnly] = useState(false);

  const [collectSources, setCollectSources] = useState<string[]>([]);
  const [collecting, setCollecting] = useState(false);
  const [collectFeedback, setCollectFeedback] = useState<CollectResponse | null>(null);
  const [savingId, setSavingId] = useState<number | null>(null);

  const buildFilters = useCallback(
    (targetOffset: number): NewsQueryFilters => ({
      q: query.trim() || undefined,
      label: labelFilter || undefined,
      source: sourceFilter || undefined,
      source_type: sourceTypeFilter || undefined,
      protocol: protocolFilter.trim() || undefined,
      date_from: dateFrom || undefined,
      date_to: dateTo || undefined,
      ambiguous_only: ambiguousOnly || undefined,
      limit: PAGE_SIZE,
      offset: targetOffset,
    }),
    [
      query,
      labelFilter,
      sourceFilter,
      sourceTypeFilter,
      protocolFilter,
      dateFrom,
      dateTo,
      ambiguousOnly,
    ]
  );

  const loadFeed = useCallback(
    async (targetOffset = offset) => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiClient.getNews(buildFilters(targetOffset));
        setItems(res.items);
        setTotal(res.total);
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Erro ao carregar notícias do feed."
        );
      } finally {
        setLoading(false);
      }
    },
    [buildFilters, offset]
  );

  useEffect(() => {
    void loadFeed(offset);
  }, [loadFeed, offset]);

  async function handleCollectNow() {
    setCollecting(true);
    setError(null);
    setCollectFeedback(null);
    try {
      const res = await apiClient.collect({
        sources: collectSources.length > 0 ? collectSources : undefined,
        limit_per_source: 5,
      });
      setCollectFeedback(res);
      setOffset(0);
      await loadFeed(0);
      onDataMutated?.();
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Falha ao acionar coleta sob demanda."
      );
    } finally {
      setCollecting(false);
    }
  }

  async function handleCorrectLabel(itemId: number, newLabel: NewsLabel) {
    setSavingId(itemId);
    setError(null);
    try {
      const updated = await apiClient.updateLabel(itemId, {
        label: newLabel,
        reviewer: "frontend-user",
      });
      setItems((prev) => prev.map((it) => (it.id === itemId ? updated : it)));
      onDataMutated?.();
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Erro ao salvar correção do rótulo."
      );
    } finally {
      setSavingId(null);
    }
  }

  function toggleCollectSource(name: string) {
    setCollectSources((prev) =>
      prev.includes(name) ? prev.filter((s) => s !== name) : [...prev, name]
    );
  }

  const exportUrl = apiClient.getExportUrl(buildFilters(0));
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-5">
      {/* Coleta imediata e Exportação */}
      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-ink">
              Feed Multi-Fonte & Curadoria de Rótulos
            </h2>
            <p className="text-xs text-slate-500">
              Acione coletas sob demanda, inspecione protocolos e corrija classificações ambíguas.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <a
              href={exportUrl}
              download="noticias_monitoradas.csv"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-ink hover:bg-slate-50"
            >
              <Download className="h-3.5 w-3.5" />
              Exportar CSV
            </a>
            <button
              type="button"
              onClick={handleCollectNow}
              disabled={collecting}
              className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-2 text-xs font-semibold text-white hover:bg-emerald-800 disabled:opacity-60"
            >
              <Play className="h-3.5 w-3.5" />
              {collecting ? "Coletando..." : "Coletar Agora"}
            </button>
          </div>
        </div>

        {availableSources.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-slate-500">Fontes da coleta (vazio = padrão):</span>
            {availableSources
              .filter((s) => s.name !== "reddit")
              .map((src) => {
                const selected = collectSources.includes(src.name);
                return (
                  <button
                    key={src.name}
                    type="button"
                    onClick={() => toggleCollectSource(src.name)}
                    className={`rounded-full border px-2.5 py-0.5 font-medium transition ${
                      selected
                        ? "border-emerald-600 bg-emerald-50 text-emerald-900"
                        : "border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100"
                    }`}
                  >
                    {src.name}
                  </button>
                );
              })}
          </div>
        )}

        {collectFeedback && (
          <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-950">
            <p className="font-semibold">
              Coleta #{collectFeedback.run_id} concluída em{" "}
              {(collectFeedback.duration_ms / 1000).toFixed(1)}s: {collectFeedback.inserted}{" "}
              novo(s), {collectFeedback.deduplicated} deduplicado(s) ({collectFeedback.errors_count}{" "}
              erro(s)).
            </p>
            <div className="mt-1 flex flex-wrap gap-2 text-[11px]">
              {collectFeedback.sources.map((s) => (
                <span key={s.source} className="rounded bg-white/80 px-2 py-0.5">
                  {s.source}: {s.success ? `${s.items_count} itens` : `Erro (${s.error})`}
                </span>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Filtros */}
      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="relative lg:col-span-2">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <input
              type="search"
              aria-label="Buscar por título ou texto"
              value={query}
              onChange={(e) => {
                setOffset(0);
                setQuery(e.target.value);
              }}
              placeholder="Buscar no título ou texto..."
              className="w-full rounded-lg border border-slate-300 py-2 pl-9 pr-3 text-xs text-ink focus:border-emerald-600 focus:outline-none"
            />
          </div>

          <select
            aria-label="Filtrar por classificação"
            value={labelFilter}
            onChange={(e) => {
              setOffset(0);
              setLabelFilter(e.target.value);
            }}
            className="rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-xs text-ink"
          >
            <option value="">Todas as classes</option>
            {LABELS.map((lbl) => (
              <option key={lbl} value={lbl}>
                {lbl}
              </option>
            ))}
          </select>

          <select
            aria-label="Filtrar por fonte"
            value={sourceFilter}
            onChange={(e) => {
              setOffset(0);
              setSourceFilter(e.target.value);
            }}
            className="rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-xs text-ink"
          >
            <option value="">Todas as fontes</option>
            {availableSources.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>

          <select
            aria-label="Filtrar por tipo de fonte"
            value={sourceTypeFilter}
            onChange={(e) => {
              setOffset(0);
              setSourceTypeFilter(e.target.value);
            }}
            className="rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-xs text-ink"
          >
            <option value="">Portal + Social</option>
            <option value="news">Portais (news)</option>
            <option value="social">Redes Sociais (social)</option>
            <option value="manual">Manual</option>
          </select>

          <input
            type="text"
            aria-label="Filtrar por protocolo"
            value={protocolFilter}
            onChange={(e) => {
              setOffset(0);
              setProtocolFilter(e.target.value);
            }}
            placeholder="Protocolo (ex.: TLSv1.3)"
            className="rounded-lg border border-slate-300 px-2.5 py-2 text-xs text-ink"
          />
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-600">
          <label className="flex items-center gap-1.5">
            <span>De:</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => {
                setOffset(0);
                setDateFrom(e.target.value);
              }}
              className="rounded border border-slate-300 px-2 py-1 text-xs"
            />
          </label>
          <label className="flex items-center gap-1.5">
            <span>Até:</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => {
                setOffset(0);
                setDateTo(e.target.value);
              }}
              className="rounded border border-slate-300 px-2 py-1 text-xs"
            />
          </label>
          <label className="inline-flex items-center gap-1.5 font-medium">
            <input
              type="checkbox"
              checked={ambiguousOnly}
              onChange={(e) => {
                setOffset(0);
                setAmbiguousOnly(e.target.checked);
              }}
              className="h-4 w-4 rounded border-slate-300 text-amber-600"
            />
            Somente casos ambíguos
          </label>
          <span className="ml-auto font-mono text-slate-500">
            {total} registro(s) encontrado(s)
          </span>
        </div>
      </section>

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-800"
        >
          {error}
        </div>
      )}

      {/* Tabela responsiva */}
      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
            <thead className="bg-slate-50 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3">Data / Fonte</th>
                <th className="px-4 py-3">Conteúdo</th>
                <th className="px-4 py-3">Protocolo / Rede</th>
                <th className="px-4 py-3">Classificação</th>
                <th className="px-4 py-3">Curadoria</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading && items.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                    Carregando notícias...
                  </td>
                </tr>
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                    Nenhum item encontrado para os filtros selecionados.
                  </td>
                </tr>
              ) : (
                items.map((item) => (
                  <tr key={item.id} className="align-top hover:bg-slate-50/70">
                    <td className="whitespace-nowrap px-4 py-3">
                      <div className="font-mono font-medium text-ink">
                        {item.published_date ?? "Sem data"}
                      </div>
                      <div className="mt-1 inline-flex items-center gap-1 rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                        {item.source} · {item.source_type}
                      </div>
                    </td>
                    <td className="max-w-md px-4 py-3">
                      <div className="font-semibold text-ink">{item.title}</div>
                      <p className="mt-1 line-clamp-2 text-slate-600">{item.text}</p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-2">
                        {item.url.startsWith("http") && (
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 hover:underline"
                          >
                            Abrir fonte <ExternalLink className="h-3 w-3" />
                          </a>
                        )}
                        {item.keywords.map((k) => (
                          <span
                            key={k.keyword}
                            className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-800"
                          >
                            {k.keyword}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-[11px] text-slate-600">
                      <div className="inline-flex items-center gap-1 text-ink">
                        <Shield className="h-3.5 w-3.5 text-emerald-700" />
                        <span>{item.protocol_summary}</span>
                      </div>
                      {item.telemetry && (
                        <div className="mt-1 text-[10px] text-slate-500">
                          {item.telemetry.resolved_ips[0] ?? item.telemetry.host ?? ""}
                          {item.telemetry.ttfb_ms !== null
                            ? ` · ${item.telemetry.ttfb_ms}ms`
                            : ""}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <LabelBadge
                        label={item.effective_label}
                        confidence={item.confidence}
                        isAmbiguous={item.is_ambiguous}
                      />
                      {item.reviews_count > 0 && (
                        <div className="mt-1 text-[10px] text-slate-500">
                          Revisado ({item.reviews_count}x · IA: {item.predicted_label})
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <select
                        aria-label={`Corrigir rótulo de ${item.title}`}
                        value={item.effective_label}
                        disabled={savingId === item.id}
                        onChange={(e) =>
                          void handleCorrectLabel(item.id, e.target.value as NewsLabel)
                        }
                        className="rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-ink focus:border-emerald-600 focus:outline-none disabled:opacity-50"
                      >
                        {LABELS.map((lbl) => (
                          <option key={lbl} value={lbl}>
                            {lbl}
                          </option>
                        ))}
                      </select>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Paginação */}
        <div className="flex items-center justify-between border-t border-slate-200 px-4 py-3 text-xs text-slate-600">
          <span>
            Página {currentPage} de {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={offset === 0 || loading}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              className="rounded border border-slate-300 px-3 py-1 font-medium hover:bg-slate-50 disabled:opacity-40"
            >
              Anterior
            </button>
            <button
              type="button"
              disabled={offset + PAGE_SIZE >= total || loading}
              onClick={() => setOffset(offset + PAGE_SIZE)}
              className="rounded border border-slate-300 px-3 py-1 font-medium hover:bg-slate-50 disabled:opacity-40"
            >
              Próxima
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
