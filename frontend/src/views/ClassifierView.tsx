import { FormEvent, useState } from "react";
import { AlertCircle, CheckCircle2, Link2, Sparkles, Type } from "lucide-react";

import { ApiError, apiClient } from "../api/client";
import { HighlightKeywords } from "../components/HighlightKeywords";
import { LabelBadge } from "../components/LabelBadge";
import type { ClassifyResponse } from "../types";

interface ClassifierViewProps {
  onSavedToDb?: () => void;
}

const SAMPLE_EXAMPLES = [
  {
    label: "Exemplo Boa (Amazônia)",
    title: "NO satélite que vai descobrir quanto a Amazônia 'pesa'",
    text: "Sistema de observação conta com um radar especial que vai medir a quantidade de gás armazenado pelas árvores nas florestas tropicais.",
  },
  {
    label: "Exemplo Ruim (Poluição)",
    title: "Alerta de poluição e incêndios em reservas naturais",
    text: "Índices críticos de qualidade do ar e queimadas elevam riscos ambientais.",
  },
  {
    label: "Exemplo Técnico (Temperatura)",
    title: "Nova frente fria deve derrubar temperaturas no Sudeste",
    text: "Previsão meteorológica aponta variação de temperatura e chuvas isoladas.",
  },
];

const BAR_COLORS: Record<string, string> = {
  Boa: "bg-emerald-600",
  Ruim: "bg-rose-600",
  Irrelevante: "bg-slate-500",
};

export function ClassifierView({ onSavedToDb }: ClassifierViewProps) {
  const [mode, setMode] = useState<"manual" | "url">("manual");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [saveToDb, setSaveToDb] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassifyResponse | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    if (mode === "manual" && !title.trim() && !text.trim()) {
      setError("Informe ao menos o título ou o texto da notícia.");
      return;
    }
    if (mode === "url" && !url.trim()) {
      setError("Informe uma URL http:// ou https:// válida.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await apiClient.classify(
        mode === "manual"
          ? { title: title.trim(), text: text.trim(), save_to_db: saveToDb }
          : { url: url.trim(), save_to_db: saveToDb }
      );
      setResult(response);
      if (saveToDb && response.created && onSavedToDb) {
        onSavedToDb();
      }
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Falha ao conectar com a API de classificação."
      );
    } finally {
      setSubmitting(false);
    }
  }

  const matchedKeywordList = result?.keywords.map((k) => k.keyword) ?? [];

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Formulário */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-ink">
              Classificador Interativo de Notícias
            </h2>
            <p className="text-xs text-slate-500">
              Teste manchetes manuais ou extraia conteúdo e telemetria diretamente de uma URL.
            </p>
          </div>
        </div>

        {/* Alternador de modo */}
        <div role="tablist" aria-label="Modo de entrada" className="mb-4 inline-flex rounded-lg bg-slate-100 p-1">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "manual"}
            onClick={() => setMode("manual")}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
              mode === "manual"
                ? "bg-white text-ink shadow-sm"
                : "text-slate-600 hover:text-ink"
            }`}
          >
            <Type className="h-3.5 w-3.5" />
            Título e Texto
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "url"}
            onClick={() => setMode("url")}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition ${
              mode === "url"
                ? "bg-white text-ink shadow-sm"
                : "text-slate-600 hover:text-ink"
            }`}
          >
            <Link2 className="h-3.5 w-3.5" />
            Analisar URL
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === "manual" ? (
            <>
              <div className="flex flex-wrap gap-2">
                {SAMPLE_EXAMPLES.map((sample) => (
                  <button
                    key={sample.label}
                    type="button"
                    onClick={() => {
                      setTitle(sample.title);
                      setText(sample.text);
                      setError(null);
                    }}
                    className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:border-emerald-300 hover:bg-emerald-50"
                  >
                    {sample.label}
                  </button>
                ))}
              </div>

              <div>
                <label
                  htmlFor="classify-title"
                  className="mb-1 block text-xs font-semibold text-slate-700"
                >
                  Título da notícia
                </label>
                <input
                  id="classify-title"
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Ex.: Nova usina solar abastece 50 mil residências"
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-ink focus:border-emerald-600 focus:outline-none"
                />
              </div>

              <div>
                <label
                  htmlFor="classify-text"
                  className="mb-1 block text-xs font-semibold text-slate-700"
                >
                  Texto / Resumo
                </label>
                <textarea
                  id="classify-text"
                  rows={5}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Cole o resumo ou corpo da reportagem..."
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-ink focus:border-emerald-600 focus:outline-none"
                />
              </div>
            </>
          ) : (
            <div>
              <label
                htmlFor="classify-url"
                className="mb-1 block text-xs font-semibold text-slate-700"
              >
                URL da matéria
              </label>
              <input
                id="classify-url"
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://g1.globo.com/meio-ambiente/..."
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-ink focus:border-emerald-600 focus:outline-none"
              />
              <p className="mt-1 text-[11px] text-slate-500">
                A API extrairá o título, texto e telemetria real DNS/TLS/HTTP da página.
              </p>
            </div>
          )}

          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={saveToDb}
              onChange={(e) => setSaveToDb(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 text-emerald-700 focus:ring-emerald-600"
            />
            Salvar item classificado no histórico (SQLite)
          </label>

          {error && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:opacity-60"
          >
            <Sparkles className="h-4 w-4" />
            {submitting ? "Classificando com BERT + MLP..." : "Classificar Agora"}
          </button>
        </form>
      </section>

      {/* Resultado */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 lg:col-span-6">
        <h2 className="text-base font-semibold text-ink">Resultado da Análise</h2>
        {!result ? (
          <div className="mt-4 flex h-64 items-center justify-center rounded-lg border border-dashed border-slate-200 p-6 text-center text-xs text-slate-500">
            Envie um título/texto ou URL ao lado para visualizar a classe prevista, confiança por categoria e palavras-chave ambientais destacadas.
          </div>
        ) : (
          <div className="mt-4 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 p-3">
              <div className="text-xs font-medium text-slate-600">Classificação Prevista</div>
              <LabelBadge
                label={result.label}
                confidence={result.confidence}
                isAmbiguous={result.is_ambiguous}
              />
            </div>

            {result.is_ambiguous && result.ambiguous_reason && (
              <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900">
                <p className="font-semibold">Atenção — Classificação Ambígua</p>
                <p className="mt-0.5">{result.ambiguous_reason}</p>
              </div>
            )}

            {result.saved_item_id !== null && (
              <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-900">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                <span>
                  {result.created
                    ? `Salvo no histórico (ID #${result.saved_item_id}).`
                    : `Item já existente no banco (ID #${result.saved_item_id}, deduplicado).`}
                </span>
              </div>
            )}

            {/* Barras de probabilidade */}
            <div className="space-y-2.5">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Probabilidades por Classe
              </h3>
              {Object.entries(result.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([clsName, prob]) => (
                  <div key={clsName}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="font-medium text-ink">{clsName}</span>
                      <span className="font-mono text-slate-600">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className={`h-full rounded-full transition-all ${
                          BAR_COLORS[clsName] ?? "bg-slate-500"
                        }`}
                        style={{ width: `${Math.min(100, Math.max(2, prob * 100))}%` }}
                      />
                    </div>
                  </div>
                ))}
            </div>

            {/* Palavras-chave */}
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Palavras-Chave Detectadas
              </h3>
              {result.keywords.length === 0 ? (
                <p className="text-xs text-slate-500">
                  Nenhuma palavra-chave monitorada encontrada no texto.
                </p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {result.keywords.map((kw) => (
                    <span
                      key={kw.keyword}
                      className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-900"
                    >
                      {kw.keyword} ({kw.occurrences}x)
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Texto analisado com destaque */}
            <div className="rounded-lg border border-slate-200 p-3 text-xs leading-relaxed">
              <p className="font-semibold text-ink">
                <HighlightKeywords
                  text={result.extracted_title}
                  keywords={matchedKeywordList}
                />
              </p>
              {result.extracted_text && (
                <p className="mt-1.5 text-slate-600">
                  <HighlightKeywords
                    text={result.extracted_text}
                    keywords={matchedKeywordList}
                  />
                </p>
              )}
              {result.protocol_summary && (
                <p className="mt-2 font-mono text-[11px] text-slate-500">
                  Protocolo: {result.protocol_summary}
                </p>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
