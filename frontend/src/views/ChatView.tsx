import { FormEvent, useState } from "react";
import {
  Bot,
  ExternalLink,
  Send,
  Sparkles,
  User,
} from "lucide-react";

import { ApiError, apiClient } from "../api/client";
import { LabelBadge } from "../components/LabelBadge";
import type { ChatMessageItem } from "../types";

interface ChatViewProps {
  messages: ChatMessageItem[];
  onMessagesChange: (updater: (prev: ChatMessageItem[]) => ChatMessageItem[]) => void;
  onOpenFeedWithQuery?: (query: string) => void;
}

const QUICK_SUGGESTIONS = [
  "Qual o panorama ambiental desta semana?",
  "Quais são as piores notícias recentes?",
  "Mostre notícias sobre desmatamento",
  "Resumo dos protocolos de rede identificados",
  "Classifique: Pesquisadores criam filtro sustentável que reduz poluição em rios",
];

const INTENT_LABELS: Record<string, string> = {
  trend_summary: "Tendências",
  news_search: "Busca no Acervo",
  protocol_summary: "Protocolos de Rede",
  model_performance: "Desempenho IA",
  classify_text: "Classificação Direta",
};

export function ChatView({
  messages,
  onMessagesChange,
  onOpenFeedWithQuery,
}: ChatViewProps) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function sendMessage(textToSend: string) {
    const trimmed = textToSend.trim();
    if (!trimmed || sending) return;

    setError(null);
    const userMsg: ChatMessageItem = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
    };

    const historyPayload = messages
      .filter((m) => m.id !== "welcome")
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));

    onMessagesChange((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const reply = await apiClient.chat({
        message: trimmed,
        history: historyPayload,
      });

      const assistantMsg: ChatMessageItem = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: reply.answer,
        intent: reply.intent,
        mode: reply.mode,
        citations: reply.citations,
        classification: reply.classification,
        suggestedFollowups: reply.suggested_followups,
      };
      onMessagesChange((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Falha ao enviar mensagem para o ChatBot."
      );
    } finally {
      setSending(false);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void sendMessage(input);
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Painel lateral de atalhos */}
      <aside className="rounded-xl border border-slate-200 bg-white p-4 lg:col-span-4">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-emerald-700" />
          <h2 className="text-base font-semibold text-ink">
            Assistente EcoMonitor
          </h2>
        </div>
        <p className="mt-1.5 text-xs leading-relaxed text-slate-600">
          Consulte tendências do acervo SQLite, notícias por tema/classe, telemetria TLS/HTTP ou peça para classificar um texto diretamente no chat.
        </p>

        <div className="mt-4">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Perguntas Rápidas
          </h3>
          <div className="space-y-2">
            {QUICK_SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                disabled={sending}
                onClick={() => void sendMessage(suggestion)}
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-left text-xs font-medium text-ink transition hover:border-emerald-300 hover:bg-emerald-50 disabled:opacity-50"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </aside>

      {/* Janela de conversa */}
      <section className="flex h-[680px] flex-col rounded-xl border border-slate-200 bg-white lg:col-span-8">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-ink">
            <Sparkles className="h-4 w-4 text-emerald-700" />
            <span>Conversa da Sessão</span>
          </div>
          <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-[11px] font-medium text-slate-600">
            NLU Local + BERT/MLP + SQLite
          </span>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-emerald-800">
                  <Bot className="h-4 w-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-xl px-4 py-3 text-xs leading-relaxed ${
                  msg.role === "user"
                    ? "bg-emerald-700 text-white"
                    : "border border-slate-200 bg-slate-50 text-ink"
                }`}
              >
                {msg.role === "assistant" && (msg.intent || msg.mode) && (
                  <div className="mb-1.5 flex flex-wrap items-center gap-1.5 text-[10px]">
                    {msg.intent && (
                      <span className="rounded bg-white px-2 py-0.5 font-semibold text-emerald-800">
                        {INTENT_LABELS[msg.intent] ?? msg.intent}
                      </span>
                    )}
                    {msg.mode && (
                      <span className="rounded bg-slate-200/70 px-2 py-0.5 font-mono text-slate-600">
                        {msg.mode === "llm_rag" ? "RAG + LLM" : "Modo Local"}
                      </span>
                    )}
                  </div>
                )}

                <p className="whitespace-pre-line">{msg.content}</p>

                {/* Card de classificação inline */}
                {msg.classification && (
                  <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-slate-600">
                        Resultado PLN
                      </span>
                      <LabelBadge
                        label={msg.classification.label}
                        confidence={msg.classification.confidence}
                        isAmbiguous={msg.classification.is_ambiguous}
                      />
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-[11px] font-mono text-slate-600">
                      {Object.entries(msg.classification.probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([cls, prob]) => (
                          <span key={cls} className="rounded bg-slate-100 px-2 py-0.5">
                            {cls}: {(prob * 100).toFixed(1)}%
                          </span>
                        ))}
                    </div>
                  </div>
                )}

                {/* Notícias citadas com links clicáveis */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 space-y-1.5">
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                      Fontes Citadas ({msg.citations.length})
                    </p>
                    {msg.citations.map((cit) => (
                      <div
                        key={cit.id}
                        className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="truncate font-semibold text-ink">
                            {cit.title}
                          </div>
                          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                            <span>{cit.source}</span>
                            <span>·</span>
                            <span className="font-mono">
                              {cit.published_date ?? "Sem data"}
                            </span>
                            <span>·</span>
                            <LabelBadge
                              label={cit.effective_label}
                              confidence={cit.confidence}
                            />
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {onOpenFeedWithQuery && (
                            <button
                              type="button"
                              onClick={() => onOpenFeedWithQuery(cit.title)}
                              className="rounded border border-slate-200 px-2 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-50"
                            >
                              Ver no Feed
                            </button>
                          )}
                          {cit.url.startsWith("http") && (
                            <a
                              href={cit.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 rounded border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-semibold text-emerald-800 hover:bg-emerald-100"
                            >
                              Abrir <ExternalLink className="h-3 w-3" />
                            </a>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Sugestões de continuação */}
                {msg.suggestedFollowups && msg.suggestedFollowups.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {msg.suggestedFollowups.map((followup) => (
                      <button
                        key={followup}
                        type="button"
                        disabled={sending}
                        onClick={() => void sendMessage(followup)}
                        className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:border-emerald-300 hover:bg-emerald-50 disabled:opacity-50"
                      >
                        {followup}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {msg.role === "user" && (
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-200 text-slate-700">
                  <User className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}

          {sending && (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Bot className="h-4 w-4 animate-pulse text-emerald-700" />
              <span>Consultando acervo e modelo IA...</span>
            </div>
          )}
        </div>

        {error && (
          <div
            role="alert"
            className="mx-4 mb-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800"
          >
            {error}
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          className="flex items-center gap-2 border-t border-slate-200 p-3"
        >
          <input
            type="text"
            aria-label="Mensagem para o ChatBot Ambiental"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Pergunte sobre tendências, notícias ou digite 'Classifique: ...'"
            className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm text-ink focus:border-emerald-600 focus:outline-none"
          />
          <button
            type="submit"
            disabled={sending || !input.trim()}
            className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-800 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
            Enviar
          </button>
        </form>
      </section>
    </div>
  );
}
