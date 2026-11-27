import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type {
  AnalyticsResponse,
  ChatResponse,
  ClassifyResponse,
  CollectResponse,
  HealthResponse,
  NewsItem,
  NewsListResponse,
} from "./types";

const mockHealth: HealthResponse = {
  status: "ok",
  model_loaded: true,
  model_name: "neuralmind/bert-large-portuguese-cased",
  total_stored_items: 2,
};

const mockAnalytics: AnalyticsResponse = {
  overview: {
    total_items: 2,
    ambiguous_items: 1,
    by_label: { Boa: 1, Irrelevante: 0, Ruim: 1 },
    by_source: { g1: 2 },
    recent_runs: [],
  },
  daily_series: [
    { date: "2026-11-26", Boa: 1, Ruim: 1, Irrelevante: 0, total: 2 },
  ],
  keyword_distribution: {
    keywords: ["incêndios", "desmatamento", "aquecimento", "temperatura", "poluição"],
    counts: {
      Boa: { incêndios: 0, desmatamento: 1, aquecimento: 0, temperatura: 0, poluição: 0 },
      Irrelevante: { incêndios: 0, desmatamento: 0, aquecimento: 0, temperatura: 0, poluição: 0 },
      Ruim: { incêndios: 1, desmatamento: 0, aquecimento: 0, temperatura: 0, poluição: 1 },
    },
    percentages: {
      Boa: { incêndios: 0, desmatamento: 100, aquecimento: 0, temperatura: 0, poluição: 0 },
      Irrelevante: { incêndios: 0, desmatamento: 0, aquecimento: 0, temperatura: 0, poluição: 0 },
      Ruim: { incêndios: 100, desmatamento: 0, aquecimento: 0, temperatura: 0, poluição: 100 },
    },
    totals_by_keyword: {
      incêndios: 1,
      desmatamento: 1,
      aquecimento: 0,
      temperatura: 0,
      poluição: 1,
    },
  },
  protocol_stats: {
    total_records: 2,
    errors_count: 0,
    schemes: { https: 2 },
    http_versions: { "HTTP/1.1": 2 },
    alpn_protocols: { h2: 2 },
    tls_versions: { "TLSv1.3": 2 },
    cipher_suites: { TLS_AES_256_GCM_SHA384: 2 },
    status_codes: { "200": 2 },
    avg_dns_ms: 1.5,
    avg_tls_ms: 32.0,
    avg_ttfb_ms: 140.0,
  },
  model_metrics: {
    model_config: {
      bert_model_name: "neuralmind/bert-large-portuguese-cased",
      pooling: "cls",
      classifier_type: "mlp",
      use_scaler: true,
    },
    holdout: {
      accuracy: 0.903,
      f1_macro: 0.903,
      log_loss: 0.33,
      polarization_errors: {
        boa_classified_as_ruim_count: 2,
        boa_classified_as_ruim_rate: 0.057,
      },
    },
    cross_validation: { n_splits: 5, accuracy_mean: 0.869, f1_macro_mean: 0.869 },
  },
  benchmark_summary: {
    total_configurations: 26,
    best_configuration: { name: "bert-large-portuguese-cased[mean,scaler]+svc" },
  },
  sources_available: [{ name: "g1", source_type: "news" }],
};

const initialNewsItem: NewsItem = {
  id: 10,
  source: "g1",
  source_type: "news",
  title: "Temperatura varia no litoral",
  text: "Previsão indica aumento leve de temperatura.",
  url: "https://g1.globo.com/10",
  canonical_url: "https://g1.globo.com/10",
  author: null,
  published_at: "2026-11-26T10:00:00Z",
  published_date: "2026-11-26",
  collected_at: "2026-11-26T10:05:00Z",
  predicted_label: "Ruim",
  effective_label: "Ruim",
  confidence: 0.56,
  probabilities: { Boa: 0.2, Irrelevante: 0.24, Ruim: 0.56 },
  is_ambiguous: true,
  ambiguous_reason: "confiança abaixo do limiar",
  protocol_summary: "HTTP/1.1 | ALPN h2 | TLSv1.3",
  keywords: [{ keyword: "temperatura", occurrences: 2 }],
  reviews_count: 0,
  telemetry: {
    host: "g1.globo.com",
    http_version: "HTTP/1.1",
    alpn_protocol: "h2",
    tls_version: "TLSv1.3",
    cipher_suite: "TLS_AES_256_GCM_SHA384",
    status_code: 200,
    dns_ms: 1.1,
    tls_ms: 25.0,
    ttfb_ms: 110.0,
    resolved_ips: ["186.192.83.7"],
  },
};

function okJson(payload: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => payload,
  } as Response;
}

describe("EcoMonitor IA App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renderiza dashboard, classifica notícia manual e corrige rótulo no feed", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/health")) return okJson(mockHealth);
      if (url.includes("/api/analytics")) return okJson(mockAnalytics);
      if (url.includes("/api/classify") && init?.method === "POST") {
        const res: ClassifyResponse = {
          label: "Boa",
          confidence: 0.94,
          probabilities: { Boa: 0.94, Irrelevante: 0.04, Ruim: 0.02 },
          is_ambiguous: false,
          ambiguous_reason: null,
          matched_ambiguous_terms: [],
          keywords: [{ keyword: "desmatamento", occurrences: 1 }],
          extracted_title: "Reflorestamento reduz desmatamento",
          extracted_text: "Nova área protegida no Pará.",
          url: null,
          protocol_summary: null,
          network_telemetry: null,
          saved_item_id: null,
          created: null,
        };
        return okJson(res);
      }
      if (url.includes("/api/collect") && init?.method === "POST") {
        const res: CollectResponse = {
          run_id: 3,
          collected: 1,
          inserted: 1,
          deduplicated: 0,
          errors_count: 0,
          duration_ms: 420,
          sources: [
            { source: "g1", source_type: "news", success: true, items_count: 1, duration_ms: 420, error: null },
          ],
        };
        return okJson(res);
      }
      if (url.includes("/api/news/10/label") && init?.method === "PATCH") {
        return okJson({
          ...initialNewsItem,
          effective_label: "Irrelevante",
          reviews_count: 1,
        });
      }
      if (url.includes("/api/chat") && init?.method === "POST") {
        const res: ChatResponse = {
          answer: "Encontrei 1 registro sobre desmatamento.",
          intent: "news_search",
          mode: "local",
          citations: [
            {
              id: 10,
              title: "Temperatura varia no litoral",
              url: "https://g1.globo.com/10",
              source: "g1",
              effective_label: "Irrelevante",
              confidence: 0.56,
              published_date: "2026-11-26",
              protocol_summary: "HTTP/1.1 | ALPN h2 | TLSv1.3",
            },
          ],
          classification: null,
          suggested_followups: ["Resumo dos protocolos de rede identificados"],
        };
        return okJson(res);
      }
      if (url.includes("/api/news")) {
        const listRes: NewsListResponse = {
          items: [initialNewsItem],
          total: 1,
          limit: 12,
          offset: 0,
        };
        return okJson(listRes);
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    // 1. Dashboard carrega KPIs e status online
    await waitFor(() => {
      expect(screen.getByText("API + Modelo Online")).toBeInTheDocument();
    });
    expect(screen.getByText("90.3%")).toBeInTheDocument();

    // 2. Navega para Classificador IA e envia formulário
    fireEvent.click(screen.getByRole("button", { name: /Classificador IA/i }));
    fireEvent.change(screen.getByLabelText("Título da notícia"), {
      target: { value: "Reflorestamento reduz desmatamento" },
    });
    fireEvent.change(screen.getByLabelText("Texto / Resumo"), {
      target: { value: "Nova área protegida no Pará." },
    });
    fireEvent.click(screen.getByRole("button", { name: /Classificar Agora/i }));

    await waitFor(() => {
      expect(screen.getAllByText("94.0%")).toHaveLength(2);
    });
    expect(screen.getByText("desmatamento (1x)")).toBeInTheDocument();

    // 3. Navega para Feed & Curadoria, aciona coleta e corrige rótulo
    fireEvent.click(screen.getByRole("button", { name: /Feed & Curadoria/i }));
    await waitFor(() => {
      expect(screen.getByText("Temperatura varia no litoral")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /Coletar Agora/i }));
    await waitFor(() => {
      expect(screen.getByText(/Coleta #3 concluída/i)).toBeInTheDocument();
    });

    const labelSelect = screen.getByLabelText("Corrigir rótulo de Temperatura varia no litoral");
    fireEvent.change(labelSelect, { target: { value: "Irrelevante" } });
    await waitFor(() => {
      expect(screen.getByText(/Revisado \(1x · IA: Ruim\)/i)).toBeInTheDocument();
    });

    // 4. Navega para ChatBot IA e faz consulta ao acervo
    fireEvent.click(screen.getByRole("button", { name: /ChatBot IA/i }));
    fireEvent.change(screen.getByLabelText("Mensagem para o ChatBot Ambiental"), {
      target: { value: "Mostre notícias sobre desmatamento" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Enviar/i }));

    await waitFor(() => {
      expect(screen.getByText("Encontrei 1 registro sobre desmatamento.")).toBeInTheDocument();
    });
    expect(screen.getByText("Fontes Citadas (1)")).toBeInTheDocument();
  });
});
