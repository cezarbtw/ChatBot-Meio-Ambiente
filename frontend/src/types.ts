export type NewsLabel = "Boa" | "Ruim" | "Irrelevante";

export interface KeywordMatch {
  keyword: string;
  occurrences: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string | null;
  total_stored_items: number;
}

export interface ClassifyRequest {
  title?: string;
  text?: string;
  url?: string;
  save_to_db?: boolean;
}

export interface ClassifyResponse {
  label: NewsLabel;
  confidence: number;
  probabilities: Record<string, number>;
  is_ambiguous: boolean;
  ambiguous_reason: string | null;
  matched_ambiguous_terms: string[];
  keywords: KeywordMatch[];
  extracted_title: string;
  extracted_text: string;
  url: string | null;
  protocol_summary: string | null;
  network_telemetry: Record<string, unknown> | null;
  saved_item_id: number | null;
  created: boolean | null;
}

export interface NewsTelemetrySummary {
  host: string | null;
  http_version: string | null;
  alpn_protocol: string | null;
  tls_version: string | null;
  cipher_suite: string | null;
  status_code: number | null;
  dns_ms: number | null;
  tls_ms: number | null;
  ttfb_ms: number | null;
  resolved_ips: string[];
}

export interface NewsItem {
  id: number;
  source: string | null;
  source_type: string | null;
  title: string;
  text: string;
  url: string;
  canonical_url: string;
  author: string | null;
  published_at: string | null;
  published_date: string | null;
  collected_at: string;
  predicted_label: NewsLabel;
  effective_label: NewsLabel;
  confidence: number;
  probabilities: Record<string, number>;
  is_ambiguous: boolean;
  ambiguous_reason: string | null;
  protocol_summary: string;
  keywords: KeywordMatch[];
  reviews_count: number;
  telemetry: NewsTelemetrySummary | null;
}

export interface NewsListResponse {
  items: NewsItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface SourceReport {
  source: string;
  source_type: string;
  success: boolean;
  items_count: number;
  duration_ms: number;
  error: string | null;
}

export interface CollectResponse {
  run_id: number;
  collected: number;
  inserted: number;
  deduplicated: number;
  errors_count: number;
  duration_ms: number;
  sources: SourceReport[];
}

export interface DailySeriesPoint {
  date: string;
  Boa: number;
  Ruim: number;
  Irrelevante: number;
  total: number;
}

export interface AnalyticsResponse {
  overview: {
    total_items: number;
    ambiguous_items: number;
    by_label: Record<string, number>;
    by_source: Record<string, number>;
    recent_runs: Array<{
      id: number;
      started_at: string;
      sources_requested: string;
      items_collected: number;
      items_inserted: number;
      items_deduplicated: number;
      errors_count: number;
    }>;
  };
  daily_series: DailySeriesPoint[];
  keyword_distribution: {
    keywords: string[];
    counts: Record<string, Record<string, number>>;
    percentages: Record<string, Record<string, number>>;
    totals_by_keyword: Record<string, number>;
  };
  protocol_stats: {
    total_records: number;
    errors_count: number;
    schemes: Record<string, number>;
    http_versions: Record<string, number>;
    alpn_protocols: Record<string, number>;
    tls_versions: Record<string, number>;
    cipher_suites: Record<string, number>;
    status_codes: Record<string, number>;
    avg_dns_ms: number | null;
    avg_tls_ms: number | null;
    avg_ttfb_ms: number | null;
  };
  model_metrics: {
    model_config?: {
      bert_model_name?: string;
      pooling?: string;
      classifier_type?: string;
      use_scaler?: boolean;
    };
    holdout?: {
      accuracy: number;
      f1_macro: number;
      log_loss: number;
      polarization_errors?: {
        boa_classified_as_ruim_count: number;
        boa_classified_as_ruim_rate: number;
      };
    };
    cross_validation?: {
      n_splits: number;
      accuracy_mean: number;
      f1_macro_mean: number;
    };
  } | null;
  benchmark_summary: {
    total_configurations: number;
    best_configuration?: {
      name: string;
      cv_f1_macro_mean?: number;
      holdout_accuracy?: number;
    };
  } | null;
  sources_available: Array<{
    name: string;
    source_type: string;
  }>;
}
