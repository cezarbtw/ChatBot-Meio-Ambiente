import type {
  AnalyticsResponse,
  ClassifyRequest,
  ClassifyResponse,
  CollectResponse,
  HealthResponse,
  NewsItem,
  NewsLabel,
  NewsListResponse,
} from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = `Erro HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (typeof data?.detail === "string") {
        detail = data.detail;
      } else if (Array.isArray(data?.detail) && data.detail[0]?.msg) {
        detail = String(data.detail[0].msg);
      }
    } catch {
      // ignora payload não JSON
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export interface NewsQueryFilters {
  label?: string;
  source?: string;
  source_type?: string;
  date_from?: string;
  date_to?: string;
  protocol?: string;
  q?: string;
  ambiguous_only?: boolean;
  limit?: number;
  offset?: number;
}

function toQueryString(filters: NewsQueryFilters): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "" && value !== false) {
      params.set(key, String(value));
    }
  }
  const str = params.toString();
  return str ? `?${str}` : "";
}

export const apiClient = {
  getHealth(): Promise<HealthResponse> {
    return requestJson<HealthResponse>("/api/health");
  },

  classify(payload: ClassifyRequest): Promise<ClassifyResponse> {
    return requestJson<ClassifyResponse>("/api/classify", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  collect(payload: {
    sources?: string[];
    limit_per_source?: number;
    inspect_items?: boolean;
  }): Promise<CollectResponse> {
    return requestJson<CollectResponse>("/api/collect", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  getNews(filters: NewsQueryFilters = {}): Promise<NewsListResponse> {
    return requestJson<NewsListResponse>(`/api/news${toQueryString(filters)}`);
  },

  updateLabel(
    itemId: number,
    payload: { label: NewsLabel; reviewer?: string; notes?: string }
  ): Promise<NewsItem> {
    return requestJson<NewsItem>(`/api/news/${itemId}/label`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  },

  getAnalytics(filters: {
    date_from?: string;
    date_to?: string;
    source?: string;
  } = {}): Promise<AnalyticsResponse> {
    const params = new URLSearchParams();
    if (filters.date_from) params.set("date_from", filters.date_from);
    if (filters.date_to) params.set("date_to", filters.date_to);
    if (filters.source) params.set("source", filters.source);
    const qs = params.toString();
    return requestJson<AnalyticsResponse>(`/api/analytics${qs ? `?${qs}` : ""}`);
  },

  getExportUrl(filters: NewsQueryFilters = {}): string {
    return `${API_BASE}/api/news/export${toQueryString(filters)}`;
  },
};
