"""Schemas Pydantic para validação de entrada/saída da API FastAPI."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator

from src.config import VALID_LABELS

ValidLabel = Literal["Boa", "Irrelevante", "Ruim"]


class KeywordMatchOut(BaseModel):
    keyword: str
    occurrences: int


class ClassifyRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Título da notícia ou publicação.")
    text: Optional[str] = Field(default=None, description="Corpo ou resumo do texto.")
    url: Optional[str] = Field(default=None, description="URL para extração automática de conteúdo.")
    save_to_db: bool = Field(
        default=False,
        description="Se verdadeiro, armazena o item classificado no banco SQLite.",
    )

    @root_validator
    def _validate_content_or_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        title = (values.get("title") or "").strip()
        text = (values.get("text") or "").strip()
        url = (values.get("url") or "").strip()
        if not url and not (title or text):
            raise ValueError("Informe ao menos 'title'/'text' ou uma 'url' válida.")
        values["title"] = title or None
        values["text"] = text or None
        values["url"] = url or None
        return values


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    is_ambiguous: bool
    ambiguous_reason: Optional[str] = None
    matched_ambiguous_terms: List[str] = Field(default_factory=list)
    keywords: List[KeywordMatchOut] = Field(default_factory=list)
    extracted_title: str
    extracted_text: str
    url: Optional[str] = None
    protocol_summary: Optional[str] = None
    network_telemetry: Optional[Dict[str, Any]] = None
    saved_item_id: Optional[int] = None
    created: Optional[bool] = None


class CollectRequest(BaseModel):
    sources: Optional[List[str]] = Field(
        default=None,
        description="Lista de fontes (ex.: ['g1', 'mastodon']). Omitir usa as fontes padrão.",
    )
    limit_per_source: int = Field(default=10, ge=1, le=50)
    inspect_items: bool = Field(
        default=False,
        description="Inspeciona individualmente a URL de cada item coletado.",
    )


class SourceReportOut(BaseModel):
    source: str
    source_type: str
    success: bool
    items_count: int
    duration_ms: float
    error: Optional[str] = None
    source_telemetry: Optional[Dict[str, Any]] = None


class CollectResponse(BaseModel):
    run_id: int
    collected: int
    inserted: int
    deduplicated: int
    errors_count: int
    duration_ms: float
    sources: List[SourceReportOut]


class NewsItemOut(BaseModel):
    id: int
    source: Optional[str] = None
    source_type: Optional[str] = None
    title: str
    text: str
    url: str
    canonical_url: str
    author: Optional[str] = None
    published_at: Optional[str] = None
    published_date: Optional[str] = None
    collected_at: str
    predicted_label: str
    effective_label: str
    confidence: float
    probabilities: Dict[str, float]
    is_ambiguous: bool
    ambiguous_reason: Optional[str] = None
    protocol_summary: str
    keywords: List[KeywordMatchOut] = Field(default_factory=list)
    reviews_count: int = 0
    telemetry: Optional[Dict[str, Any]] = None


class NewsListResponse(BaseModel):
    items: List[NewsItemOut]
    total: int
    limit: int
    offset: int


class LabelUpdateRequest(BaseModel):
    label: str = Field(..., description="Novo rótulo: Boa, Ruim ou Irrelevante.")
    reviewer: str = Field(default="human", min_length=1, max_length=64)
    notes: Optional[str] = Field(default=None, max_length=500)

    @validator("label")
    def _validate_label(cls, value: str) -> str:
        clean = value.strip()
        if clean not in VALID_LABELS:
            raise ValueError(f"Rótulo inválido '{value}'. Use: {VALID_LABELS}")
        return clean


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    total_stored_items: int


class AnalyticsResponse(BaseModel):
    overview: Dict[str, Any]
    daily_series: List[Dict[str, Any]]
    keyword_distribution: Dict[str, Any]
    protocol_stats: Dict[str, Any]
    model_metrics: Optional[Dict[str, Any]] = None
    benchmark_summary: Optional[Dict[str, Any]] = None
    sources_available: List[Dict[str, str]]
