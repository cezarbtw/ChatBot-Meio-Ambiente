"""Definição das rotas REST em FastAPI."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from src.api.dependencies import (
    AppContainer,
    get_chatbot,
    get_classifier,
    get_container,
    get_repository,
)
from src.api.extractor import UrlExtractionError, fetch_and_extract_url
from src.api.schemas import (
    AnalyticsResponse,
    ChatCitationOut,
    ChatClassificationOut,
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    CollectRequest,
    CollectResponse,
    HealthResponse,
    KeywordMatchOut,
    LabelUpdateRequest,
    NewsItemOut,
    NewsListResponse,
    SourceReportOut,
)
from src.collectors import COLLECTOR_FACTORIES, CollectedItem
from src.config import BENCHMARK_FILENAME, DEFAULT_BERT_MODEL, METRICS_FILENAME
from src.pipeline import run_monitoring_cycle
from src.storage import NewsRepository, compute_content_hash, extract_keyword_counts

router = APIRouter(prefix="/api", tags=["Monitor Ambiental"])


def _read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@router.get("/health", response_model=HealthResponse)
def health_check(container: AppContainer = Depends(get_container)) -> HealthResponse:
    """Verifica saúde da API, modelo carregado e contagem de registros no SQLite."""
    overview = container.repository.get_overview_stats()
    model_name = None
    if container.classifier is not None:
        model_name = getattr(
            container.classifier.embedder, "model_name", DEFAULT_BERT_MODEL
        )
    return HealthResponse(
        status="ok",
        model_loaded=container.classifier is not None,
        model_name=model_name,
        total_stored_items=overview["total_items"],
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify_news(
    payload: ClassifyRequest,
    classifier=Depends(get_classifier),
    repo: NewsRepository = Depends(get_repository),
) -> ClassifyResponse:
    """Classifica título + texto manual ou extrai conteúdo de uma URL."""
    title = payload.title or ""
    text = payload.text or ""
    url = payload.url
    published_at: Optional[str] = None
    protocol_summary: Optional[str] = None
    telemetry_dict: Optional[Dict[str, Any]] = None

    if url:
        try:
            extracted = fetch_and_extract_url(url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UrlExtractionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        title = title or extracted.title
        text = text or extracted.text
        url = extracted.url
        published_at = extracted.published_at
        protocol_summary = extracted.telemetry.summary_protocol()
        telemetry_dict = extracted.telemetry.to_dict()

    if not title and not text:
        raise HTTPException(
            status_code=422, detail="Conteúdo vazio após extração."
        )

    pred = classifier.predict_detailed(title, text)
    kw_counts = extract_keyword_counts(title, text)
    keywords_out = [
        KeywordMatchOut(keyword=k, occurrences=v) for k, v in kw_counts.items()
    ]

    saved_item_id: Optional[int] = None
    created: Optional[bool] = None
    if payload.save_to_db:
        item = CollectedItem(
            source="url" if url else "manual",
            source_type="manual",
            title=title or text[:80],
            text=text or title,
            url=url or f"manual://{compute_content_hash(title, text)[:16]}",
            published_at=published_at,
            protocol=protocol_summary or "manual",
            network_telemetry=telemetry_dict,
        )
        saved_dict, created = repo.save_single_item(item, pred)
        saved_item_id = saved_dict["id"]

    return ClassifyResponse(
        label=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        is_ambiguous=pred.is_ambiguous,
        ambiguous_reason=pred.ambiguous_reason,
        matched_ambiguous_terms=pred.matched_ambiguous_terms,
        keywords=keywords_out,
        extracted_title=title,
        extracted_text=text,
        url=url,
        protocol_summary=protocol_summary,
        network_telemetry=telemetry_dict,
        saved_item_id=saved_item_id,
        created=created,
    )


@router.post("/collect", response_model=CollectResponse)
def trigger_collection(
    payload: CollectRequest,
    container: AppContainer = Depends(get_container),
    classifier=Depends(get_classifier),
) -> CollectResponse:
    """Aciona coleta sob demanda nas fontes solicitadas e persiste no SQLite."""
    if payload.sources is not None:
        unknown = [s for s in payload.sources if s not in COLLECTOR_FACTORIES]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Fontes desconhecidas: {unknown}. Disponíveis: {sorted(COLLECTOR_FACTORIES)}",
            )

    custom_collectors = (
        container.collectors_factory()
        if container.collectors_factory is not None
        else None
    )
    cycle = run_monitoring_cycle(
        sources=payload.sources,
        limit_per_source=payload.limit_per_source,
        inspect_items=payload.inspect_items,
        classifier=classifier,
        collectors=custom_collectors,
        repository=container.repository,
    )

    return CollectResponse(
        run_id=cycle.run_id,
        collected=cycle.collected,
        inserted=cycle.inserted,
        deduplicated=cycle.deduplicated,
        errors_count=cycle.errors_count,
        duration_ms=round(cycle.duration_ms, 2),
        sources=[
            SourceReportOut(
                source=rep.source,
                source_type=rep.source_type,
                success=rep.success,
                items_count=len(rep.items),
                duration_ms=round(rep.duration_ms, 2),
                error=rep.error,
                source_telemetry=rep.source_telemetry,
            )
            for rep in cycle.source_reports
        ],
    )


@router.get("/news", response_model=NewsListResponse)
def list_news(
    label: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    protocol: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    ambiguous_only: bool = Query(default=False),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    repo: NewsRepository = Depends(get_repository),
) -> NewsListResponse:
    """Lista notícias/posts armazenados com filtros combináveis e paginação."""
    items, total = repo.list_news(
        label=label,
        source=source,
        source_type=source_type,
        date_from=date_from,
        date_to=date_to,
        protocol=protocol,
        query=q,
        ambiguous_only=ambiguous_only,
        limit=limit,
        offset=offset,
    )
    return NewsListResponse(
        items=[NewsItemOut(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/news/export")
def export_news_csv(
    label: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    source_type: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    protocol: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    ambiguous_only: bool = Query(default=False),
    repo: NewsRepository = Depends(get_repository),
) -> Response:
    """Exporta notícias filtradas em formato CSV UTF-8."""
    items, _ = repo.list_news(
        label=label,
        source=source,
        source_type=source_type,
        date_from=date_from,
        date_to=date_to,
        protocol=protocol,
        query=q,
        ambiguous_only=ambiguous_only,
        limit=1000,
        offset=0,
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "source",
            "source_type",
            "published_date",
            "effective_label",
            "predicted_label",
            "confidence",
            "is_ambiguous",
            "protocol_summary",
            "title",
            "text",
            "url",
        ]
    )
    for it in items:
        writer.writerow(
            [
                it["id"],
                it["source"],
                it["source_type"],
                it["published_date"] or "",
                it["effective_label"],
                it["predicted_label"],
                round(it["confidence"], 4),
                it["is_ambiguous"],
                it["protocol_summary"],
                it["title"],
                it["text"],
                it["url"],
            ]
        )

    return Response(
        content=buffer.getvalue().encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="noticias_monitoradas.csv"'},
    )


@router.get("/news/{item_id}", response_model=NewsItemOut)
def get_news_detail(
    item_id: int,
    repo: NewsRepository = Depends(get_repository),
) -> NewsItemOut:
    """Retorna os detalhes completos de uma notícia pelo ID."""
    item = repo.get_news_by_id(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Notícia {item_id} não encontrada.")
    return NewsItemOut(**item)


@router.patch("/news/{item_id}/label", response_model=NewsItemOut)
def update_news_label(
    item_id: int,
    payload: LabelUpdateRequest,
    repo: NewsRepository = Depends(get_repository),
) -> NewsItemOut:
    """Registra correção manual do rótulo para curadoria e retreinamento futuro."""
    try:
        updated = repo.add_label_review(
            item_id=item_id,
            new_label=payload.label,
            reviewer=payload.reviewer,
            notes=payload.notes,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return NewsItemOut(**updated)


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    container: AppContainer = Depends(get_container),
) -> AnalyticsResponse:
    """Retorna agregações para o dashboard: séries diárias, palavras-chave, rede e modelo."""
    repo = container.repository
    sources_available = []
    for name, factory in sorted(COLLECTOR_FACTORIES.items()):
        collector = factory()
        sources_available.append(
            {"name": name, "source_type": collector.source_type}
        )

    benchmark_raw = _read_optional_json(container.artifacts_dir / BENCHMARK_FILENAME)
    benchmark_summary = None
    if benchmark_raw:
        benchmark_summary = {
            "generated_at_utc": benchmark_raw.get("generated_at_utc"),
            "total_configurations": benchmark_raw.get("total_configurations"),
            "best_configuration": benchmark_raw.get("best_configuration"),
            "top_results": (benchmark_raw.get("results") or [])[:8],
        }

    return AnalyticsResponse(
        overview=repo.get_overview_stats(),
        daily_series=repo.get_daily_series(
            date_from=date_from, date_to=date_to, source=source
        ),
        keyword_distribution=repo.get_keyword_distribution(),
        protocol_stats=repo.get_protocol_stats(),
        model_metrics=_read_optional_json(
            container.artifacts_dir / METRICS_FILENAME
        ),
        benchmark_summary=benchmark_summary,
        sources_available=sources_available,
    )


@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(
    payload: ChatRequest,
    chatbot=Depends(get_chatbot),
) -> ChatResponse:
    """Responde consultas sobre o acervo, protocolos ou classifica texto enviado no chat."""
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    reply = chatbot.reply(
        message=payload.message,
        history=history,
        use_llm=payload.use_llm,
    )
    classification_out = None
    if reply.classification is not None:
        classification_out = ChatClassificationOut(
            label=reply.classification.label,
            confidence=reply.classification.confidence,
            probabilities=reply.classification.probabilities,
            is_ambiguous=reply.classification.is_ambiguous,
            ambiguous_reason=reply.classification.ambiguous_reason,
            keywords=[
                KeywordMatchOut(keyword=k["keyword"], occurrences=k["occurrences"])
                for k in reply.classification.keywords
            ],
        )

    return ChatResponse(
        answer=reply.answer,
        intent=reply.intent,
        mode=reply.mode,
        citations=[
            ChatCitationOut(
                id=c.id,
                title=c.title,
                url=c.url,
                source=c.source,
                effective_label=c.effective_label,
                confidence=c.confidence,
                published_date=c.published_date,
                protocol_summary=c.protocol_summary,
            )
            for c in reply.citations
        ],
        classification=classification_out,
        suggested_followups=reply.suggested_followups,
        analytics_snapshot=reply.analytics_snapshot,
    )
