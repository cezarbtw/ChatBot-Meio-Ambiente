"""Repositório para persistência, deduplicação, curadoria e consultas analíticas."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import Engine, func, or_, select
from sqlalchemy.orm import Session, joinedload

from src.collectors.models import CollectedItem, CollectorResult
from src.config import DEFAULT_KEYWORDS, VALID_LABELS
from src.nlp.classifier import PredictionResult
from src.storage.database import create_session_factory
from src.storage.dedup import (
    canonicalize_url,
    compute_content_hash,
    extract_keyword_counts,
)
from src.storage.models import (
    CollectionRun,
    KeywordMatchRecord,
    LabelReviewRecord,
    NetworkTelemetryRecord,
    NewsItemRecord,
    SourceRecord,
)


@dataclass
class SaveBatchResult:
    """Resumo de gravação incremental de uma rodada de coleta."""

    run_id: int
    items_collected: int
    items_inserted: int
    items_deduplicated: int
    errors_count: int


def _telemetry_dict_to_record(
    tel: Dict[str, Any],
    source_name: str,
    run_id: Optional[int] = None,
    news_item_id: Optional[int] = None,
) -> NetworkTelemetryRecord:
    cert = tel.get("certificate") or {}
    cert_issuer = cert.get("issuer_org") or cert.get("issuer_cn")
    return NetworkTelemetryRecord(
        news_item_id=news_item_id,
        run_id=run_id,
        source_name=source_name,
        url=str(tel.get("url") or ""),
        final_url=str(tel.get("final_url") or tel.get("url") or ""),
        scheme=str(tel.get("scheme") or ""),
        host=str(tel.get("host") or ""),
        port=int(tel.get("port") or 443),
        resolved_ips_json=json.dumps(tel.get("resolved_ips") or []),
        http_version=tel.get("http_version"),
        alpn_protocol=tel.get("alpn_protocol"),
        tls_version=tel.get("tls_version"),
        cipher_suite=tel.get("cipher_suite"),
        cert_issuer=cert_issuer,
        cert_valid_until=cert.get("valid_until"),
        cert_verified=cert.get("verified"),
        status_code=tel.get("status_code"),
        content_type=tel.get("content_type"),
        redirect_count=len(tel.get("redirect_chain") or []),
        dns_ms=tel.get("dns_ms"),
        tcp_ms=tel.get("tcp_ms"),
        tls_ms=tel.get("tls_ms"),
        ttfb_ms=tel.get("ttfb_ms"),
        total_ms=tel.get("total_ms"),
        error=tel.get("error"),
        inspected_at=str(
            tel.get("inspected_at") or datetime.now(timezone.utc).isoformat()
        ),
    )


def _serialize_news_item(item: NewsItemRecord) -> Dict[str, Any]:
    probabilities = json.loads(item.probabilities_json)
    latest_tel = item.telemetry_entries[-1] if item.telemetry_entries else None
    return {
        "id": item.id,
        "source": item.source.name if item.source else None,
        "source_type": item.source.source_type if item.source else None,
        "title": item.title,
        "text": item.text,
        "url": item.url,
        "canonical_url": item.canonical_url,
        "author": item.author,
        "published_at": item.published_at,
        "published_date": item.published_date,
        "collected_at": item.collected_at,
        "predicted_label": item.predicted_label,
        "effective_label": item.effective_label,
        "confidence": item.confidence,
        "probabilities": probabilities,
        "is_ambiguous": item.is_ambiguous,
        "ambiguous_reason": item.ambiguous_reason,
        "protocol_summary": item.protocol_summary,
        "keywords": [
            {"keyword": kw.keyword, "occurrences": kw.occurrences}
            for kw in item.keyword_matches
        ],
        "reviews_count": len(item.label_reviews),
        "telemetry": (
            {
                "host": latest_tel.host,
                "http_version": latest_tel.http_version,
                "alpn_protocol": latest_tel.alpn_protocol,
                "tls_version": latest_tel.tls_version,
                "cipher_suite": latest_tel.cipher_suite,
                "status_code": latest_tel.status_code,
                "dns_ms": latest_tel.dns_ms,
                "tls_ms": latest_tel.tls_ms,
                "ttfb_ms": latest_tel.ttfb_ms,
                "resolved_ips": json.loads(latest_tel.resolved_ips_json),
            }
            if latest_tel is not None
            else None
        ),
    }


class NewsRepository:
    """Camada de acesso e agregação analítica sobre o banco SQLite."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.session_factory = create_session_factory(engine)

    def _get_or_create_source(
        self, session: Session, name: str, source_type: str
    ) -> SourceRecord:
        existing = session.scalar(
            select(SourceRecord).where(SourceRecord.name == name)
        )
        if existing is not None:
            return existing
        created = SourceRecord(name=name, source_type=source_type)
        session.add(created)
        session.flush()
        return created

    def save_collection_batch(
        self,
        sources_requested: Sequence[str],
        items_with_predictions: Sequence[Tuple[CollectedItem, PredictionResult]],
        reports: Optional[Sequence[CollectorResult]] = None,
        notes: Optional[str] = None,
    ) -> SaveBatchResult:
        """Grava rodada de coleta com deduplicação por URL canônica e hash de conteúdo."""
        reports_list = list(reports or ())
        errors_count = sum(1 for rep in reports_list if not rep.success)
        started_at = datetime.now(timezone.utc).isoformat()

        with self.session_factory() as session:
            run = CollectionRun(
                started_at=started_at,
                sources_requested=",".join(sources_requested),
                items_collected=len(items_with_predictions),
                errors_count=errors_count,
                notes=notes,
            )
            session.add(run)
            session.flush()

            existing_urls = set(
                session.scalars(select(NewsItemRecord.canonical_url)).all()
            )
            existing_hashes = set(
                session.scalars(select(NewsItemRecord.content_hash)).all()
            )

            inserted = 0
            deduplicated = 0

            for item, pred in items_with_predictions:
                canon_url = canonicalize_url(item.url) or item.url
                c_hash = compute_content_hash(item.title, item.text)

                if canon_url in existing_urls or c_hash in existing_hashes:
                    deduplicated += 1
                    continue

                source_rec = self._get_or_create_source(
                    session, item.source, item.source_type
                )
                published_date = (
                    item.published_date
                    if item.published_date != "Data não disponível"
                    else None
                )

                news_rec = NewsItemRecord(
                    source_id=source_rec.id,
                    run_id=run.id,
                    title=item.title,
                    text=item.text,
                    url=item.url,
                    canonical_url=canon_url,
                    content_hash=c_hash,
                    author=item.author,
                    published_at=item.published_at,
                    published_date=published_date,
                    collected_at=item.collected_at,
                    predicted_label=pred.label,
                    effective_label=pred.label,
                    confidence=float(pred.confidence),
                    probabilities_json=json.dumps(
                        pred.probabilities, ensure_ascii=False
                    ),
                    is_ambiguous=bool(pred.is_ambiguous),
                    ambiguous_reason=pred.ambiguous_reason,
                    protocol_summary=item.protocol,
                )
                session.add(news_rec)
                session.flush()

                kw_counts = extract_keyword_counts(item.title, item.text)
                for kw, occ in kw_counts.items():
                    session.add(
                        KeywordMatchRecord(
                            news_item_id=news_rec.id,
                            keyword=kw,
                            occurrences=occ,
                        )
                    )

                if item.network_telemetry:
                    session.add(
                        _telemetry_dict_to_record(
                            item.network_telemetry,
                            source_name=item.source,
                            run_id=run.id,
                            news_item_id=news_rec.id,
                        )
                    )

                existing_urls.add(canon_url)
                existing_hashes.add(c_hash)
                inserted += 1

            # Persiste telemetria de nível de fonte mesmo se todos os itens forem deduplicados
            for rep in reports_list:
                for tel in rep.telemetry_records:
                    session.add(
                        _telemetry_dict_to_record(
                            tel,
                            source_name=rep.source,
                            run_id=run.id,
                            news_item_id=None,
                        )
                    )

            run.items_inserted = inserted
            run.items_deduplicated = deduplicated
            run.finished_at = datetime.now(timezone.utc).isoformat()
            session.commit()

            return SaveBatchResult(
                run_id=run.id,
                items_collected=len(items_with_predictions),
                items_inserted=inserted,
                items_deduplicated=deduplicated,
                errors_count=errors_count,
            )

    def save_single_item(
        self,
        item: CollectedItem,
        prediction: PredictionResult,
    ) -> Tuple[Dict[str, Any], bool]:
        """Persiste um único item classificado; retorna (item_dict, created_flag)."""
        canon_url = canonicalize_url(item.url) or item.url
        c_hash = compute_content_hash(item.title, item.text)

        with self.session_factory() as session:
            existing = session.scalar(
                select(NewsItemRecord)
                .options(
                    joinedload(NewsItemRecord.source),
                    joinedload(NewsItemRecord.keyword_matches),
                    joinedload(NewsItemRecord.label_reviews),
                    joinedload(NewsItemRecord.telemetry_entries),
                )
                .where(
                    or_(
                        NewsItemRecord.canonical_url == canon_url,
                        NewsItemRecord.content_hash == c_hash,
                    )
                )
            )
            if existing is not None:
                return _serialize_news_item(existing), False

            source_rec = self._get_or_create_source(
                session, item.source, item.source_type
            )
            published_date = (
                item.published_date
                if item.published_date != "Data não disponível"
                else None
            )
            news_rec = NewsItemRecord(
                source_id=source_rec.id,
                title=item.title,
                text=item.text,
                url=item.url,
                canonical_url=canon_url,
                content_hash=c_hash,
                author=item.author,
                published_at=item.published_at,
                published_date=published_date,
                collected_at=item.collected_at,
                predicted_label=prediction.label,
                effective_label=prediction.label,
                confidence=float(prediction.confidence),
                probabilities_json=json.dumps(
                    prediction.probabilities, ensure_ascii=False
                ),
                is_ambiguous=bool(prediction.is_ambiguous),
                ambiguous_reason=prediction.ambiguous_reason,
                protocol_summary=item.protocol,
            )
            session.add(news_rec)
            session.flush()

            for kw, occ in extract_keyword_counts(item.title, item.text).items():
                session.add(
                    KeywordMatchRecord(
                        news_item_id=news_rec.id,
                        keyword=kw,
                        occurrences=occ,
                    )
                )

            if item.network_telemetry:
                session.add(
                    _telemetry_dict_to_record(
                        item.network_telemetry,
                        source_name=item.source,
                        news_item_id=news_rec.id,
                    )
                )

            session.commit()
            loaded = session.scalar(
                select(NewsItemRecord)
                .options(
                    joinedload(NewsItemRecord.source),
                    joinedload(NewsItemRecord.keyword_matches),
                    joinedload(NewsItemRecord.label_reviews),
                    joinedload(NewsItemRecord.telemetry_entries),
                )
                .where(NewsItemRecord.id == news_rec.id)
            )
            assert loaded is not None
            return _serialize_news_item(loaded), True

    def list_news(
        self,
        label: Optional[str] = None,
        source: Optional[str] = None,
        source_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        protocol: Optional[str] = None,
        query: Optional[str] = None,
        ambiguous_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Lista notícias com filtros combináveis, paginação e contagem total."""
        with self.session_factory() as session:
            stmt = select(NewsItemRecord).join(NewsItemRecord.source)
            count_stmt = select(func.count(NewsItemRecord.id)).join(
                NewsItemRecord.source
            )

            filters = []
            if label:
                filters.append(NewsItemRecord.effective_label == label)
            if source:
                filters.append(SourceRecord.name == source)
            if source_type:
                filters.append(SourceRecord.source_type == source_type)
            if date_from:
                filters.append(NewsItemRecord.published_date >= date_from)
            if date_to:
                filters.append(NewsItemRecord.published_date <= date_to)
            if protocol:
                filters.append(NewsItemRecord.protocol_summary.ilike(f"%{protocol}%"))
            if query:
                like_term = f"%{query}%"
                filters.append(
                    or_(
                        NewsItemRecord.title.ilike(like_term),
                        NewsItemRecord.text.ilike(like_term),
                    )
                )
            if ambiguous_only:
                filters.append(NewsItemRecord.is_ambiguous.is_(True))

            for cond in filters:
                stmt = stmt.where(cond)
                count_stmt = count_stmt.where(cond)

            total = int(session.scalar(count_stmt) or 0)
            rows = (
                session.scalars(
                    stmt.options(
                        joinedload(NewsItemRecord.source),
                        joinedload(NewsItemRecord.keyword_matches),
                        joinedload(NewsItemRecord.label_reviews),
                        joinedload(NewsItemRecord.telemetry_entries),
                    )
                    .order_by(
                        NewsItemRecord.published_date.desc(),
                        NewsItemRecord.id.desc(),
                    )
                    .offset(offset)
                    .limit(limit)
                )
                .unique()
                .all()
            )
            return [_serialize_news_item(row) for row in rows], total

    def get_news_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Busca uma notícia pelo ID completo com telemetria e palavras-chave."""
        with self.session_factory() as session:
            row = session.scalar(
                select(NewsItemRecord)
                .options(
                    joinedload(NewsItemRecord.source),
                    joinedload(NewsItemRecord.keyword_matches),
                    joinedload(NewsItemRecord.label_reviews),
                    joinedload(NewsItemRecord.telemetry_entries),
                )
                .where(NewsItemRecord.id == item_id)
            )
            return _serialize_news_item(row) if row is not None else None

    def add_label_review(
        self,
        item_id: int,
        new_label: str,
        reviewer: str = "human",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Registra revisão manual de rótulo e atualiza effective_label."""
        clean_label = new_label.strip()
        if clean_label not in VALID_LABELS:
            raise ValueError(
                f"Rótulo inválido '{new_label}'. Valores aceitos: {VALID_LABELS}"
            )

        with self.session_factory() as session:
            item = session.scalar(
                select(NewsItemRecord)
                .options(
                    joinedload(NewsItemRecord.source),
                    joinedload(NewsItemRecord.keyword_matches),
                    joinedload(NewsItemRecord.label_reviews),
                    joinedload(NewsItemRecord.telemetry_entries),
                )
                .where(NewsItemRecord.id == item_id)
            )
            if item is None:
                raise LookupError(f"Notícia id={item_id} não encontrada.")

            review = LabelReviewRecord(
                news_item_id=item.id,
                previous_label=item.effective_label,
                new_label=clean_label,
                reviewer=reviewer,
                notes=notes,
            )
            item.effective_label = clean_label
            session.add(review)
            session.commit()
            session.refresh(item)
            return _serialize_news_item(item)

    def get_daily_series(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retorna série temporal diária por classe para gráficos multi-dia."""
        with self.session_factory() as session:
            stmt = (
                select(
                    NewsItemRecord.published_date,
                    NewsItemRecord.effective_label,
                    func.count(NewsItemRecord.id),
                )
                .join(NewsItemRecord.source)
                .where(NewsItemRecord.published_date.is_not(None))
            )
            if date_from:
                stmt = stmt.where(NewsItemRecord.published_date >= date_from)
            if date_to:
                stmt = stmt.where(NewsItemRecord.published_date <= date_to)
            if source:
                stmt = stmt.where(SourceRecord.name == source)

            stmt = stmt.group_by(
                NewsItemRecord.published_date, NewsItemRecord.effective_label
            ).order_by(NewsItemRecord.published_date.asc())

            by_date: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {lbl: 0 for lbl in VALID_LABELS}
            )
            for pub_date, lbl, cnt in session.execute(stmt).all():
                if pub_date:
                    by_date[pub_date][lbl] = int(cnt)

            series = []
            for pub_date in sorted(by_date):
                counts = by_date[pub_date]
                series.append(
                    {
                        "date": pub_date,
                        "Boa": counts.get("Boa", 0),
                        "Ruim": counts.get("Ruim", 0),
                        "Irrelevante": counts.get("Irrelevante", 0),
                        "total": sum(counts.values()),
                    }
                )
            return series

    def get_keyword_distribution(self) -> Dict[str, Any]:
        """Calcula ocorrências absolutas e percentuais das palavras-chave por classe."""
        with self.session_factory() as session:
            rows = session.execute(
                select(
                    NewsItemRecord.effective_label,
                    KeywordMatchRecord.keyword,
                    func.sum(KeywordMatchRecord.occurrences),
                )
                .join(KeywordMatchRecord.news_item)
                .group_by(NewsItemRecord.effective_label, KeywordMatchRecord.keyword)
            ).all()

        counts_by_label: Dict[str, Dict[str, int]] = {
            lbl: {kw: 0 for kw in DEFAULT_KEYWORDS} for lbl in VALID_LABELS
        }
        totals_by_keyword: Dict[str, int] = {kw: 0 for kw in DEFAULT_KEYWORDS}

        for lbl, kw, total_occ in rows:
            occ_int = int(total_occ or 0)
            if lbl in counts_by_label:
                counts_by_label[lbl][kw] = occ_int
            totals_by_keyword[kw] = totals_by_keyword.get(kw, 0) + occ_int

        percentages_by_label: Dict[str, Dict[str, float]] = {}
        for lbl in VALID_LABELS:
            percentages_by_label[lbl] = {}
            for kw in DEFAULT_KEYWORDS:
                total_kw = totals_by_keyword.get(kw, 0)
                pct = (
                    round((counts_by_label[lbl][kw] / total_kw) * 100.0, 2)
                    if total_kw > 0
                    else 0.0
                )
                percentages_by_label[lbl][kw] = pct

        return {
            "keywords": list(DEFAULT_KEYWORDS),
            "counts": counts_by_label,
            "percentages": percentages_by_label,
            "totals_by_keyword": totals_by_keyword,
        }

    def get_protocol_stats(self) -> Dict[str, Any]:
        """Agrega estatísticas de protocolos e latências das telemetrias armazenadas."""
        with self.session_factory() as session:
            records = session.scalars(select(NetworkTelemetryRecord)).all()

        schemes: Counter = Counter()
        http_versions: Counter = Counter()
        alpn_protocols: Counter = Counter()
        tls_versions: Counter = Counter()
        cipher_suites: Counter = Counter()
        status_codes: Counter = Counter()
        errors_count = 0
        dns_vals: List[float] = []
        tls_vals: List[float] = []
        ttfb_vals: List[float] = []

        for rec in records:
            if rec.scheme:
                schemes[rec.scheme] += 1
            if rec.http_version:
                http_versions[rec.http_version] += 1
            if rec.alpn_protocol:
                alpn_protocols[rec.alpn_protocol] += 1
            if rec.tls_version:
                tls_versions[rec.tls_version] += 1
            if rec.cipher_suite:
                cipher_suites[rec.cipher_suite] += 1
            if rec.status_code is not None:
                status_codes[str(rec.status_code)] += 1
            if rec.error:
                errors_count += 1
            if rec.dns_ms is not None:
                dns_vals.append(rec.dns_ms)
            if rec.tls_ms is not None:
                tls_vals.append(rec.tls_ms)
            if rec.ttfb_ms is not None:
                ttfb_vals.append(rec.ttfb_ms)

        return {
            "total_records": len(records),
            "errors_count": errors_count,
            "schemes": dict(schemes),
            "http_versions": dict(http_versions),
            "alpn_protocols": dict(alpn_protocols),
            "tls_versions": dict(tls_versions),
            "cipher_suites": dict(cipher_suites),
            "status_codes": dict(status_codes),
            "avg_dns_ms": round(sum(dns_vals) / len(dns_vals), 2) if dns_vals else None,
            "avg_tls_ms": round(sum(tls_vals) / len(tls_vals), 2) if tls_vals else None,
            "avg_ttfb_ms": round(sum(ttfb_vals) / len(ttfb_vals), 2) if ttfb_vals else None,
        }

    def get_overview_stats(self) -> Dict[str, Any]:
        """Retorna visão geral do banco para dashboard, CLI e ChatBot."""
        with self.session_factory() as session:
            total_items = int(
                session.scalar(select(func.count(NewsItemRecord.id))) or 0
            )
            ambiguous_items = int(
                session.scalar(
                    select(func.count(NewsItemRecord.id)).where(
                        NewsItemRecord.is_ambiguous.is_(True)
                    )
                )
                or 0
            )
            label_rows = session.execute(
                select(NewsItemRecord.effective_label, func.count(NewsItemRecord.id))
                .group_by(NewsItemRecord.effective_label)
            ).all()
            source_rows = session.execute(
                select(SourceRecord.name, func.count(NewsItemRecord.id))
                .join(NewsItemRecord.source)
                .group_by(SourceRecord.name)
            ).all()
            recent_runs = session.scalars(
                select(CollectionRun).order_by(CollectionRun.id.desc()).limit(5)
            ).all()

        by_label = {lbl: 0 for lbl in VALID_LABELS}
        for lbl, cnt in label_rows:
            by_label[lbl] = int(cnt)

        return {
            "total_items": total_items,
            "ambiguous_items": ambiguous_items,
            "by_label": by_label,
            "by_source": {name: int(cnt) for name, cnt in source_rows},
            "recent_runs": [
                {
                    "id": r.id,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "sources_requested": r.sources_requested,
                    "items_collected": r.items_collected,
                    "items_inserted": r.items_inserted,
                    "items_deduplicated": r.items_deduplicated,
                    "errors_count": r.errors_count,
                }
                for r in recent_runs
            ],
        }
