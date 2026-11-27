"""Modelos ORM SQLAlchemy para persistência histórica, curadoria e telemetria."""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    """Base declarativa SQLAlchemy 2.0."""


class SourceRecord(Base):
    """Fonte monitorada (portal de notícias ou rede social)."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), default=_utc_now_iso, nullable=False)

    items: Mapped[List["NewsItemRecord"]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class CollectionRun(Base):
    """Registro de execução do pipeline de coleta e classificação."""

    __tablename__ = "collection_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[str] = mapped_column(String(64), default=_utc_now_iso, nullable=False)
    finished_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sources_requested: Mapped[str] = mapped_column(Text, nullable=False)
    items_collected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    items_deduplicated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    errors_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    items: Mapped[List["NewsItemRecord"]] = relationship(back_populates="run")
    telemetry_entries: Mapped[List["NetworkTelemetryRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class NewsItemRecord(Base):
    """Notícia ou publicação social armazenada com classificação e probabilidades."""

    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("collection_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(
        String(1024), unique=True, nullable=False, index=True
    )
    content_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    published_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    published_date: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True, index=True
    )
    collected_at: Mapped[str] = mapped_column(String(64), default=_utc_now_iso, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    effective_label: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities_json: Mapped[str] = mapped_column(Text, nullable=False)
    is_ambiguous: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ambiguous_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    protocol_summary: Mapped[str] = mapped_column(String(128), default="https", nullable=False)

    source: Mapped["SourceRecord"] = relationship(back_populates="items")
    run: Mapped[Optional["CollectionRun"]] = relationship(back_populates="items")
    keyword_matches: Mapped[List["KeywordMatchRecord"]] = relationship(
        back_populates="news_item", cascade="all, delete-orphan"
    )
    label_reviews: Mapped[List["LabelReviewRecord"]] = relationship(
        back_populates="news_item", cascade="all, delete-orphan"
    )
    telemetry_entries: Mapped[List["NetworkTelemetryRecord"]] = relationship(
        back_populates="news_item", cascade="all, delete-orphan"
    )


class KeywordMatchRecord(Base):
    """Ocorrências de palavras-chave ambientais detectadas em cada item."""

    __tablename__ = "keyword_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_item_id: Mapped[int] = mapped_column(
        ForeignKey("news_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    keyword: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False)

    news_item: Mapped["NewsItemRecord"] = relationship(back_populates="keyword_matches")


class LabelReviewRecord(Base):
    """Histórico de revisões humanas de rótulo para curadoria e retreinamento."""

    __tablename__ = "label_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_item_id: Mapped[int] = mapped_column(
        ForeignKey("news_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    previous_label: Mapped[str] = mapped_column(String(32), nullable=False)
    new_label: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(64), default="human", nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[str] = mapped_column(String(64), default=_utc_now_iso, nullable=False)

    news_item: Mapped["NewsItemRecord"] = relationship(back_populates="label_reviews")


class NetworkTelemetryRecord(Base):
    """Telemetria de protocolo HTTP/ALPN/TLS/DNS capturada durante a coleta."""

    __tablename__ = "network_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_item_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("news_items.id", ondelete="CASCADE"), nullable=True, index=True
    )
    run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("collection_runs.id", ondelete="CASCADE"), nullable=True, index=True
    )
    source_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    final_url: Mapped[str] = mapped_column(Text, nullable=False)
    scheme: Mapped[str] = mapped_column(String(16), nullable=False)
    host: Mapped[str] = mapped_column(String(255), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    resolved_ips_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    http_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    alpn_protocol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    tls_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    cipher_suite: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    cert_issuer: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    cert_valid_until: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cert_verified: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    redirect_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dns_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tcp_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tls_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ttfb_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    inspected_at: Mapped[str] = mapped_column(String(64), default=_utc_now_iso, nullable=False)

    news_item: Mapped[Optional["NewsItemRecord"]] = relationship(
        back_populates="telemetry_entries"
    )
    run: Mapped[Optional["CollectionRun"]] = relationship(back_populates="telemetry_entries")
