"""Pacote de persistência SQLite/SQLAlchemy, deduplicação e consultas analíticas."""

from src.storage.database import build_sqlite_url, create_db_engine, create_session_factory, init_db
from src.storage.dedup import canonicalize_url, compute_content_hash, extract_keyword_counts
from src.storage.models import (
    Base,
    CollectionRun,
    KeywordMatchRecord,
    LabelReviewRecord,
    NetworkTelemetryRecord,
    NewsItemRecord,
    SourceRecord,
)
from src.storage.repository import NewsRepository, SaveBatchResult

__all__ = [
    "Base",
    "CollectionRun",
    "KeywordMatchRecord",
    "LabelReviewRecord",
    "NetworkTelemetryRecord",
    "NewsItemRecord",
    "NewsRepository",
    "SaveBatchResult",
    "SourceRecord",
    "build_sqlite_url",
    "canonicalize_url",
    "compute_content_hash",
    "create_db_engine",
    "create_session_factory",
    "extract_keyword_counts",
    "init_db",
]
