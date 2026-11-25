"""Coletores multi-fonte para portais ambientais e redes sociais."""

from src.collectors.base import BaseCollector, build_http_session
from src.collectors.g1 import G1Collector
from src.collectors.manager import (
    COLLECTOR_FACTORIES,
    DEFAULT_SOURCES,
    collect_from_sources,
    create_collector,
)
from src.collectors.models import CollectedItem, CollectorResult
from src.collectors.rss import RssNewsCollector, strip_html
from src.collectors.social import MastodonCollector, RedditCollector

__all__ = [
    "BaseCollector",
    "COLLECTOR_FACTORIES",
    "CollectedItem",
    "CollectorResult",
    "DEFAULT_SOURCES",
    "G1Collector",
    "MastodonCollector",
    "RedditCollector",
    "RssNewsCollector",
    "build_http_session",
    "collect_from_sources",
    "create_collector",
    "strip_html",
]
