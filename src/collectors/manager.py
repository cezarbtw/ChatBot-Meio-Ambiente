"""Registro de coletores e orquestração com isolamento de falhas por fonte."""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.collectors.base import BaseCollector
from src.collectors.g1 import G1Collector
from src.collectors.models import CollectedItem, CollectorResult
from src.collectors.rss import RssNewsCollector
from src.collectors.social import MastodonCollector, RedditCollector

COLLECTOR_FACTORIES: Dict[str, Callable[[], BaseCollector]] = {
    "g1": G1Collector,
    "agencia_brasil": lambda: RssNewsCollector(
        source_name="agencia_brasil",
        feed_url="https://agenciabrasil.ebc.com.br/rss/meio-ambiente/feed.xml",
    ),
    "oeco": lambda: RssNewsCollector(
        source_name="oeco",
        feed_url="https://oeco.org.br/feed/",
    ),
    "mongabay": lambda: RssNewsCollector(
        source_name="mongabay",
        feed_url="https://brasil.mongabay.com/feed/",
    ),
    "mastodon": MastodonCollector,
    "reddit": RedditCollector,
}

DEFAULT_SOURCES: Tuple[str, ...] = (
    "g1",
    "agencia_brasil",
    "oeco",
    "mongabay",
    "mastodon",
)


def create_collector(source_name: str) -> BaseCollector:
    """Instancia um coletor pelo nome registrado."""
    factory = COLLECTOR_FACTORIES.get(source_name)
    if factory is None:
        available = ", ".join(sorted(COLLECTOR_FACTORIES))
        raise ValueError(
            f"Fonte desconhecida '{source_name}'. Fontes disponíveis: {available}"
        )
    return factory()


def collect_from_sources(
    sources: Optional[Sequence[str]] = None,
    limit_per_source: int = 10,
    collectors: Optional[Sequence[BaseCollector]] = None,
    deduplicate: bool = True,
) -> Tuple[List[CollectedItem], List[CollectorResult]]:
    """Executa vários coletores independentemente; falhas em uma fonte não param as demais."""
    active_collectors: List[BaseCollector]
    if collectors is not None:
        active_collectors = list(collectors)
    else:
        target_sources = list(sources) if sources is not None else list(DEFAULT_SOURCES)
        active_collectors = [create_collector(name) for name in target_sources]

    all_items: List[CollectedItem] = []
    reports: List[CollectorResult] = []
    seen_urls = set()

    for collector in active_collectors:
        result = collector.safe_collect(limit=limit_per_source)
        reports.append(result)
        if not result.success:
            continue

        for item in result.items:
            if deduplicate and item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            all_items.append(item)

    return all_items, reports
