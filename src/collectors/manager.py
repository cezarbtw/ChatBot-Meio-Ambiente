"""Registro de coletores, orquestração e consolidação de telemetria de rede."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.collectors.base import BaseCollector, build_http_session
from src.collectors.g1 import G1Collector
from src.collectors.models import CollectedItem, CollectorResult
from src.collectors.rss import RssNewsCollector
from src.collectors.social import MastodonCollector, RedditCollector
from src.config import ARTIFACTS_DIR, TELEMETRY_FILENAME
from src.network.inspector import NetworkInspector
from src.network.models import (
    CertificateInfo,
    NetworkTelemetry,
    RedirectHop,
    aggregate_protocol_telemetry,
)
from src.network.packet_capture import PacketCaptureSummary

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


def _telemetry_from_dict(data: Dict[str, Any]) -> NetworkTelemetry:
    cert_raw = data.get("certificate")
    cert = CertificateInfo(**cert_raw) if isinstance(cert_raw, dict) else None
    hops = [
        RedirectHop(**hop) if isinstance(hop, dict) else hop
        for hop in (data.get("redirect_chain") or [])
    ]
    return NetworkTelemetry(
        url=data["url"],
        final_url=data.get("final_url", data["url"]),
        scheme=data.get("scheme", ""),
        host=data.get("host", ""),
        port=int(data.get("port", 443)),
        resolved_ips=list(data.get("resolved_ips") or []),
        http_version=data.get("http_version"),
        alpn_protocol=data.get("alpn_protocol"),
        tls_version=data.get("tls_version"),
        cipher_suite=data.get("cipher_suite"),
        certificate=cert,
        status_code=data.get("status_code"),
        content_type=data.get("content_type"),
        redirect_chain=hops,
        dns_ms=data.get("dns_ms"),
        tcp_ms=data.get("tcp_ms"),
        tls_ms=data.get("tls_ms"),
        ttfb_ms=data.get("ttfb_ms"),
        total_ms=data.get("total_ms"),
        error=data.get("error"),
        inspected_at=data.get("inspected_at", ""),
    )


def collect_from_sources(
    sources: Optional[Sequence[str]] = None,
    limit_per_source: int = 10,
    collectors: Optional[Sequence[BaseCollector]] = None,
    deduplicate: bool = True,
    inspect_items: bool = False,
    inspector: Optional[NetworkInspector] = None,
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

    item_inspector = inspector or (NetworkInspector() if inspect_items else None)
    item_session = build_http_session(max_retries=1) if inspect_items else None

    for collector in active_collectors:
        result = collector.safe_collect(limit=limit_per_source)
        reports.append(result)
        if not result.success:
            continue

        for item in result.items:
            if deduplicate and item.url in seen_urls:
                continue
            seen_urls.add(item.url)

            if inspect_items and item_inspector is not None and item.url.startswith("http"):
                tel = item_inspector.inspect_url(
                    item.url, session=item_session, method="HEAD"
                )
                if tel.status_code == 405:
                    tel = item_inspector.inspect_url(
                        item.url, session=item_session, method="GET"
                    )
                item.protocol = tel.summary_protocol()
                item.network_telemetry = tel.to_dict()

            all_items.append(item)

    return all_items, reports


def build_telemetry_report(
    reports: Sequence[CollectorResult],
    items: Optional[Sequence[CollectedItem]] = None,
    capture_summary: Optional[PacketCaptureSummary] = None,
) -> Dict[str, Any]:
    """Consolida telemetria das fontes e dos itens para API, gráficos e exportação JSON."""
    raw_records: List[Dict[str, Any]] = []
    seen_keys = set()

    for rep in reports:
        for rec in rep.telemetry_records:
            key = (rec.get("final_url"), rec.get("inspected_at"))
            if key not in seen_keys:
                seen_keys.add(key)
                raw_records.append(rec)

    for item in items or ():
        if item.network_telemetry:
            rec = item.network_telemetry
            key = (rec.get("final_url"), rec.get("inspected_at"))
            if key not in seen_keys:
                seen_keys.add(key)
                raw_records.append(rec)

    parsed_records = [_telemetry_from_dict(rec) for rec in raw_records]
    summary = aggregate_protocol_telemetry(parsed_records)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "sources": [
            {
                "source": rep.source,
                "source_type": rep.source_type,
                "success": rep.success,
                "items_count": len(rep.items),
                "duration_ms": round(rep.duration_ms, 2),
                "error": rep.error,
                "source_telemetry": rep.source_telemetry,
            }
            for rep in reports
        ],
        "packet_capture": capture_summary.to_dict() if capture_summary else None,
        "records": raw_records,
    }


def export_telemetry_report(
    report: Dict[str, Any],
    output_path: Path = ARTIFACTS_DIR / TELEMETRY_FILENAME,
) -> Path:
    """Salva o relatório JSON de telemetria de protocolos de rede."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
