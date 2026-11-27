"""Modelos estruturados de telemetria de protocolos de rede e transporte."""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class RedirectHop:
    """Um salto intermediário na cadeia de redirecionamentos HTTP."""

    url: str
    status_code: int
    http_version: Optional[str] = None


@dataclass
class CertificateInfo:
    """Metadados do certificado X.509 negociado na conexão TLS."""

    subject_cn: Optional[str] = None
    issuer_cn: Optional[str] = None
    issuer_org: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    days_remaining: Optional[int] = None
    verified: bool = True


@dataclass
class NetworkTelemetry:
    """Telemetria completa de rede, transporte, TLS e HTTP de uma requisição."""

    url: str
    final_url: str
    scheme: str
    host: str
    port: int
    resolved_ips: List[str] = field(default_factory=list)
    http_version: Optional[str] = None
    alpn_protocol: Optional[str] = None
    tls_version: Optional[str] = None
    cipher_suite: Optional[str] = None
    certificate: Optional[CertificateInfo] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    redirect_chain: List[RedirectHop] = field(default_factory=list)
    dns_ms: Optional[float] = None
    tcp_ms: Optional[float] = None
    tls_ms: Optional[float] = None
    ttfb_ms: Optional[float] = None
    total_ms: Optional[float] = None
    error: Optional[str] = None
    inspected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary_protocol(self) -> str:
        """Retorna resumo legível dos protocolos negociados (ex.: HTTP/1.1 | ALPN h2 | TLSv1.3)."""
        parts: List[str] = []
        if self.http_version:
            parts.append(self.http_version)
        elif self.scheme:
            parts.append(self.scheme.upper())
        if self.alpn_protocol:
            parts.append(f"ALPN {self.alpn_protocol}")
        if self.tls_version:
            parts.append(self.tls_version)
        return " | ".join(parts) if parts else (self.scheme or "desconhecido")

    def to_dict(self) -> Dict[str, Any]:
        """Serializa telemetria em dicionário pronto para JSON."""
        return asdict(self)


def _avg_non_null(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def aggregate_protocol_telemetry(
    records: Sequence[NetworkTelemetry],
) -> Dict[str, Any]:
    """Agrega distribuições de protocolos e latências médias para API e gráficos."""
    schemes = Counter()
    http_versions = Counter()
    alpn_protocols = Counter()
    tls_versions = Counter()
    cipher_suites = Counter()
    status_codes = Counter()
    errors_count = 0

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

    return {
        "total_inspected": len(records),
        "errors_count": errors_count,
        "schemes": dict(schemes),
        "http_versions": dict(http_versions),
        "alpn_protocols": dict(alpn_protocols),
        "tls_versions": dict(tls_versions),
        "cipher_suites": dict(cipher_suites),
        "status_codes": dict(status_codes),
        "avg_dns_ms": _avg_non_null([r.dns_ms for r in records]),
        "avg_tcp_ms": _avg_non_null([r.tcp_ms for r in records]),
        "avg_tls_ms": _avg_non_null([r.tls_ms for r in records]),
        "avg_ttfb_ms": _avg_non_null([r.ttfb_ms for r in records]),
        "avg_total_ms": _avg_non_null([r.total_ms for r in records]),
    }
