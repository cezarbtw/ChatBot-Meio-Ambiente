"""Modelos padronizados de dados para coleta multi-fonte e telemetria de rede."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CollectedItem:
    """Representa uma notícia ou publicação social normalizada."""

    source: str
    source_type: str
    title: str
    text: str
    url: str
    published_at: Optional[str] = None
    author: Optional[str] = None
    protocol: str = "https"
    network_telemetry: Optional[Dict[str, Any]] = None
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def published_date(self) -> str:
        """Retorna YYYY-MM-DD quando disponível, ou 'Data não disponível'."""
        if not self.published_at:
            return "Data não disponível"
        return self.published_at[:10]


@dataclass
class CollectorResult:
    """Resultado da execução isolada de um coletor."""

    source: str
    source_type: str
    success: bool
    items: List[CollectedItem] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    source_telemetry: Optional[Dict[str, Any]] = None
    telemetry_records: List[Dict[str, Any]] = field(default_factory=list)
