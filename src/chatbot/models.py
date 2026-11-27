"""Modelos de dados estruturados para o assistente conversacional."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatCitation:
    """Referência clicável a uma notícia ou post armazenado no banco."""

    id: int
    title: str
    url: str
    source: Optional[str]
    effective_label: str
    confidence: float
    published_date: Optional[str]
    protocol_summary: str


@dataclass
class ChatClassification:
    """Resultado de classificação direta solicitada pelo usuário no chat."""

    label: str
    confidence: float
    probabilities: Dict[str, float]
    is_ambiguous: bool
    ambiguous_reason: Optional[str] = None
    keywords: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatReply:
    """Resposta estruturada do ChatBot ambiental."""

    answer: str
    intent: str
    mode: str
    citations: List[ChatCitation] = field(default_factory=list)
    classification: Optional[ChatClassification] = None
    suggested_followups: List[str] = field(default_factory=list)
    analytics_snapshot: Optional[Dict[str, Any]] = None
