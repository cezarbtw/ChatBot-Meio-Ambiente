"""NLU determinístico local em português para roteamento de intenções e filtros."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Optional
import unicodedata

from src.collectors import COLLECTOR_FACTORIES
from src.config import DEFAULT_KEYWORDS

_CLASSIFY_PREFIX_RE = re.compile(
    r"^(?:classifi(?:que|car)|analise|avalie|julgue)(?:\s+(?:a\s+not[ií]cia|esta\s+not[ií]cia|esse\s+texto|este\s+texto|o\s+texto))?\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)

_STOPWORDS = frozenset(
    {
        "a",
        "o",
        "as",
        "os",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "para",
        "por",
        "com",
        "sobre",
        "quais",
        "qual",
        "que",
        "mostre",
        "me",
        "listar",
        "liste",
        "ver",
        "noticias",
        "noticia",
        "posts",
        "post",
        "ultimas",
        "ultimos",
        "recentes",
        "recente",
        "boas",
        "boa",
        "ruins",
        "ruim",
        "piores",
        "pior",
        "melhores",
        "melhor",
        "irrelevante",
        "irrelevantes",
        "negativas",
        "negativa",
        "positivas",
        "positiva",
        "hoje",
        "ontem",
        "semana",
        "ambientais",
        "ambiental",
        "meio",
        "ambiente",
        "esta",
        "esse",
        "sao",
        "tem",
        "como",
    }
)


def strip_accents(text: str) -> str:
    """Remove diacríticos para comparação tolerante a acentuação."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass
class ParsedIntent:
    """Intenção e entidades extraídas de uma mensagem do usuário."""

    intent: str
    text_to_classify: Optional[str] = None
    label_filter: Optional[str] = None
    source_filter: Optional[str] = None
    keyword_filter: Optional[str] = None
    search_query: Optional[str] = None
    date_from: Optional[str] = None
    ambiguous_only: bool = False


def parse_user_message(
    message: str,
    reference_dt: Optional[datetime] = None,
) -> ParsedIntent:
    """Classifica intenção local e extrai filtros de classe, fonte, palavra-chave e data."""
    raw = (message or "").strip()
    norm = strip_accents(raw).lower()
    now = reference_dt or datetime.now(timezone.utc)

    # 1. Classificação inline solicitada explicitamente
    classify_match = _CLASSIFY_PREFIX_RE.match(raw)
    if classify_match:
        return ParsedIntent(
            intent="classify_text",
            text_to_classify=classify_match.group(1).strip(),
        )

    # 2. Protocolos de rede / telemetria
    if any(
        term in norm
        for term in (
            "protocolo",
            "protocolos",
            "tls",
            "ssl",
            "http",
            "alpn",
            "cipher",
            "latencia",
            "trafego",
            "rede",
        )
    ) and "redes sociais" not in norm:
        return ParsedIntent(intent="protocol_summary")

    # 3. Métricas e desempenho do modelo
    if any(
        term in norm
        for term in (
            "acuracia",
            "f1",
            "desempenho",
            "metrica",
            "metricas",
            "benchmark",
            "modelo",
            "bert",
            "mlp",
        )
    ):
        return ParsedIntent(intent="model_performance")

    # Extração de filtros comuns para buscas e tendências
    label_filter: Optional[str] = None
    if any(w in norm for w in ("ruim", "ruins", "pior", "piores", "negativ")):
        label_filter = "Ruim"
    elif any(w in norm for w in ("boa", "boas", "melhor", "melhores", "positiv")):
        label_filter = "Boa"
    elif any(w in norm for w in ("irrelevante", "irrelevantes", "neutra", "neutras")):
        label_filter = "Irrelevante"

    source_filter: Optional[str] = None
    source_aliases = {
        "g1": "g1",
        "agencia brasil": "agencia_brasil",
        "agencia_brasil": "agencia_brasil",
        "oeco": "oeco",
        "((o))eco": "oeco",
        "mongabay": "mongabay",
        "mastodon": "mastodon",
        "reddit": "reddit",
    }
    for alias, canonical in source_aliases.items():
        if alias in norm and canonical in COLLECTOR_FACTORIES:
            source_filter = canonical
            break

    keyword_filter: Optional[str] = None
    for kw in DEFAULT_KEYWORDS:
        if strip_accents(kw).lower() in norm:
            keyword_filter = kw
            break
    if keyword_filter is None and ("queimada" in norm or "fogo" in norm):
        keyword_filter = "incêndios"

    date_from: Optional[str] = None
    if "hoje" in norm:
        date_from = now.date().isoformat()
    elif "ontem" in norm:
        date_from = (now - timedelta(days=1)).date().isoformat()
    elif "semana" in norm or "7 dias" in norm or "ultimos dias" in norm:
        date_from = (now - timedelta(days=7)).date().isoformat()

    ambiguous_only = "ambigu" in norm

    # 4. Panorama / tendências gerais
    if (
        any(
            term in norm
            for term in (
                "tendencia",
                "tendencias",
                "panorama",
                "resumo",
                "resuma",
                "estatistica",
                "estatisticas",
                "quantas",
                "quantos",
                "total",
                "ocorrencia",
                "palavras-chave",
                "palavra-chave",
            )
        )
        and not label_filter
        and not keyword_filter
        and not ambiguous_only
    ):
        return ParsedIntent(
            intent="trend_summary",
            source_filter=source_filter,
            date_from=date_from,
        )

    # 5. Busca no acervo de notícias/posts
    search_query: Optional[str] = keyword_filter
    if search_query is None and not label_filter and not source_filter and not ambiguous_only:
        tokens = [
            tok
            for tok in re.findall(r"[a-z0-9çãõáéíóúâêôàü]+", norm)
            if len(tok) >= 3 and tok not in _STOPWORDS
        ]
        if tokens:
            search_query = tokens[0]

    return ParsedIntent(
        intent="news_search",
        label_filter=label_filter,
        source_filter=source_filter,
        keyword_filter=keyword_filter,
        search_query=search_query,
        date_from=date_from,
        ambiguous_only=ambiguous_only,
    )
