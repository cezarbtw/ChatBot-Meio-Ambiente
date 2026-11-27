"""Extração de título, conteúdo, data e telemetria de rede a partir de uma URL."""

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests

from src.collectors.base import build_http_session
from src.collectors.date_parser import extract_published_date
from src.network.inspector import NetworkInspector
from src.network.models import NetworkTelemetry


class UrlExtractionError(RuntimeError):
    """Erro ao baixar ou extrair texto de uma URL fornecida pelo usuário."""


@dataclass
class ExtractedArticle:
    url: str
    title: str
    text: str
    published_at: Optional[str]
    telemetry: NetworkTelemetry


def _clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_article_from_html(
    html: str,
    url: str,
    telemetry: NetworkTelemetry,
    max_text_chars: int = 2000,
) -> ExtractedArticle:
    """Extrai título, corpo/resumo e data de HTML já carregado."""
    soup = BeautifulSoup(html, "html.parser")

    title_candidates = [
        soup.select_one("meta[property='og:title']"),
        soup.select_one("meta[name='twitter:title']"),
    ]
    title = ""
    for tag in title_candidates:
        if tag and tag.get("content"):
            title = _clean_whitespace(str(tag.get("content")))
            if title:
                break

    if not title:
        h1 = soup.select_one("h1")
        if h1:
            title = _clean_whitespace(h1.get_text(" ", strip=True))
    if not title and soup.title:
        title = _clean_whitespace(soup.title.get_text(" ", strip=True))

    text = ""
    desc_candidates = [
        soup.select_one("meta[property='og:description']"),
        soup.select_one("meta[name='description']"),
    ]
    desc_text = ""
    for tag in desc_candidates:
        if tag and tag.get("content"):
            desc_text = _clean_whitespace(str(tag.get("content")))
            if desc_text:
                break

    paragraphs = [
        _clean_whitespace(p.get_text(" ", strip=True))
        for p in soup.select("article p, main p, p")
    ]
    paragraphs = [p for p in paragraphs if len(p) >= 30]
    body_text = " ".join(paragraphs[:6])[:max_text_chars].strip()

    if desc_text and body_text and desc_text not in body_text:
        text = f"{desc_text} {body_text}"[:max_text_chars].strip()
    else:
        text = body_text or desc_text

    published_at = extract_published_date(article_html=html, url=url)
    return ExtractedArticle(
        url=url,
        title=title,
        text=text,
        published_at=published_at,
        telemetry=telemetry,
    )


def fetch_and_extract_url(
    url: str,
    session: Optional[requests.Session] = None,
    inspector: Optional[NetworkInspector] = None,
    timeout: float = 15.0,
) -> ExtractedArticle:
    """Baixa uma URL HTTP(S), inspeciona rede/TLS e extrai título e conteúdo."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL inválida. Utilize um endereço http:// ou https://.")

    active_session = session or build_http_session(max_retries=1)
    active_inspector = inspector or NetworkInspector(timeout=min(timeout, 10.0))

    t0 = perf_counter()
    try:
        response = active_session.get(url, timeout=timeout, allow_redirects=True)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        telemetry = active_inspector.inspect_response(
            response, request_url=url, total_ms=elapsed_ms
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise UrlExtractionError(f"Falha ao acessar URL '{url}': {exc}") from exc

    article = extract_article_from_html(response.text, response.url or url, telemetry)
    if not article.title and not article.text:
        raise UrlExtractionError("Não foi possível extrair título ou texto da página.")
    return article
