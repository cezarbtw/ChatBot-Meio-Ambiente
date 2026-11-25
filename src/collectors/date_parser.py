"""Extração de datas multi-estratégia (JSON-LD, meta tags, RSS, URL e texto PT-BR)."""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import re
from typing import Any, Iterable, Optional

from bs4 import BeautifulSoup

_DATE_KEYS = ("datePublished", "uploadDate", "dateCreated", "dateModified")
_URL_DATE_SLASH = re.compile(r"/(\d{4}/\d{2}/\d{2})/")
_URL_DATE_DASH = re.compile(r"/(\d{4}-\d{2}-\d{2})(?:/|$)")
_RELATIVE_HOURS = re.compile(r"h[aá]\s+(\d+)\s+hora", re.IGNORECASE)
_RELATIVE_MINUTES = re.compile(r"h[aá]\s+(\d+)\s+minuto", re.IGNORECASE)
_RELATIVE_DAYS = re.compile(r"h[aá]\s+(\d+)\s+dia", re.IGNORECASE)
_BR_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")


def normalize_datetime_str(raw: Optional[str]) -> Optional[str]:
    """Normaliza string ISO-8601 ou RFC-2822 para ISO-8601 com data YYYY-MM-DD."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    iso_candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return dt.isoformat()
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(text)
        return dt.isoformat()
    except (TypeError, ValueError):
        pass

    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)
    return None


def _iter_json_ld_objects(data: Any) -> Iterable[dict]:
    if isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                yield from _iter_json_ld_objects(item)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_json_ld_objects(item)


def parse_json_ld_date(soup: BeautifulSoup) -> Optional[str]:
    """Extrai data de publicação de blocos <script type='application/ld+json'>."""
    for script in soup.select("script[type='application/ld+json']"):
        raw = script.string or script.get_text()
        if not raw or not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in _iter_json_ld_objects(parsed):
            for key in _DATE_KEYS:
                val = obj.get(key)
                if isinstance(val, str):
                    normalized = normalize_datetime_str(val)
                    if normalized:
                        return normalized
    return None


def parse_meta_date(soup: BeautifulSoup) -> Optional[str]:
    """Extrai data de meta tags OpenGraph/article/Schema.org ou tags <time>."""
    selectors = (
        ("meta[property='article:published_time']", "content"),
        ("meta[property='og:article:published_time']", "content"),
        ("meta[property='og:published_time']", "content"),
        ("meta[name='article:published_time']", "content"),
        ("meta[name='dt_pub']", "content"),
        ("meta[name='publish-date']", "content"),
        ("meta[name='pubdate']", "content"),
        ("meta[itemprop='datePublished']", "content"),
        ("time[datetime]", "datetime"),
    )
    for selector, attr in selectors:
        tag = soup.select_one(selector)
        if tag:
            normalized = normalize_datetime_str(tag.get(attr))
            if normalized:
                return normalized
    return None


def parse_url_date(url: str) -> Optional[str]:
    """Extrai YYYY-MM-DD de padrões de data na URL (ex.: /2026/11/25/)."""
    if not url:
        return None
    match_slash = _URL_DATE_SLASH.search(url)
    if match_slash:
        return match_slash.group(1).replace("/", "-")
    match_dash = _URL_DATE_DASH.search(url)
    if match_dash:
        return match_dash.group(1)
    return None


def parse_relative_pt_date(
    text: Optional[str], reference_dt: Optional[datetime] = None
) -> Optional[str]:
    """Converte expressões relativas em português ('Hoje', 'Ontem', 'Há 2 dias') para YYYY-MM-DD."""
    if not text or not isinstance(text, str):
        return None
    clean = text.strip()
    if not clean:
        return None

    ref = reference_dt or datetime.now(timezone.utc)
    lowered = clean.lower()

    if "hoje" in lowered or "agora" in lowered:
        return ref.date().isoformat()
    if "ontem" in lowered:
        return (ref - timedelta(days=1)).date().isoformat()

    m_min = _RELATIVE_MINUTES.search(clean)
    if m_min:
        return (ref - timedelta(minutes=int(m_min.group(1)))).date().isoformat()

    m_hour = _RELATIVE_HOURS.search(clean)
    if m_hour:
        return (ref - timedelta(hours=int(m_hour.group(1)))).date().isoformat()

    m_day = _RELATIVE_DAYS.search(clean)
    if m_day:
        return (ref - timedelta(days=int(m_day.group(1)))).date().isoformat()

    m_br = _BR_DATE.search(clean)
    if m_br:
        day, month, year = m_br.groups()
        return f"{year}-{month}-{day}"

    return None


def extract_published_date(
    article_html: Optional[str] = None,
    url: str = "",
    card_date_text: Optional[str] = None,
    reference_dt: Optional[datetime] = None,
) -> Optional[str]:
    """Resolve data de publicação por cascata: JSON-LD -> meta tags -> URL -> texto do card."""
    if article_html:
        soup = BeautifulSoup(article_html, "html.parser")
        json_ld_date = parse_json_ld_date(soup)
        if json_ld_date:
            return json_ld_date
        meta_date = parse_meta_date(soup)
        if meta_date:
            return meta_date

    url_date = parse_url_date(url)
    if url_date:
        return url_date

    return parse_relative_pt_date(card_date_text, reference_dt=reference_dt)
