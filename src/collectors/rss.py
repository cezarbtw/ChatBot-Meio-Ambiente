"""Coletor genérico de feeds RSS 2.0 e Atom para portais ambientais."""

import re
from typing import List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests

from src.collectors.base import BaseCollector
from src.collectors.date_parser import normalize_datetime_str, parse_url_date
from src.collectors.models import CollectedItem

_DC_NS = "{http://purl.org/dc/elements/1.1/}"
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def strip_html(raw: Optional[str]) -> str:
    """Remove tags HTML de descrições RSS preservando texto limpo."""
    if not raw or not isinstance(raw, str):
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


class RssNewsCollector(BaseCollector):
    """Coleta notícias de feeds RSS 2.0 ou Atom padronizando campos e datas."""

    source_type = "news"

    def __init__(
        self,
        source_name: str,
        feed_url: str,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            session=session,
        )
        self.source_name = source_name
        self.feed_url = feed_url

    def _parse_rss_item(self, item: ET.Element) -> Optional[CollectedItem]:
        title = strip_html(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            return None

        raw_desc = item.findtext("description") or item.findtext(f"{_CONTENT_NS}encoded")
        text = strip_html(raw_desc)

        raw_date = item.findtext("pubDate") or item.findtext(f"{_DC_NS}date")
        published_at = normalize_datetime_str(raw_date) or parse_url_date(link)

        author = (
            strip_html(item.findtext(f"{_DC_NS}creator"))
            or strip_html(item.findtext("author"))
            or None
        )
        protocol = urlparse(link).scheme or "desconhecido"

        return CollectedItem(
            source=self.source_name,
            source_type=self.source_type,
            title=title,
            text=text,
            url=link,
            published_at=published_at,
            author=author,
            protocol=protocol,
        )

    def _parse_atom_entry(self, entry: ET.Element) -> Optional[CollectedItem]:
        title = strip_html(entry.findtext(f"{_ATOM_NS}title") or entry.findtext("title"))
        link = ""
        for tag in (entry.findall(f"{_ATOM_NS}link") + entry.findall("link")):
            rel = tag.get("rel", "alternate")
            href = (tag.get("href") or tag.text or "").strip()
            if href and rel in ("alternate", ""):
                link = href
                break
        if not title or not link:
            return None

        raw_desc = (
            entry.findtext(f"{_ATOM_NS}summary")
            or entry.findtext(f"{_ATOM_NS}content")
            or entry.findtext("summary")
            or entry.findtext("content")
        )
        text = strip_html(raw_desc)

        raw_date = (
            entry.findtext(f"{_ATOM_NS}published")
            or entry.findtext(f"{_ATOM_NS}updated")
            or entry.findtext("published")
            or entry.findtext("updated")
        )
        published_at = normalize_datetime_str(raw_date) or parse_url_date(link)

        author_tag = entry.find(f"{_ATOM_NS}author/{_ATOM_NS}name")
        if author_tag is None:
            author_tag = entry.find("author/name")
        author = strip_html(author_tag.text if author_tag is not None else None) or None
        protocol = urlparse(link).scheme or "desconhecido"

        return CollectedItem(
            source=self.source_name,
            source_type=self.source_type,
            title=title,
            text=text,
            url=link,
            published_at=published_at,
            author=author,
            protocol=protocol,
        )

    def collect(self, limit: int = 10) -> List[CollectedItem]:
        response = self._get(self.feed_url)
        root = ET.fromstring(response.content.strip())

        items: List[CollectedItem] = []
        rss_items = root.findall(".//item")
        if rss_items:
            for item in rss_items[:limit]:
                parsed = self._parse_rss_item(item)
                if parsed is not None:
                    items.append(parsed)
            return items

        atom_entries = root.findall(f".//{_ATOM_NS}entry") or root.findall(".//entry")
        for entry in atom_entries[:limit]:
            parsed = self._parse_atom_entry(entry)
            if parsed is not None:
                items.append(parsed)
        return items
