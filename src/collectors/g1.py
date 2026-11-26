"""Coletor aprimorado para o portal G1 Meio Ambiente."""

from typing import List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from src.collectors.base import BaseCollector
from src.collectors.date_parser import extract_published_date, parse_url_date
from src.collectors.models import CollectedItem


class G1Collector(BaseCollector):
    """Coleta manchetes e resumos do G1 Meio Ambiente com extração multi-estratégia de data."""

    source_name = "g1"
    source_type = "news"
    DEFAULT_URL = "https://g1.globo.com/meio-ambiente/"

    def __init__(
        self,
        url: str = DEFAULT_URL,
        enrich_missing_dates: bool = True,
        always_fetch_article: bool = False,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        session: Optional[requests.Session] = None,
        inspector=None,
        inspect_network: Optional[bool] = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            session=session,
            inspector=inspector,
            inspect_network=inspect_network,
        )
        self.url = url
        self.enrich_missing_dates = enrich_missing_dates
        self.always_fetch_article = always_fetch_article

    def _fetch_article_html(self, article_url: str) -> Optional[str]:
        try:
            response = self._get(article_url)
            return response.text
        except requests.RequestException:
            return None

    def collect(self, limit: int = 10) -> List[CollectedItem]:
        response = self._get(self.url)
        listing_telemetry = self.last_telemetry
        soup = BeautifulSoup(response.content, "html.parser")

        items: List[CollectedItem] = []
        for artigo in soup.select(".feed-post")[:limit]:
            titulo_tag = artigo.select_one(".feed-post-body-title")
            link_tag = artigo.select_one("a")
            if not (titulo_tag and link_tag):
                continue

            raw_href = link_tag.get("href") or ""
            link = urljoin(self.url, raw_href) if raw_href else ""
            if not link:
                continue

            titulo = titulo_tag.get_text(strip=True)
            texto_tag = artigo.select_one(".feed-post-body-resumo")
            texto = texto_tag.get_text(strip=True) if texto_tag else ""

            dt_tag = artigo.select_one(".feed-post-datetime")
            card_dt_text = dt_tag.get_text(strip=True) if dt_tag else None

            article_html: Optional[str] = None
            item_telemetry = listing_telemetry
            url_has_date = parse_url_date(link) is not None
            if self.always_fetch_article or (self.enrich_missing_dates and not url_has_date):
                article_html = self._fetch_article_html(link)
                if article_html is not None and self.last_telemetry is not None:
                    item_telemetry = self.last_telemetry

            published_at = extract_published_date(
                article_html=article_html,
                url=link,
                card_date_text=card_dt_text,
            )

            protocol = (
                item_telemetry.summary_protocol()
                if item_telemetry is not None
                else (urlparse(link).scheme or "desconhecido")
            )
            items.append(
                CollectedItem(
                    source=self.source_name,
                    source_type=self.source_type,
                    title=titulo,
                    text=texto,
                    url=link,
                    published_at=published_at,
                    protocol=protocol,
                    network_telemetry=item_telemetry.to_dict() if item_telemetry else None,
                )
            )

        return items
