"""Coletores de redes sociais e comunidades públicas (Mastodon e Reddit)."""

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests

from src.collectors.base import BaseCollector
from src.collectors.date_parser import normalize_datetime_str
from src.collectors.models import CollectedItem


def _extract_mastodon_text(raw_html: Optional[str]) -> str:
    """Extrai texto de HTML do Mastodon preservando URLs inteiras entre spans."""
    if not raw_html or not isinstance(raw_html, str):
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["p", "br"]):
        tag.append(" ")
    text = soup.get_text("", strip=False)
    return re.sub(r"\s+", " ", text).strip()


def _derive_social_title(text: str, card_title: Optional[str] = None, max_len: int = 110) -> str:
    if card_title and card_title.strip():
        return card_title.strip()
    clean = re.sub(r"https?://\S+", "", text).strip() or text.strip()
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) <= max_len:
        return clean
    truncated = clean[: max_len - 3].rsplit(" ", 1)[0].strip()
    return f"{truncated}..." if truncated else f"{clean[: max_len - 3]}..."


class MastodonCollector(BaseCollector):
    """Coleta publicações públicas por hashtags ambientais na API aberta do Mastodon."""

    source_name = "mastodon"
    source_type = "social"
    DEFAULT_INSTANCE = "https://mastodon.social"
    DEFAULT_HASHTAGS = ("meioambiente", "amazonia", "desmatamento", "sustentabilidade")

    def __init__(
        self,
        instance_url: str = DEFAULT_INSTANCE,
        hashtags: Sequence[str] = DEFAULT_HASHTAGS,
        languages: Optional[Sequence[str]] = None,
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
        self.instance_url = instance_url.rstrip("/")
        self.hashtags = tuple(hashtags)
        self.languages = tuple(lang.lower() for lang in languages) if languages else None

    def _status_to_item(self, status: Dict[str, Any]) -> Optional[CollectedItem]:
        url = (status.get("url") or status.get("uri") or "").strip()
        if not url:
            return None

        if self.languages is not None:
            lang = (status.get("language") or "").lower()
            if lang and lang not in self.languages:
                return None

        body_text = _extract_mastodon_text(status.get("content"))
        card = status.get("card") or {}
        card_title = (card.get("title") or "").strip()
        card_desc = (card.get("description") or "").strip()

        combined_text = body_text
        if card_desc and card_desc not in body_text:
            combined_text = f"{body_text} {card_desc}".strip() if body_text else card_desc

        if not combined_text and not card_title:
            return None

        title = _derive_social_title(combined_text or card_title, card_title=card_title)
        account = status.get("account") or {}
        acct = (account.get("acct") or "").strip()
        display_name = (account.get("display_name") or "").strip()
        author = f"{display_name} (@{acct})" if display_name and acct else (f"@{acct}" if acct else None)

        published_at = normalize_datetime_str(status.get("created_at"))
        protocol = urlparse(url).scheme or "desconhecido"

        return CollectedItem(
            source=self.source_name,
            source_type=self.source_type,
            title=title,
            text=combined_text or title,
            url=url,
            published_at=published_at,
            author=author,
            protocol=protocol,
        )

    def collect(self, limit: int = 10) -> List[CollectedItem]:
        seen_urls = set()
        items: List[CollectedItem] = []
        last_error: Optional[Exception] = None
        any_tag_succeeded = False

        per_tag_limit = max(limit, 10)
        for tag in self.hashtags:
            endpoint = f"{self.instance_url}/api/v1/timelines/tag/{tag}"
            try:
                response = self._get(endpoint, params={"limit": per_tag_limit})
                statuses = response.json()
                any_tag_succeeded = True
            except Exception as exc:
                last_error = exc
                continue

            if not isinstance(statuses, list):
                continue

            for status in statuses:
                item = self._status_to_item(status)
                if item is None or item.url in seen_urls:
                    continue
                if self.last_telemetry is not None:
                    item.protocol = self.last_telemetry.summary_protocol()
                    item.network_telemetry = self.last_telemetry.to_dict()
                seen_urls.add(item.url)
                items.append(item)
                if len(items) >= limit:
                    return items

        if not any_tag_succeeded and last_error is not None:
            raise last_error
        return items


class RedditCollector(BaseCollector):
    """Coleta publicações públicas recentes via busca JSON do Reddit."""

    source_name = "reddit"
    source_type = "social"
    DEFAULT_URL = "https://www.reddit.com/search.json"

    def __init__(
        self,
        query: str = '"meio ambiente" OR amazonia OR desmatamento',
        search_url: str = DEFAULT_URL,
        timeout: float = 15.0,
        max_retries: int = 1,
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
        self.query = query
        self.search_url = search_url

    def collect(self, limit: int = 10) -> List[CollectedItem]:
        response = self._get(
            self.search_url,
            params={"q": self.query, "sort": "new", "limit": limit},
        )
        telemetry = self.last_telemetry
        payload = response.json()
        children = payload.get("data", {}).get("children", [])

        items: List[CollectedItem] = []
        for child in children[:limit]:
            post = child.get("data") or {}
            title = (post.get("title") or "").strip()
            permalink = (post.get("permalink") or "").strip()
            url = f"https://www.reddit.com{permalink}" if permalink else (post.get("url") or "").strip()
            if not title or not url:
                continue

            text = (post.get("selftext") or "").strip() or title
            author = (post.get("author") or "").strip() or None
            created_utc = post.get("created_utc")
            published_at: Optional[str] = None
            if isinstance(created_utc, (int, float)):
                published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

            protocol = (
                telemetry.summary_protocol()
                if telemetry is not None
                else (urlparse(url).scheme or "desconhecido")
            )
            items.append(
                CollectedItem(
                    source=self.source_name,
                    source_type=self.source_type,
                    title=title,
                    text=text,
                    url=url,
                    published_at=published_at,
                    author=author,
                    protocol=protocol,
                    network_telemetry=telemetry.to_dict() if telemetry else None,
                )
            )
        return items
