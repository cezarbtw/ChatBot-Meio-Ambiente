"""Interface base e sessão HTTP resiliente (timeout, User-Agent, retry/backoff)."""

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.collectors.models import CollectedItem, CollectorResult

DEFAULT_USER_AGENT = (
    "ChatBotMeioAmbiente/1.0 (+https://github.com/cezarbtw/ChatBot-Meio-Ambiente)"
)


def build_http_session(
    max_retries: int = 2,
    backoff_factor: float = 0.5,
    user_agent: str = DEFAULT_USER_AGENT,
) -> requests.Session:
    """Cria requests.Session com User-Agent explícito e política de retry/backoff."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        }
    )
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class BaseCollector(ABC):
    """Classe base para coletores de portais de notícias e redes sociais."""

    source_name: str = "base"
    source_type: str = "news"

    def __init__(
        self,
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.session = session or build_http_session(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    def _get(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response

    @abstractmethod
    def collect(self, limit: int = 10) -> List[CollectedItem]:
        """Executa a coleta e retorna lista padronizada de CollectedItem."""

    def safe_collect(self, limit: int = 10) -> CollectorResult:
        """Executa a coleta isolando exceções para não interromper outras fontes."""
        t0 = perf_counter()
        try:
            items = self.collect(limit=limit)
            return CollectorResult(
                source=self.source_name,
                source_type=self.source_type,
                success=True,
                items=items,
                duration_ms=(perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:
            return CollectorResult(
                source=self.source_name,
                source_type=self.source_type,
                success=False,
                items=[],
                error=str(exc),
                duration_ms=(perf_counter() - t0) * 1000.0,
            )
