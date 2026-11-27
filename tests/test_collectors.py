"""Testes offline para coletores multi-fonte, sessão HTTP e isolamento de falhas."""

import json

import requests

from src.collectors.base import BaseCollector, DEFAULT_USER_AGENT, build_http_session
from src.collectors.g1 import G1Collector
from src.collectors.manager import collect_from_sources
from src.collectors.models import CollectedItem
from src.collectors.rss import RssNewsCollector
from src.collectors.social import MastodonCollector, RedditCollector


class _FakeResponse:
    def __init__(self, body, status_code: int = 200):
        self.status_code = status_code
        if isinstance(body, (dict, list)):
            self._json = body
            self.content = json.dumps(body).encode("utf-8")
            self.text = json.dumps(body)
        elif isinstance(body, bytes):
            self._json = None
            self.content = body
            self.text = body.decode("utf-8")
        else:
            self._json = None
            self.text = body
            self.content = body.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json if self._json is not None else json.loads(self.text)


class _StubSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        handler = self.routes.get(url)
        if handler is None:
            raise requests.ConnectionError(f"Unmocked URL: {url}")
        if isinstance(handler, Exception):
            raise handler
        return handler


def test_build_http_session_configures_user_agent_and_retries():
    session = build_http_session(max_retries=3, backoff_factor=0.2)
    assert session.headers["User-Agent"] == DEFAULT_USER_AGENT
    adapter = session.get_adapter("https://g1.globo.com")
    assert adapter.max_retries.total == 3


def test_g1_collector_recovers_dates_for_non_date_urls():
    listing_html = """
    <html><body>
      <div class="feed-post">
        <a class="feed-post-body-title" href="https://g1.globo.com/meio-ambiente/noticia/2026/11/25/floresta.ghtml">Recuperação da floresta</a>
        <div class="feed-post-body-resumo">Área nativa cresce 20%.</div>
        <span class="feed-post-datetime">Hoje</span>
      </div>
      <div class="feed-post">
        <a class="feed-post-body-title" href="https://g1.globo.com/meio-ambiente/video/cop31-sem-acordo-123.ghtml">COP31 vídeo</a>
        <div class="feed-post-body-resumo">Resumo do evento.</div>
        <span class="feed-post-datetime">Há 2 dias</span>
      </div>
    </body></html>
    """
    video_html = """
    <html><head>
      <script type="application/ld+json">
        {"@type": "VideoObject", "uploadDate": "2026-11-23T18:00:00-03:00"}
      </script>
    </head></html>
    """
    session = _StubSession(
        {
            G1Collector.DEFAULT_URL: _FakeResponse(listing_html),
            "https://g1.globo.com/meio-ambiente/video/cop31-sem-acordo-123.ghtml": _FakeResponse(video_html),
        }
    )

    items = G1Collector(timeout=12.0, session=session).collect(limit=10)
    assert len(items) == 2
    assert items[0].published_date == "2026-11-25"
    assert items[1].published_date == "2026-11-23"
    assert all(call[2] == 12.0 for call in session.calls)


def test_rss_collector_parses_rss2_and_atom():
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel>
        <item>
          <title>Energia solar cresce no Nordeste</title>
          <link>https://agenciabrasil.ebc.com.br/meio-ambiente/noticia/2026-11/energia</link>
          <description>&lt;p&gt;Novo &lt;b&gt;parque&lt;/b&gt; entra em operação.&lt;/p&gt;</description>
          <pubDate>Wed, 25 Nov 2026 15:41:00 -0300</pubDate>
          <dc:creator>Repórter Ambiental</dc:creator>
        </item>
      </channel>
    </rss>
    """
    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Monitoramento do Cerrado</title>
        <link rel="alternate" href="https://portal.org/cerrado" />
        <summary>Dados mostram queda de queimadas.</summary>
        <published>2026-11-24T10:00:00Z</published>
        <author><name>Equipe Editorial</name></author>
      </entry>
    </feed>
    """
    rss_session = _StubSession({"https://rss.test/feed.xml": _FakeResponse(rss_xml)})
    rss_items = RssNewsCollector("agencia_brasil", "https://rss.test/feed.xml", session=rss_session).collect()
    assert len(rss_items) == 1
    assert rss_items[0].text == "Novo parque entra em operação."
    assert rss_items[0].author == "Repórter Ambiental"
    assert rss_items[0].published_date == "2026-11-25"

    atom_session = _StubSession({"https://atom.test/feed": _FakeResponse(atom_xml)})
    atom_items = RssNewsCollector("atom_portal", "https://atom.test/feed", session=atom_session).collect()
    assert len(atom_items) == 1
    assert atom_items[0].url == "https://portal.org/cerrado"
    assert atom_items[0].author == "Equipe Editorial"
    assert atom_items[0].published_date == "2026-11-24"


def test_mastodon_and_reddit_collectors():
    mastodon_payload = [
        {
            "url": "https://mastodon.social/@eco/1",
            "created_at": "2026-11-25T15:32:22.000Z",
            "language": "pt",
            "content": "<p>Mutirão de <b>reflorestamento</b> planta 500 mudas no cerrado.</p>",
            "account": {"acct": "eco@mastodon.social", "display_name": "Eco Brasil"},
            "card": {"title": "Mutirão no Cerrado", "description": "Ação comunitária."},
        },
        {
            "url": "https://mastodon.social/@eco/2",
            "created_at": "2026-11-25T16:00:00.000Z",
            "language": "pt",
            "content": '<p>Gestão ambiental avança <a href="https://exemplo.org/materia"><span class="invisible">https://</span><span>exemplo.org/materia</span></a></p>',
            "account": {"acct": "eco@mastodon.social", "display_name": "Eco Brasil"},
        },
    ]
    reddit_payload = {
        "data": {
            "children": [
                {
                    "data": {
                        "title": "Projeto recupera manguezais",
                        "selftext": "Comunidade local ampliou área protegida.",
                        "author": "usuario_verde",
                        "permalink": "/r/meioambiente/comments/abc/manguezais/",
                        "created_utc": 1795612800,
                    }
                }
            ]
        }
    }

    m_session = _StubSession(
        {"https://mastodon.social/api/v1/timelines/tag/meioambiente": _FakeResponse(mastodon_payload)}
    )
    m_items = MastodonCollector(hashtags=("meioambiente",), session=m_session).collect(limit=5)
    assert len(m_items) == 2
    assert m_items[0].source_type == "social"
    assert m_items[0].title == "Mutirão no Cerrado"
    assert m_items[0].author == "Eco Brasil (@eco@mastodon.social)"
    assert m_items[0].published_date == "2026-11-25"
    assert m_items[1].title == "Gestão ambiental avança"
    assert "https://exemplo.org/materia" in m_items[1].text

    r_session = _StubSession({RedditCollector.DEFAULT_URL: _FakeResponse(reddit_payload)})
    r_items = RedditCollector(session=r_session).collect(limit=5)
    assert len(r_items) == 1
    assert r_items[0].source == "reddit"
    assert r_items[0].source_type == "social"
    assert r_items[0].author == "usuario_verde"


def test_collect_from_sources_isolates_failures():
    class _FailingCollector(BaseCollector):
        source_name = "broken_source"
        source_type = "social"

        def collect(self, limit: int = 10):
            raise requests.Timeout("timeout simulado")

    class _OkCollector(BaseCollector):
        source_name = "ok_source"
        source_type = "news"

        def collect(self, limit: int = 10):
            return [
                CollectedItem(
                    source="ok_source",
                    source_type="news",
                    title="Parque protegido",
                    text="Nova unidade de conservação criada.",
                    url="https://ok.test/1",
                    published_at="2026-11-25",
                )
            ]

    items, reports = collect_from_sources(collectors=[_FailingCollector(), _OkCollector()])
    assert len(items) == 1
    assert items[0].source == "ok_source"
    assert len(reports) == 2
    assert reports[0].success is False
    assert "timeout simulado" in reports[0].error
    assert reports[1].success is True
