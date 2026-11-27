"""Testes offline de extração de datas (JSON-LD, meta tags, RSS, URL e relativo PT-BR)."""

from datetime import datetime, timezone

from src.collectors.date_parser import (
    extract_published_date,
    normalize_datetime_str,
    parse_relative_pt_date,
    parse_url_date,
)


def test_extract_date_from_json_ld_graph():
    html = """
    <html><head>
      <script type="application/ld+json">
        {"@context": "https://schema.org", "@graph": [
          {"@type": "Organization", "name": "Portal"},
          {"@type": "NewsArticle", "datePublished": "2026-11-24T14:20:00-03:00"}
        ]}
      </script>
    </head></html>
    """
    result = extract_published_date(article_html=html, url="https://exemplo.com/video/123")
    assert result == "2026-11-24T14:20:00-03:00"


def test_extract_date_from_meta_tag_when_url_has_no_date():
    html = """
    <html><head>
      <meta property="article:published_time" content="2026-11-25T11:30:00Z" />
    </head></html>
    """
    result = extract_published_date(article_html=html, url="https://g1.globo.com/video/abc")
    assert result == "2026-11-25T11:30:00+00:00"


def test_extract_date_falls_back_to_url_and_relative_pt():
    assert parse_url_date("https://g1.globo.com/meio-ambiente/noticia/2026/11/25/a.ghtml") == "2026-11-25"

    ref = datetime(2026, 11, 25, 16, 0, tzinfo=timezone.utc)
    assert parse_relative_pt_date("Hoje", reference_dt=ref) == "2026-11-25"
    assert parse_relative_pt_date("Ontem", reference_dt=ref) == "2026-11-24"
    assert parse_relative_pt_date("Há 2 dias", reference_dt=ref) == "2026-11-23"
    assert parse_relative_pt_date("Há 4 horas", reference_dt=ref) == "2026-11-25"
    assert parse_relative_pt_date("20/11/2026", reference_dt=ref) == "2026-11-20"

    assert (
        extract_published_date(
            article_html="<html><body>Sem meta</body></html>",
            url="https://g1.globo.com/meio-ambiente/video/cop31-123.ghtml",
            card_date_text="Há 2 dias",
            reference_dt=ref,
        )
        == "2026-11-23"
    )


def test_normalize_rfc2822_rss_date():
    normalized = normalize_datetime_str("Wed, 25 Nov 2026 15:41:00 -0300")
    assert normalized == "2026-11-25T15:41:00-03:00"
