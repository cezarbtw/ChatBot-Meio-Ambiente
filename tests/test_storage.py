"""Testes offline de persistência SQLite, deduplicação, curadoria e analytics."""

import pytest

from src.collectors.models import CollectedItem, CollectorResult
from src.nlp.classifier import PredictionResult
from src.storage import (
    NewsRepository,
    canonicalize_url,
    compute_content_hash,
    create_db_engine,
    init_db,
)


def _make_repo(tmp_path) -> NewsRepository:
    engine = create_db_engine(tmp_path / "test_monitor.db")
    init_db(engine)
    return NewsRepository(engine)


def _sample_prediction(label: str = "Boa", conf: float = 0.88, ambiguous: bool = False):
    probs = {"Boa": 0.06, "Irrelevante": 0.06, "Ruim": 0.06}
    probs[label] = conf
    return PredictionResult(
        label=label,
        confidence=conf,
        probabilities=probs,
        is_ambiguous=ambiguous,
        ambiguous_reason="baixa confiança" if ambiguous else None,
    )


def test_canonicalize_url_and_content_hash():
    u1 = "https://G1.globo.com:443/meio-ambiente/noticia/a.ghtml/?utm_source=x&b=2&a=1#topo"
    u2 = "https://g1.globo.com/meio-ambiente/noticia/a.ghtml?a=1&b=2&fbclid=abc"
    assert canonicalize_url(u1) == canonicalize_url(u2)
    assert canonicalize_url(u1) == "https://g1.globo.com/meio-ambiente/noticia/a.ghtml?a=1&b=2"

    h1 = compute_content_hash("  Nova Usina Solar ", "<p>Abastece 50 mil casas</p>")
    h2 = compute_content_hash("nova usina solar", "abastece 50 mil casas https://x.co")
    assert h1 == h2


def test_save_collection_batch_deduplicates_across_runs(tmp_path):
    repo = _make_repo(tmp_path)

    tel = {
        "url": "https://g1.globo.com/meio-ambiente/",
        "final_url": "https://g1.globo.com/meio-ambiente/",
        "scheme": "https",
        "host": "g1.globo.com",
        "port": 443,
        "resolved_ips": ["186.192.83.7"],
        "http_version": "HTTP/1.1",
        "alpn_protocol": "h2",
        "tls_version": "TLSv1.3",
        "cipher_suite": "TLS_AES_256_GCM_SHA384",
        "status_code": 200,
        "dns_ms": 1.2,
        "tls_ms": 30.5,
        "ttfb_ms": 120.0,
    }
    item1 = CollectedItem(
        source="g1",
        source_type="news",
        title="Incêndios atingem área protegida",
        text="Combate aos incêndios mobiliza brigadas contra poluição.",
        url="https://g1.globo.com/meio-ambiente/noticia/2026/11/25/a.ghtml?utm_source=rss",
        published_at="2026-11-25T12:00:00Z",
        protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
        network_telemetry=tel,
    )
    item2 = CollectedItem(
        source="agencia_brasil",
        source_type="news",
        title="Reflorestamento amplia cobertura nativa",
        text="Programa reduz desmatamento na Amazônia.",
        url="https://agenciabrasil.ebc.com.br/meio-ambiente/b",
        published_at="2026-11-24T10:00:00Z",
        protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
        network_telemetry=tel,
    )
    report = CollectorResult(
        source="g1", source_type="news", success=True, items=[item1], telemetry_records=[tel]
    )

    first = repo.save_collection_batch(
        sources_requested=["g1", "agencia_brasil"],
        items_with_predictions=[
            (item1, _sample_prediction("Ruim", 0.91)),
            (item2, _sample_prediction("Boa", 0.89)),
        ],
        reports=[report],
    )
    assert first.items_inserted == 2
    assert first.items_deduplicated == 0

    # Reexecução: URL equivalente sem utm_source e conteúdo duplicado com URL diferente
    dup_by_url = CollectedItem(
        source="g1",
        source_type="news",
        title="Título alterado mas mesma URL canônica",
        text="Outro resumo.",
        url="https://g1.globo.com/meio-ambiente/noticia/2026/11/25/a.ghtml",
        published_at="2026-11-25",
    )
    dup_by_hash = CollectedItem(
        source="mastodon",
        source_type="social",
        title="Reflorestamento amplia cobertura nativa",
        text="Programa reduz desmatamento na Amazônia.",
        url="https://mastodon.social/@eco/999",
        published_at="2026-11-24",
    )
    second = repo.save_collection_batch(
        sources_requested=["g1", "mastodon"],
        items_with_predictions=[
            (dup_by_url, _sample_prediction("Ruim", 0.90)),
            (dup_by_hash, _sample_prediction("Boa", 0.85)),
        ],
    )
    assert second.items_inserted == 0
    assert second.items_deduplicated == 2


def test_filtering_label_review_daily_series_keywords_and_protocols(tmp_path):
    repo = _make_repo(tmp_path)

    tel = {
        "url": "https://g1.globo.com/meio-ambiente/",
        "final_url": "https://g1.globo.com/meio-ambiente/",
        "scheme": "https",
        "host": "g1.globo.com",
        "port": 443,
        "resolved_ips": ["186.192.83.7"],
        "http_version": "HTTP/1.1",
        "alpn_protocol": "h2",
        "tls_version": "TLSv1.3",
        "status_code": 200,
        "dns_ms": 2.0,
        "tls_ms": 40.0,
        "ttfb_ms": 150.0,
    }
    items = [
        (
            CollectedItem(
                source="g1",
                source_type="news",
                title="Temperatura sobe no verão",
                text="Previsão indica aquecimento leve.",
                url="https://g1.globo.com/1",
                published_at="2026-11-24T10:00:00Z",
                protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
                network_telemetry=tel,
            ),
            _sample_prediction("Ruim", 0.55, ambiguous=True),
        ),
        (
            CollectedItem(
                source="oeco",
                source_type="news",
                title="Combate ao desmatamento avança",
                text="Redução de incêndios e desmatamento em reservas.",
                url="https://oeco.org.br/2",
                published_at="2026-11-25T10:00:00Z",
                protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
                network_telemetry=tel,
            ),
            _sample_prediction("Boa", 0.92),
        ),
    ]
    repo.save_collection_batch(["g1", "oeco"], items)

    filtered, total = repo.list_news(ambiguous_only=True)
    assert total == 1
    assert filtered[0]["source"] == "g1"
    target_id = filtered[0]["id"]

    updated = repo.add_label_review(
        target_id, new_label="Irrelevante", reviewer="analyst", notes="Previsão meteorológica"
    )
    assert updated["predicted_label"] == "Ruim"
    assert updated["effective_label"] == "Irrelevante"
    assert updated["reviews_count"] == 1

    series = repo.get_daily_series()
    assert len(series) == 2
    assert series[0] == {"date": "2026-11-24", "Boa": 0, "Ruim": 0, "Irrelevante": 1, "total": 1}
    assert series[1] == {"date": "2026-11-25", "Boa": 1, "Ruim": 0, "Irrelevante": 0, "total": 1}

    kw = repo.get_keyword_distribution()
    assert kw["totals_by_keyword"]["desmatamento"] == 2
    assert kw["percentages"]["Boa"]["desmatamento"] == 100.0
    assert kw["percentages"]["Irrelevante"]["temperatura"] == 100.0

    proto = repo.get_protocol_stats()
    assert proto["total_records"] == 2
    assert proto["alpn_protocols"] == {"h2": 2}
    assert proto["tls_versions"] == {"TLSv1.3": 2}
    assert proto["avg_dns_ms"] == 2.0

    overview = repo.get_overview_stats()
    assert overview["total_items"] == 2
    assert overview["by_label"] == {"Boa": 1, "Irrelevante": 1, "Ruim": 0}

    with pytest.raises(ValueError):
        repo.add_label_review(target_id, "Invalido")
