"""Testes offline do motor conversacional (NLU, SQLite, BERT+MLP e LLM opcional)."""

import json
from types import SimpleNamespace

import requests

from src.chatbot import ChatBotService, OptionalLlmClient, parse_user_message
from src.collectors.models import CollectedItem
from src.config import METRICS_FILENAME
from src.nlp.classifier import PredictionResult
from src.storage import NewsRepository, create_db_engine, init_db


class _StubClassifier:
    def predict_detailed(self, title: str, text: str) -> PredictionResult:
        low = f"{title} {text}".lower()
        if "poluição" in low or "desmatamento" in low:
            return PredictionResult(
                label="Ruim",
                confidence=0.91,
                probabilities={"Boa": 0.05, "Irrelevante": 0.04, "Ruim": 0.91},
                is_ambiguous=False,
            )
        return PredictionResult(
            label="Boa",
            confidence=0.88,
            probabilities={"Boa": 0.88, "Irrelevante": 0.07, "Ruim": 0.05},
            is_ambiguous=False,
        )


def _seeded_repo(tmp_path) -> NewsRepository:
    engine = create_db_engine(tmp_path / "chat_test.db")
    init_db(engine)
    repo = NewsRepository(engine)

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
        "dns_ms": 1.0,
        "tls_ms": 20.0,
        "ttfb_ms": 100.0,
    }
    repo.save_collection_batch(
        sources_requested=["g1", "oeco"],
        items_with_predictions=[
            (
                CollectedItem(
                    source="g1",
                    source_type="news",
                    title="Queimadas e desmatamento preocupam especialistas",
                    text="Focos de incêndios e desmatamento aumentam na região.",
                    url="https://g1.globo.com/meio-ambiente/1",
                    published_at="2026-11-26T10:00:00Z",
                    protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
                    network_telemetry=tel,
                ),
                PredictionResult(
                    label="Ruim",
                    confidence=0.93,
                    probabilities={"Boa": 0.03, "Irrelevante": 0.04, "Ruim": 0.93},
                    is_ambiguous=False,
                ),
            ),
            (
                CollectedItem(
                    source="oeco",
                    source_type="news",
                    title="Usina solar amplia energia limpa",
                    text="Projeto reduz emissões no Nordeste.",
                    url="https://oeco.org.br/2",
                    published_at="2026-11-25T10:00:00Z",
                    protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
                    network_telemetry=tel,
                ),
                PredictionResult(
                    label="Boa",
                    confidence=0.89,
                    probabilities={"Boa": 0.89, "Irrelevante": 0.06, "Ruim": 0.05},
                    is_ambiguous=False,
                ),
            ),
        ],
    )
    return repo


def test_parse_user_message_extracts_intents_and_filters():
    c = parse_user_message("Classifique: Novo filtro reduz poluição em rios")
    assert c.intent == "classify_text"
    assert c.text_to_classify == "Novo filtro reduz poluição em rios"

    p = parse_user_message("Quais protocolos TLS e ALPN foram identificados?")
    assert p.intent == "protocol_summary"

    m = parse_user_message("Qual a acurácia e F1 do modelo BERT?")
    assert m.intent == "model_performance"

    t = parse_user_message("Qual o panorama ambiental desta semana?")
    assert t.intent == "trend_summary"
    assert t.date_from is not None

    s = parse_user_message("Mostre as piores notícias ruins sobre desmatamento no G1")
    assert s.intent == "news_search"
    assert s.label_filter == "Ruim"
    assert s.keyword_filter == "desmatamento"
    assert s.source_filter == "g1"


def test_chatbot_service_handles_all_intents_locally(tmp_path):
    repo = _seeded_repo(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / METRICS_FILENAME).write_text(
        json.dumps(
            {
                "model_config": {
                    "bert_model_name": "bert-test",
                    "pooling": "cls",
                    "use_scaler": True,
                },
                "holdout": {"accuracy": 0.903, "f1_macro": 0.903},
                "cross_validation": {"n_splits": 5, "f1_macro_mean": 0.869},
            }
        ),
        encoding="utf-8",
    )

    service = ChatBotService(
        repository=repo,
        classifier=_StubClassifier(),
        artifacts_dir=artifacts_dir,
        llm_client=OptionalLlmClient(api_key=""),
    )

    trend = service.reply("Qual o panorama ambiental?")
    assert trend.intent == "trend_summary"
    assert trend.mode == "local"
    assert len(trend.citations) == 2

    search = service.reply("Mostre notícias ruins sobre desmatamento")
    assert search.intent == "news_search"
    assert len(search.citations) == 1
    assert search.citations[0].effective_label == "Ruim"

    proto = service.reply("Resumo dos protocolos de rede")
    assert proto.intent == "protocol_summary"
    assert "TLSv1.3" in proto.answer

    perf = service.reply("Qual o desempenho do modelo?")
    assert perf.intent == "model_performance"
    assert "90.3%" in perf.answer

    clf = service.reply("Classifique: Desmatamento cresce em reserva")
    assert clf.intent == "classify_text"
    assert clf.classification is not None
    assert clf.classification.label == "Ruim"
    assert clf.classification.keywords[0]["keyword"] == "desmatamento"


def test_optional_llm_connector_and_graceful_fallback(tmp_path, monkeypatch):
    repo = _seeded_repo(tmp_path)

    def _ok_post(*args, **kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"content": "Resumo gerado via LLM com base no SQLite."}}]
            },
        )

    monkeypatch.setattr(requests, "post", _ok_post)
    service = ChatBotService(
        repository=repo,
        classifier=_StubClassifier(),
        llm_client=OptionalLlmClient(api_key="sk-test-key"),
    )
    rag_reply = service.reply("Qual o panorama desta semana?")
    assert rag_reply.mode == "llm_rag"
    assert "Resumo gerado via LLM" in rag_reply.answer

    def _fail_post(*args, **kwargs):
        raise requests.Timeout("timeout simulado")

    monkeypatch.setattr(requests, "post", _fail_post)
    fallback_reply = service.reply("Qual o panorama desta semana?")
    assert fallback_reply.mode == "local"
    assert "itens monitorados" in fallback_reply.answer
