"""Testes de integração offline da API FastAPI usando TestClient."""

import json

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.extractor import ExtractedArticle
from src.collectors.base import BaseCollector
from src.collectors.models import CollectedItem
from src.config import BENCHMARK_FILENAME, METRICS_FILENAME
from src.network.models import NetworkTelemetry
from src.nlp.classifier import PredictionResult


class _FakeEmbedder:
    model_name = "fake-bert-api"


class _FakeClassifier:
    embedder = _FakeEmbedder()

    def predict_detailed(self, title: str, text: str) -> PredictionResult:
        return self.predict_many_detailed([(title, text)])[0]

    def predict_many_detailed(self, items):
        results = []
        for title, text in items:
            low = f"{title} {text}".lower()
            if "temperatura" in low or "aquecimento" in low:
                results.append(
                    PredictionResult(
                        label="Irrelevante",
                        confidence=0.55,
                        probabilities={"Boa": 0.15, "Irrelevante": 0.55, "Ruim": 0.30},
                        is_ambiguous=True,
                        ambiguous_reason="confiança abaixo do limiar",
                        matched_ambiguous_terms=["temperatura"],
                    )
                )
            elif "incêndio" in low or "desmatamento" in low or "poluição" in low:
                results.append(
                    PredictionResult(
                        label="Ruim",
                        confidence=0.92,
                        probabilities={"Boa": 0.04, "Irrelevante": 0.04, "Ruim": 0.92},
                        is_ambiguous=False,
                    )
                )
            else:
                results.append(
                    PredictionResult(
                        label="Boa",
                        confidence=0.89,
                        probabilities={"Boa": 0.89, "Irrelevante": 0.06, "Ruim": 0.05},
                        is_ambiguous=False,
                    )
                )
        return results


class _FakeCollector(BaseCollector):
    source_name = "g1"
    source_type = "news"

    def collect(self, limit: int = 10):
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
            "dns_ms": 1.1,
            "tls_ms": 25.4,
            "ttfb_ms": 110.2,
        }
        return [
            CollectedItem(
                source="g1",
                source_type="news",
                title="Novo parque protege nascentes",
                text="Projeto ambiental amplia área verde sem desmatamento.",
                url="https://g1.globo.com/meio-ambiente/noticia/2026/11/26/parque.ghtml",
                published_at="2026-11-26T10:00:00Z",
                protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
                network_telemetry=tel,
            ),
            CollectedItem(
                source="g1",
                source_type="news",
                title="Frente fria muda temperatura no Sul",
                text="Previsão aponta variação de temperatura no fim de semana.",
                url="https://g1.globo.com/meio-ambiente/noticia/2026/11/25/frente.ghtml",
                published_at="2026-11-25T10:00:00Z",
                protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
                network_telemetry=tel,
            ),
        ][:limit]


def _make_app(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / METRICS_FILENAME).write_text(
        json.dumps({"accuracy": 0.903, "f1_macro": 0.903}),
        encoding="utf-8",
    )
    (artifacts_dir / BENCHMARK_FILENAME).write_text(
        json.dumps(
            {
                "total_configurations": 1,
                "best_configuration": {"name": "bert-large[cls,scaler]+mlp"},
                "results": [{"name": "bert-large[cls,scaler]+mlp", "cv_f1_macro_mean": 0.87}],
            }
        ),
        encoding="utf-8",
    )
    return create_app(
        db_path=tmp_path / "api_test.db",
        artifacts_dir=artifacts_dir,
        classifier=_FakeClassifier(),
        load_model_on_startup=False,
        collectors_factory=lambda: [_FakeCollector()],
    )


def test_health_docs_and_cors(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "fake-bert-api"
        assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"

        docs = client.get("/docs")
        assert docs.status_code == 200
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert "/api/classify" in schema.json()["paths"]


def test_classify_manual_and_url_and_validation(tmp_path, monkeypatch):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # Payload vazio -> 422
        empty_res = client.post("/api/classify", json={})
        assert empty_res.status_code == 422

        # Classificação manual com gravação no banco
        res = client.post(
            "/api/classify",
            json={
                "title": "Incêndio atinge reserva florestal",
                "text": "Brigadas combatem incêndio e poluição na mata.",
                "save_to_db": True,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["label"] == "Ruim"
        assert data["confidence"] == 0.92
        assert any(k["keyword"] == "poluição" for k in data["keywords"])
        assert data["saved_item_id"] is not None
        assert data["created"] is True

        # Reenvio idêntico deduplica
        res_dup = client.post(
            "/api/classify",
            json={
                "title": "Incêndio atinge reserva florestal",
                "text": "Brigadas combatem incêndio e poluição na mata.",
                "save_to_db": True,
            },
        )
        assert res_dup.status_code == 200
        assert res_dup.json()["created"] is False

        # Classificação por URL usando extração mockada
        def _fake_extract(url: str):
            return ExtractedArticle(
                url=url,
                title="Energia solar avança no Ceará",
                text="Nova usina limpa entra em operação no interior.",
                published_at="2026-11-26T12:00:00Z",
                telemetry=NetworkTelemetry(
                    url=url,
                    final_url=url,
                    scheme="https",
                    host="portal.exemplo.br",
                    port=443,
                    resolved_ips=["203.0.113.10"],
                    http_version="HTTP/1.1",
                    alpn_protocol="h2",
                    tls_version="TLSv1.3",
                    status_code=200,
                ),
            )

        monkeypatch.setattr("src.api.routes.fetch_and_extract_url", _fake_extract)
        url_res = client.post(
            "/api/classify",
            json={"url": "https://portal.exemplo.br/energia-solar"},
        )
        assert url_res.status_code == 200
        url_data = url_res.json()
        assert url_data["label"] == "Boa"
        assert url_data["extracted_title"] == "Energia solar avança no Ceará"
        assert url_data["protocol_summary"] == "HTTP/1.1 | ALPN h2 | TLSv1.3"


def test_collect_news_list_label_review_export_and_analytics(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # 1. Coleta sob demanda (1a vez insere 2; 2a vez deduplica 2)
        c1 = client.post("/api/collect", json={"sources": ["g1"], "limit_per_source": 5})
        assert c1.status_code == 200
        assert c1.json()["inserted"] == 2
        assert c1.json()["deduplicated"] == 0

        c2 = client.post("/api/collect", json={"sources": ["g1"], "limit_per_source": 5})
        assert c2.status_code == 200
        assert c2.json()["inserted"] == 0
        assert c2.json()["deduplicated"] == 2

        # Fonte inválida -> 422
        bad_src = client.post("/api/collect", json={"sources": ["fonte_inexistente"]})
        assert bad_src.status_code == 422

        # 2. Listagem filtrada e busca
        all_news = client.get("/api/news").json()
        assert all_news["total"] == 2

        ambiguous_news = client.get("/api/news?ambiguous_only=true").json()
        assert ambiguous_news["total"] == 1
        target_id = ambiguous_news["items"][0]["id"]

        # 3. Detalhe e correção humana de rótulo
        detail = client.get(f"/api/news/{target_id}")
        assert detail.status_code == 200

        patch_res = client.patch(
            f"/api/news/{target_id}/label",
            json={"label": "Boa", "reviewer": "revisor1", "notes": "Ajuste manual"},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["effective_label"] == "Boa"
        assert patch_res.json()["reviews_count"] == 1

        # 404 para ID inexistente e 422 para rótulo inválido
        assert client.patch("/api/news/99999/label", json={"label": "Boa"}).status_code == 404
        assert client.patch(f"/api/news/{target_id}/label", json={"label": "Outro"}).status_code == 422

        # 4. Exportação CSV
        csv_res = client.get("/api/news/export")
        assert csv_res.status_code == 200
        assert "text/csv" in csv_res.headers["content-type"]
        assert "Novo parque protege nascentes" in csv_res.text

        # 5. Analytics consolidado
        analytics = client.get("/api/analytics")
        assert analytics.status_code == 200
        payload = analytics.json()
        assert payload["overview"]["total_items"] == 2
        assert len(payload["daily_series"]) == 2
        assert payload["keyword_distribution"]["totals_by_keyword"]["desmatamento"] == 1
        assert payload["model_metrics"]["accuracy"] == 0.903
        assert payload["benchmark_summary"]["total_configurations"] == 1
        assert len(payload["sources_available"]) >= 5

        # 6. ChatBot endpoint (/api/chat)
        chat_trend = client.post(
            "/api/chat",
            json={"message": "Qual o panorama ambiental desta semana?"},
        )
        assert chat_trend.status_code == 200
        trend_data = chat_trend.json()
        assert trend_data["intent"] == "trend_summary"
        assert trend_data["mode"] == "local"
        assert len(trend_data["citations"]) >= 1

        chat_clf = client.post(
            "/api/chat",
            json={"message": "Classifique: Novo incêndio atinge reserva"},
        )
        assert chat_clf.status_code == 200
        assert chat_clf.json()["intent"] == "classify_text"
        assert chat_clf.json()["classification"]["label"] == "Ruim"

        # Mensagem vazia -> 422
        assert client.post("/api/chat", json={"message": "   "}).status_code == 422
