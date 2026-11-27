"""Testes offline do pipeline incremental e agendado de monitoramento."""

from src.collectors.base import BaseCollector
from src.collectors.models import CollectedItem
from src.nlp.classifier import PredictionResult
from src.pipeline import run_monitoring_cycle, run_scheduled_monitor
from src.storage import NewsRepository, create_db_engine, init_db


class _StubCollector(BaseCollector):
    source_name = "stub_portal"
    source_type = "news"

    def collect(self, limit: int = 10):
        return [
            CollectedItem(
                source="stub_portal",
                source_type="news",
                title="Parque protege nascentes",
                text="Iniciativa reduz poluição de rios.",
                url="https://stub.test/1",
                published_at="2026-11-26T09:00:00Z",
                protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
            ),
            CollectedItem(
                source="stub_portal",
                source_type="news",
                title="Queimadas ameaçam reserva",
                text="Focos de incêndios aumentam na região.",
                url="https://stub.test/2",
                published_at="2026-11-26T10:00:00Z",
                protocol="HTTP/1.1 | ALPN h2 | TLSv1.3",
            ),
        ][:limit]


class _StubClassifier:
    def predict_many_detailed(self, items):
        out = []
        for title, _text in items:
            lbl = "Boa" if "protege" in title.lower() else "Ruim"
            out.append(
                PredictionResult(
                    label=lbl,
                    confidence=0.9,
                    probabilities={"Boa": 0.9 if lbl == "Boa" else 0.05, "Irrelevante": 0.05, "Ruim": 0.9 if lbl == "Ruim" else 0.05},
                    is_ambiguous=False,
                )
            )
        return out


def test_monitoring_cycle_and_scheduled_runs_deduplicate(tmp_path):
    engine = create_db_engine(tmp_path / "pipeline.db")
    init_db(engine)
    repo = NewsRepository(engine)
    clf = _StubClassifier()

    sleep_calls = []
    history = run_scheduled_monitor(
        interval_seconds=10.0,
        max_runs=2,
        classifier=clf,
        collectors_factory=lambda: [_StubCollector()],
        repository=repo,
        sleep_fn=lambda sec: sleep_calls.append(sec),
    )

    assert len(history) == 2
    assert sleep_calls == [10.0]
    assert history[0].collected == 2
    assert history[0].inserted == 2
    assert history[0].deduplicated == 0
    assert history[1].collected == 2
    assert history[1].inserted == 0
    assert history[1].deduplicated == 2

    overview = repo.get_overview_stats()
    assert overview["total_items"] == 2
    assert len(overview["recent_runs"]) == 2

    single = run_monitoring_cycle(
        classifier=clf,
        collectors=[_StubCollector()],
        repository=repo,
    )
    assert single.inserted == 0
    assert single.deduplicated == 2
