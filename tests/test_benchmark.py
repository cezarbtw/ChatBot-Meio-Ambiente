"""Testes para o módulo unificado de benchmark (TF-IDF e embeddings)."""

import json

import numpy as np
import pandas as pd

from src.nlp.benchmark import (
    export_benchmark_report,
    run_embedding_benchmark,
    run_tfidf_benchmark,
)


class _DummyEmbedder:
    def __init__(self, model_name="dummy", max_length=128, pooling="mean"):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling

    def embed_batch(self, texts, batch_size=16):
        rows = []
        for t in texts:
            low = t.lower()
            rows.append(
                [
                    1.0 if "boa" in low else 0.0,
                    1.0 if "ruim" in low else 0.0,
                    1.0 if "irrelevante" in low else 0.0,
                    len(low) / 50.0,
                ]
            )
        return np.array(rows, dtype=np.float32)


def _sample_df() -> pd.DataFrame:
    boa_topics = ["solar", "eólica", "reflorestamento", "reciclagem", "oceano", "fauna", "floresta", "energia"]
    ruim_topics = ["poluição", "incêndio", "desmatamento", "vazamento", "seca", "esgoto", "enchente", "fumaça"]
    irr_topics = ["festival", "cinema", "gastronomia", "turismo", "exposição", "museu", "livro", "teatro"]
    rows = []
    for boa_w, ruim_w, irr_w in zip(boa_topics, ruim_topics, irr_topics):
        rows.append({"titulo": f"Avanço em {boa_w}", "texto": f"boa recuperação ambiental com {boa_w}", "rotulo": "Boa"})
        rows.append({"titulo": f"Crise de {ruim_w}", "texto": f"ruim impacto severo por {ruim_w}", "rotulo": "Ruim"})
        rows.append({"titulo": f"Mostra de {irr_w}", "texto": f"irrelevante evento cultural de {irr_w}", "rotulo": "Irrelevante"})
    return pd.DataFrame(rows)


def test_tfidf_and_embedding_benchmark_and_export(tmp_path):
    df = _sample_df()
    tfidf_entries = run_tfidf_benchmark(df, cv_splits=3, max_features=50)
    assert len(tfidf_entries) == 10
    assert all(e.family == "tfidf" for e in tfidf_entries)

    emb_entries = run_embedding_benchmark(
        df,
        model_names=("dummy/model",),
        poolings=("cls", "mean"),
        classifier_types=("logreg",),
        scaler_options=(True,),
        cv_splits=3,
        embedder_factory=_DummyEmbedder,
    )
    assert len(emb_entries) == 2

    out_path = tmp_path / "benchmark_report.json"
    report = export_benchmark_report(tfidf_entries + emb_entries, output_path=out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["total_configurations"] == 12
    assert loaded["best_configuration"]["name"] == report["best_configuration"]["name"]
