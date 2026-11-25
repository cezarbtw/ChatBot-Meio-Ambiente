"""Testes do pipeline de treino estratificado, CV, calibração e inferência detalhada."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ENCODER_FILENAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    MLP_FILENAME,
)
from src.nlp.classifier import (
    NewsClassifier,
    build_classifier_pipeline,
    train_classifier,
)


class FakeEmbedder:
    """Embedder determinístico leve para testar treino e inferência sem baixar BERT."""

    model_name = "fake-bert-test"
    max_length = 256
    pooling = "mean"

    def embed_batch(self, texts, batch_size=16):
        vectors = []
        for text in texts:
            low = text.lower()
            vec = np.array(
                [
                    1.0 if "boa" in low or "solar" in low else 0.0,
                    1.0 if "ruim" in low or "poluição" in low else 0.0,
                    1.0 if "irrelevante" in low or "festival" in low else 0.0,
                    len(low) / 100.0,
                ],
                dtype=np.float32,
            )
            vectors.append(vec)
        return np.vstack(vectors)


def _sample_dataframe() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append({"titulo": f"Usina solar {i}", "texto": "energia boa limpa", "rotulo": "Boa"})
        rows.append({"titulo": f"Crise poluição {i}", "texto": "notícia ruim ar", "rotulo": "Ruim"})
        rows.append({"titulo": f"Festival {i}", "texto": "evento irrelevante", "rotulo": "Irrelevante"})
    return pd.DataFrame(rows)


def test_train_save_load_and_predict_detailed(tmp_path):
    df = _sample_dataframe()
    embedder = FakeEmbedder()

    classifier, metrics = train_classifier(
        df,
        embedder=embedder,
        artifacts_dir=tmp_path,
        test_size=0.25,
        cv_splits=3,
        random_state=42,
    )

    assert metrics["classes"] == ["Boa", "Irrelevante", "Ruim"]
    assert metrics["cross_validation"]["n_splits"] == 3
    assert "polarization_errors" in metrics["holdout"]
    assert (tmp_path / MLP_FILENAME).exists()
    assert (tmp_path / ENCODER_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    assert (tmp_path / METRICS_FILENAME).exists()

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["bert_model_name"] == "fake-bert-test"
    assert metadata["pooling"] == "mean"

    report = json.loads((tmp_path / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert "holdout" in report and "cross_validation" in report

    loaded = NewsClassifier.load(artifacts_dir=tmp_path, embedder=embedder)
    detailed = loaded.predict_detailed("Nova usina solar", "Gera energia boa para cidade")
    assert detailed.label in {"Boa", "Irrelevante", "Ruim"}
    assert pytest.approx(sum(detailed.probabilities.values()), rel=1e-5) == 1.0
    assert 0.0 <= detailed.confidence <= 1.0


def test_predict_detailed_flags_ambiguity_and_technical_terms(tmp_path):
    df = _sample_dataframe()
    embedder = FakeEmbedder()
    classifier, _ = train_classifier(
        df,
        embedder=embedder,
        artifacts_dir=tmp_path,
        classifier_type="mlp_calibrated",
        cv_splits=3,
        confidence_threshold=0.80,
        margin_threshold=0.25,
    )

    class _BorderlineModel:
        def predict_proba(self, x):
            return np.tile(np.array([[0.46, 0.14, 0.40]]), (len(x), 1))

    borderline_clf = NewsClassifier(
        mlp=_BorderlineModel(),
        encoder=classifier.encoder,
        embedder=embedder,
        confidence_threshold=0.80,
        margin_threshold=0.25,
    )

    result = borderline_clf.predict_detailed(
        "Previsão aponta variação de temperatura",
        "Tendência de aquecimento moderado no litoral.",
    )
    assert result.is_ambiguous is True
    assert result.label == "Boa"
    assert result.confidence == pytest.approx(0.46)
    assert result.ambiguous_reason is not None
    assert "confiança abaixo do limiar" in result.ambiguous_reason
    assert "margem estreita" in result.ambiguous_reason
    assert "temperatura" in result.matched_ambiguous_terms
    assert "aquecimento" in result.matched_ambiguous_terms


@pytest.mark.parametrize("clf_type", ["mlp", "mlp_calibrated", "logreg", "svc"])
def test_build_classifier_pipeline_supports_all_variants(clf_type):
    pipe = build_classifier_pipeline(classifier_type=clf_type, use_scaler=True)
    assert "scaler" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_build_classifier_pipeline_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_classifier_pipeline(classifier_type="unknown")
