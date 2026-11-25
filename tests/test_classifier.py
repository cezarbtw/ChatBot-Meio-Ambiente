"""Testes do pipeline de treino, persistência leve e inferência com embedder fake."""

import json

import numpy as np
import pandas as pd

from src.config import ENCODER_FILENAME, METADATA_FILENAME, MLP_FILENAME
from src.nlp.classifier import NewsClassifier, train_classifier


class FakeEmbedder:
    """Embedder determinístico leve para testar treino e inferência sem baixar BERT."""

    model_name = "fake-bert-test"
    max_length = 64

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
    for i in range(6):
        rows.append({"titulo": f"Usina solar {i}", "texto": "energia boa limpa", "rotulo": "Boa"})
        rows.append({"titulo": f"Crise poluição {i}", "texto": "notícia ruim ar", "rotulo": "Ruim"})
        rows.append({"titulo": f"Festival {i}", "texto": "evento irrelevante", "rotulo": "Irrelevante"})
    return pd.DataFrame(rows)


def test_train_save_load_and_predict_without_bert_weights(tmp_path):
    df = _sample_dataframe()
    embedder = FakeEmbedder()

    classifier, metrics = train_classifier(
        df, embedder=embedder, artifacts_dir=tmp_path, test_size=0.25, random_state=42
    )

    assert metrics["classes"] == ["Boa", "Irrelevante", "Ruim"]
    assert (tmp_path / MLP_FILENAME).exists()
    assert (tmp_path / ENCODER_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()

    metadata = json.loads((tmp_path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["bert_model_name"] == "fake-bert-test"
    assert metadata["classes"] == ["Boa", "Irrelevante", "Ruim"]

    loaded = NewsClassifier.load(artifacts_dir=tmp_path, embedder=embedder)
    pred = loaded.predict("Nova usina solar", "Gera energia boa para cidade")
    assert pred in {"Boa", "Irrelevante", "Ruim"}
