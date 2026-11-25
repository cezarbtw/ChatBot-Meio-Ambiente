"""Treinamento, persistência leve e inferência do classificador MLP sobre embeddings."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_BERT_MODEL,
    DEFAULT_MAX_LENGTH,
    ENCODER_FILENAME,
    METADATA_FILENAME,
    MLP_FILENAME,
)
from src.nlp.dataset import validate_dataset
from src.nlp.embedder import BertEmbedder
from src.nlp.preprocessing import combine_title_and_text


class NewsClassifier:
    """Encapsula classificador MLP, LabelEncoder e extrator de embeddings."""

    def __init__(
        self,
        mlp: MLPClassifier,
        encoder: LabelEncoder,
        embedder: BertEmbedder,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mlp = mlp
        self.encoder = encoder
        self.embedder = embedder
        self.metadata = metadata or {}

    def predict(self, title: str, text: str) -> str:
        """Classifica uma única notícia em Boa, Ruim ou Irrelevante."""
        return self.predict_many([(title, text)])[0]

    def predict_many(self, items: Sequence[Tuple[str, str]]) -> List[str]:
        """Classifica uma lista de pares (titulo, texto)."""
        if not items:
            return []
        contents = [combine_title_and_text(title, text) for title, text in items]
        embeddings = self.embedder.embed_batch(contents)
        preds = self.mlp.predict(embeddings)
        return [str(label) for label in self.encoder.inverse_transform(preds)]

    def save(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        """Salva somente os artefatos leves do classificador e sua configuração."""
        target = Path(artifacts_dir)
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.mlp, target / MLP_FILENAME)
        joblib.dump(self.encoder, target / ENCODER_FILENAME)

        meta = {
            **self.metadata,
            "bert_model_name": getattr(self.embedder, "model_name", DEFAULT_BERT_MODEL),
            "max_length": getattr(self.embedder, "max_length", DEFAULT_MAX_LENGTH),
            "classes": [str(c) for c in self.encoder.classes_],
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        }
        (target / METADATA_FILENAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        artifacts_dir: Path = ARTIFACTS_DIR,
        embedder: Optional[BertEmbedder] = None,
    ) -> "NewsClassifier":
        """Carrega os artefatos leves e inicializa o embedder configurado."""
        target = Path(artifacts_dir)
        mlp = joblib.load(target / MLP_FILENAME)
        encoder = joblib.load(target / ENCODER_FILENAME)

        metadata_path = target / METADATA_FILENAME
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if embedder is None:
            embedder = BertEmbedder(
                model_name=metadata.get("bert_model_name", DEFAULT_BERT_MODEL),
                max_length=int(metadata.get("max_length", DEFAULT_MAX_LENGTH)),
            )

        return cls(mlp=mlp, encoder=encoder, embedder=embedder, metadata=metadata)


def train_classifier(
    df: pd.DataFrame,
    embedder: Optional[BertEmbedder] = None,
    artifacts_dir: Optional[Path] = ARTIFACTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[NewsClassifier, Dict[str, Any]]:
    """Valida o dataset, gera embeddings, treina o MLP e retorna modelo + métricas."""
    report = validate_dataset(df)
    if not report.is_valid:
        raise ValueError(f"Dataset inválido para treinamento: {'; '.join(report.issues)}")

    active_embedder = embedder or BertEmbedder()
    contents = [
        combine_title_and_text(row.titulo, row.texto)
        for row in df.itertuples(index=False)
    ]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df["rotulo"])

    x_train, x_test, y_train, y_test = train_test_split(
        contents, y_encoded, test_size=test_size, random_state=random_state
    )

    x_train_emb = active_embedder.embed_batch(x_train)
    x_test_emb = active_embedder.embed_batch(x_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=random_state,
        early_stopping=True,
    )
    mlp.fit(x_train_emb, y_train)
    y_pred = mlp.predict(x_test_emb)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, target_names=encoder.classes_)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "classes": [str(c) for c in encoder.classes_],
    }

    classifier = NewsClassifier(
        mlp=mlp,
        encoder=encoder,
        embedder=active_embedder,
        metadata={
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "test_accuracy": acc,
        },
    )

    if artifacts_dir is not None:
        classifier.save(artifacts_dir)

    return classifier, metrics
