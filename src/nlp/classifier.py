"""Treinamento estratificado, validação cruzada, calibração de confiança e inferência."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import (
    AMBIGUOUS_TERMS,
    ARTIFACTS_DIR,
    DEFAULT_BERT_MODEL,
    DEFAULT_CLASSIFIER_TYPE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CV_SPLITS,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MAX_LENGTH,
    DEFAULT_POOLING,
    ENCODER_FILENAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    MLP_FILENAME,
    VALID_CLASSIFIER_TYPES,
)
from src.nlp.dataset import validate_dataset
from src.nlp.embedder import BertEmbedder
from src.nlp.preprocessing import combine_title_and_text


@dataclass
class PredictionResult:
    """Resultado detalhado de classificação com probabilidades e sinalização de ambiguidade."""

    label: str
    confidence: float
    probabilities: Dict[str, float]
    is_ambiguous: bool
    ambiguous_reason: Optional[str] = None
    matched_ambiguous_terms: List[str] = field(default_factory=list)


def build_classifier_pipeline(
    classifier_type: str = DEFAULT_CLASSIFIER_TYPE,
    use_scaler: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Constrói pipeline sklearn com normalização opcional e estimador probabilístico."""
    if classifier_type not in VALID_CLASSIFIER_TYPES:
        raise ValueError(
            f"Classificador inválido '{classifier_type}'. "
            f"Opções aceitas: {VALID_CLASSIFIER_TYPES}"
        )

    if classifier_type == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        )
    elif classifier_type == "mlp_calibrated":
        base_mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        )
        estimator = CalibratedClassifierCV(base_mlp, method="sigmoid", cv=3)
    elif classifier_type == "logreg":
        estimator = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        )
    else:
        estimator = SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=random_state,
        )

    steps: List[Tuple[str, Any]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _detect_ambiguous_terms(text: str) -> List[str]:
    lowered = text.lower()
    return [term for term in AMBIGUOUS_TERMS if term in lowered]


def _compute_polarization_errors(
    cm: np.ndarray, classes: Sequence[str]
) -> Dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    boa_as_ruim = 0
    boa_total = 0
    ruim_as_boa = 0
    ruim_total = 0

    if "Boa" in class_to_idx and "Ruim" in class_to_idx:
        boa_idx = class_to_idx["Boa"]
        ruim_idx = class_to_idx["Ruim"]
        boa_total = int(cm[boa_idx].sum())
        ruim_total = int(cm[ruim_idx].sum())
        boa_as_ruim = int(cm[boa_idx, ruim_idx])
        ruim_as_boa = int(cm[ruim_idx, boa_idx])

    return {
        "boa_classified_as_ruim_count": boa_as_ruim,
        "boa_classified_as_ruim_rate": (
            float(boa_as_ruim / boa_total) if boa_total > 0 else 0.0
        ),
        "ruim_classified_as_boa_count": ruim_as_boa,
        "ruim_classified_as_boa_rate": (
            float(ruim_as_boa / ruim_total) if ruim_total > 0 else 0.0
        ),
    }


def _evaluate_cv(
    pipeline: Pipeline,
    embeddings: np.ndarray,
    y_encoded: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Dict[str, Any]:
    min_class_count = int(np.bincount(y_encoded).min())
    effective_splits = min(n_splits, min_class_count)
    if effective_splits < 2:
        return {"n_splits": 0, "skipped": True}

    cv = StratifiedKFold(
        n_splits=effective_splits, shuffle=True, random_state=random_state
    )
    scores = cross_validate(
        pipeline,
        embeddings,
        y_encoded,
        cv=cv,
        scoring=("accuracy", "f1_macro"),
        return_train_score=False,
    )
    acc_scores = [float(x) for x in scores["test_accuracy"]]
    f1_scores = [float(x) for x in scores["test_f1_macro"]]
    return {
        "n_splits": effective_splits,
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "accuracy_folds": acc_scores,
        "f1_macro_folds": f1_scores,
    }


class NewsClassifier:
    """Encapsula classificador probabilístico, LabelEncoder e extrator de embeddings."""

    def __init__(
        self,
        mlp: Any,
        encoder: LabelEncoder,
        embedder: BertEmbedder,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    ) -> None:
        self.mlp = mlp
        self.encoder = encoder
        self.embedder = embedder
        self.metadata = metadata or {}
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def _build_prediction_result(
        self, content: str, row_probs: np.ndarray
    ) -> PredictionResult:
        classes = [str(c) for c in self.encoder.classes_]
        probabilities = {
            cls_name: float(prob) for cls_name, prob in zip(classes, row_probs)
        }
        ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_conf = ordered[0]
        second_conf = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = top_conf - second_conf

        matched_terms = _detect_ambiguous_terms(content)
        low_conf = top_conf < self.confidence_threshold
        narrow_margin = margin < self.margin_threshold
        is_ambiguous = low_conf or narrow_margin

        reason: Optional[str] = None
        if is_ambiguous:
            reasons = []
            if low_conf:
                reasons.append(
                    f"confiança abaixo do limiar ({top_conf:.1%} < {self.confidence_threshold:.0%})"
                )
            if narrow_margin and len(ordered) > 1:
                reasons.append(
                    f"margem estreita entre {top_label} e {ordered[1][0]} ({margin:.1%})"
                )
            if matched_terms:
                reasons.append(
                    f"termos potencialmente neutros/técnicos presentes: {', '.join(matched_terms)}"
                )
            reason = "; ".join(reasons)

        return PredictionResult(
            label=top_label,
            confidence=float(top_conf),
            probabilities=probabilities,
            is_ambiguous=is_ambiguous,
            ambiguous_reason=reason,
            matched_ambiguous_terms=matched_terms,
        )

    def predict_detailed(self, title: str, text: str) -> PredictionResult:
        """Retorna classe, probabilidades por classe e sinalização de ambiguidade."""
        return self.predict_many_detailed([(title, text)])[0]

    def predict_many_detailed(
        self, items: Sequence[Tuple[str, str]]
    ) -> List[PredictionResult]:
        """Classifica em lote retornando probabilidades e flags de ambiguidade."""
        if not items:
            return []
        contents = [combine_title_and_text(title, text) for title, text in items]
        embeddings = self.embedder.embed_batch(contents)

        if hasattr(self.mlp, "predict_proba"):
            proba_matrix = self.mlp.predict_proba(embeddings)
        else:
            preds = self.mlp.predict(embeddings)
            n_classes = len(self.encoder.classes_)
            proba_matrix = np.zeros((len(preds), n_classes), dtype=np.float32)
            for idx, pred in enumerate(preds):
                proba_matrix[idx, int(pred)] = 1.0

        return [
            self._build_prediction_result(content, row_probs)
            for content, row_probs in zip(contents, proba_matrix)
        ]

    def predict(self, title: str, text: str) -> str:
        """Classifica uma única notícia retornando apenas o nome da classe."""
        return self.predict_detailed(title, text).label

    def predict_many(self, items: Sequence[Tuple[str, str]]) -> List[str]:
        """Classifica uma lista de pares (titulo, texto) retornando os rótulos."""
        return [res.label for res in self.predict_many_detailed(items)]

    def save(
        self,
        artifacts_dir: Path = ARTIFACTS_DIR,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Salva artefatos leves do classificador, metadados e relatório de métricas."""
        target = Path(artifacts_dir)
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.mlp, target / MLP_FILENAME)
        joblib.dump(self.encoder, target / ENCODER_FILENAME)

        meta = {
            **self.metadata,
            "bert_model_name": getattr(self.embedder, "model_name", DEFAULT_BERT_MODEL),
            "max_length": getattr(self.embedder, "max_length", DEFAULT_MAX_LENGTH),
            "pooling": getattr(self.embedder, "pooling", DEFAULT_POOLING),
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "classes": [str(c) for c in self.encoder.classes_],
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        }
        (target / METADATA_FILENAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if metrics is not None:
            serializable_metrics = {
                k: v for k, v in metrics.items() if k != "classification_report"
            }
            (target / METRICS_FILENAME).write_text(
                json.dumps(serializable_metrics, indent=2, ensure_ascii=False) + "\n",
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
                pooling=str(metadata.get("pooling", "cls")),
            )

        return cls(
            mlp=mlp,
            encoder=encoder,
            embedder=embedder,
            metadata=metadata,
            confidence_threshold=float(
                metadata.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
            ),
            margin_threshold=float(
                metadata.get("margin_threshold", DEFAULT_MARGIN_THRESHOLD)
            ),
        )


def train_classifier(
    df: pd.DataFrame,
    embedder: Optional[BertEmbedder] = None,
    artifacts_dir: Optional[Path] = ARTIFACTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    classifier_type: str = DEFAULT_CLASSIFIER_TYPE,
    use_scaler: bool = True,
    cv_splits: int = DEFAULT_CV_SPLITS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> Tuple[NewsClassifier, Dict[str, Any]]:
    """Valida dataset, gera embeddings, roda CV estratificada, treina e exporta métricas."""
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
    class_names = [str(c) for c in encoder.classes_]

    all_embeddings = active_embedder.embed_batch(contents)

    cv_pipeline = build_classifier_pipeline(
        classifier_type=classifier_type,
        use_scaler=use_scaler,
        random_state=random_state,
    )
    cv_summary = _evaluate_cv(
        cv_pipeline, all_embeddings, y_encoded, n_splits=cv_splits, random_state=random_state
    )

    indices = np.arange(len(contents))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    x_train_emb = all_embeddings[train_idx]
    x_test_emb = all_embeddings[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    pipeline = build_classifier_pipeline(
        classifier_type=classifier_type,
        use_scaler=use_scaler,
        random_state=random_state,
    )
    pipeline.fit(x_train_emb, y_train)

    y_pred = pipeline.predict(x_test_emb)
    y_proba = pipeline.predict_proba(x_test_emb)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    loss = float(log_loss(y_test, y_proba, labels=list(range(len(class_names)))))
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    report_text = classification_report(y_test, y_pred, target_names=class_names)

    per_class = {
        cls_name: {
            "precision": float(report_dict[cls_name]["precision"]),
            "recall": float(report_dict[cls_name]["recall"]),
            "f1": float(report_dict[cls_name]["f1-score"]),
            "support": int(report_dict[cls_name]["support"]),
        }
        for cls_name in class_names
    }

    metrics: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "bert_model_name": getattr(active_embedder, "model_name", DEFAULT_BERT_MODEL),
            "pooling": getattr(active_embedder, "pooling", DEFAULT_POOLING),
            "max_length": getattr(active_embedder, "max_length", DEFAULT_MAX_LENGTH),
            "classifier_type": classifier_type,
            "use_scaler": use_scaler,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
        },
        "dataset": {
            "total_samples": len(contents),
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "classes": class_names,
        },
        "holdout": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "log_loss": loss,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
            "polarization_errors": _compute_polarization_errors(cm, class_names),
        },
        "cross_validation": cv_summary,
        # Campos legados no topo para compatibilidade direta com callers antigos
        "accuracy": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "classes": class_names,
    }

    classifier = NewsClassifier(
        mlp=pipeline,
        encoder=encoder,
        embedder=active_embedder,
        metadata={
            "classifier_type": classifier_type,
            "use_scaler": use_scaler,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "cv_accuracy_mean": cv_summary.get("accuracy_mean"),
            "cv_f1_macro_mean": cv_summary.get("f1_macro_mean"),
        },
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )

    if artifacts_dir is not None:
        classifier.save(artifacts_dir, metrics=metrics)

    return classifier, metrics
