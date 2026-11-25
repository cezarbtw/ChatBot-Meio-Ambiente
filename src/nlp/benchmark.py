"""Benchmark unificado de modelos (TF-IDF clássicos, embeddings Transformer e fine-tuning)."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    ARTIFACTS_DIR,
    BENCHMARK_FILENAME,
    DEFAULT_BERT_MODEL,
    DEFAULT_CV_SPLITS,
    DEFAULT_MAX_LENGTH,
)
from src.nlp.classifier import (
    _compute_polarization_errors,
    _evaluate_cv,
    build_classifier_pipeline,
)
from src.nlp.dataset import validate_dataset
from src.nlp.embedder import BertEmbedder
from src.nlp.preprocessing import combine_title_and_text


@dataclass
class BenchmarkEntry:
    """Resultado padronizado de uma configuração avaliada no benchmark."""

    family: str
    name: str
    model_name: str
    pooling: Optional[str]
    use_scaler: bool
    holdout_accuracy: float
    holdout_f1_macro: float
    holdout_log_loss: Optional[float]
    cv_accuracy_mean: Optional[float]
    cv_accuracy_std: Optional[float]
    cv_f1_macro_mean: Optional[float]
    cv_f1_macro_std: Optional[float]
    boa_as_ruim_count: int
    boa_as_ruim_rate: float
    train_time_ms: float
    predict_time_ms: float


def _to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _prepare_corpus(df: pd.DataFrame) -> Tuple[List[str], LabelEncoder, np.ndarray]:
    report = validate_dataset(df)
    if not report.is_valid:
        raise ValueError(f"Dataset inválido: {'; '.join(report.issues)}")

    contents = [
        combine_title_and_text(row.titulo, row.texto)
        for row in df.itertuples(index=False)
    ]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df["rotulo"])
    return contents, encoder, y_encoded


def _build_tfidf_candidates(random_state: int) -> List[Tuple[str, Any, bool]]:
    return [
        ("MLP", MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1500, early_stopping=False, random_state=random_state), False),
        ("DecisionTree", tree.DecisionTreeClassifier(random_state=random_state), False),
        ("RandomForest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=random_state), False),
        ("AdaBoost", AdaBoostClassifier(random_state=random_state), False),
        ("KNN", KNeighborsClassifier(n_neighbors=5), False),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=random_state), False),
        ("SVM", SVC(probability=True, random_state=random_state), False),
        ("GaussianNB", GaussianNB(), True),
        ("LDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), True),
        ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.1), True),
    ]


def run_tfidf_benchmark(
    df: pd.DataFrame,
    test_size: float = 0.3,
    cv_splits: int = DEFAULT_CV_SPLITS,
    max_features: int = 1000,
    random_state: int = 42,
) -> List[BenchmarkEntry]:
    """Executa comparativo clássico TF-IDF sem vazamento de dados no vocabulário."""
    contents, encoder, y_encoded = _prepare_corpus(df)
    class_names = [str(c) for c in encoder.classes_]

    x_train, x_test, y_train, y_test = train_test_split(
        contents,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    min_class = int(np.bincount(y_encoded).min())
    effective_splits = min(cv_splits, min_class)
    cv = (
        StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        if effective_splits >= 2
        else None
    )

    results: List[BenchmarkEntry] = []
    for name, estimator, needs_dense in _build_tfidf_candidates(random_state):
        steps: List[Tuple[str, Any]] = [
            ("tfidf", TfidfVectorizer(max_features=max_features))
        ]
        if needs_dense:
            steps.append(
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True))
            )
        steps.append(("classifier", estimator))
        pipeline = Pipeline(steps)

        cv_acc_mean = cv_acc_std = cv_f1_mean = cv_f1_std = None
        if cv is not None:
            cv_scores = cross_validate(
                pipeline,
                contents,
                y_encoded,
                cv=cv,
                scoring=("accuracy", "f1_macro"),
            )
            cv_acc_mean = float(np.mean(cv_scores["test_accuracy"]))
            cv_acc_std = float(np.std(cv_scores["test_accuracy"]))
            cv_f1_mean = float(np.mean(cv_scores["test_f1_macro"]))
            cv_f1_std = float(np.std(cv_scores["test_f1_macro"]))

        t0 = perf_counter()
        pipeline.fit(x_train, y_train)
        train_ms = (perf_counter() - t0) * 1000.0

        t1 = perf_counter()
        y_pred = pipeline.predict(x_test)
        predict_ms = (perf_counter() - t1) * 1000.0

        loss: Optional[float] = None
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(x_test)
            loss = float(log_loss(y_test, y_proba, labels=list(range(len(class_names)))))

        cm = confusion_matrix(y_test, y_pred)
        pol = _compute_polarization_errors(cm, class_names)

        results.append(
            BenchmarkEntry(
                family="tfidf",
                name=f"TFIDF+{name}",
                model_name="TfidfVectorizer",
                pooling=None,
                use_scaler=False,
                holdout_accuracy=float(accuracy_score(y_test, y_pred)),
                holdout_f1_macro=float(f1_score(y_test, y_pred, average="macro")),
                holdout_log_loss=loss,
                cv_accuracy_mean=cv_acc_mean,
                cv_accuracy_std=cv_acc_std,
                cv_f1_macro_mean=cv_f1_mean,
                cv_f1_macro_std=cv_f1_std,
                boa_as_ruim_count=pol["boa_classified_as_ruim_count"],
                boa_as_ruim_rate=pol["boa_classified_as_ruim_rate"],
                train_time_ms=train_ms,
                predict_time_ms=predict_ms,
            )
        )

    return results


def run_embedding_benchmark(
    df: pd.DataFrame,
    model_names: Sequence[str] = (DEFAULT_BERT_MODEL,),
    poolings: Sequence[str] = ("cls", "mean"),
    classifier_types: Sequence[str] = ("mlp", "mlp_calibrated", "logreg", "svc"),
    scaler_options: Sequence[bool] = (False, True),
    max_length: int = DEFAULT_MAX_LENGTH,
    test_size: float = 0.2,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = 42,
    embedder_factory: Optional[Any] = None,
) -> List[BenchmarkEntry]:
    """Compara modelos Transformer, pooling ([CLS] vs mean), StandardScaler e cabeças."""
    contents, encoder, y_encoded = _prepare_corpus(df)
    class_names = [str(c) for c in encoder.classes_]

    indices = np.arange(len(contents))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    results: List[BenchmarkEntry] = []
    for model_name in model_names:
        short_model = model_name.split("/")[-1]
        for pooling in poolings:
            if embedder_factory is not None:
                embedder = embedder_factory(model_name=model_name, max_length=max_length, pooling=pooling)
            else:
                embedder = BertEmbedder(
                    model_name=model_name,
                    max_length=max_length,
                    pooling=pooling,
                )

            all_embeddings = embedder.embed_batch(contents)
            x_train = all_embeddings[train_idx]
            x_test = all_embeddings[test_idx]

            for use_scaler in scaler_options:
                for clf_type in classifier_types:
                    cv_pipe = build_classifier_pipeline(
                        classifier_type=clf_type,
                        use_scaler=use_scaler,
                        random_state=random_state,
                    )
                    cv_summary = _evaluate_cv(
                        cv_pipe,
                        all_embeddings,
                        y_encoded,
                        n_splits=cv_splits,
                        random_state=random_state,
                    )

                    pipe = build_classifier_pipeline(
                        classifier_type=clf_type,
                        use_scaler=use_scaler,
                        random_state=random_state,
                    )
                    t0 = perf_counter()
                    pipe.fit(x_train, y_train)
                    train_ms = (perf_counter() - t0) * 1000.0

                    t1 = perf_counter()
                    y_pred = pipe.predict(x_test)
                    predict_ms = (perf_counter() - t1) * 1000.0

                    y_proba = pipe.predict_proba(x_test)
                    loss = float(
                        log_loss(y_test, y_proba, labels=list(range(len(class_names))))
                    )
                    cm = confusion_matrix(y_test, y_pred)
                    pol = _compute_polarization_errors(cm, class_names)

                    scaler_tag = "scaler" if use_scaler else "raw"
                    entry_name = f"{short_model}[{pooling},{scaler_tag}]+{clf_type}"
                    results.append(
                        BenchmarkEntry(
                            family="transformer_embeddings",
                            name=entry_name,
                            model_name=model_name,
                            pooling=pooling,
                            use_scaler=use_scaler,
                            holdout_accuracy=float(accuracy_score(y_test, y_pred)),
                            holdout_f1_macro=float(
                                f1_score(y_test, y_pred, average="macro")
                            ),
                            holdout_log_loss=loss,
                            cv_accuracy_mean=cv_summary.get("accuracy_mean"),
                            cv_accuracy_std=cv_summary.get("accuracy_std"),
                            cv_f1_macro_mean=cv_summary.get("f1_macro_mean"),
                            cv_f1_macro_std=cv_summary.get("f1_macro_std"),
                            boa_as_ruim_count=pol["boa_classified_as_ruim_count"],
                            boa_as_ruim_rate=pol["boa_classified_as_ruim_rate"],
                            train_time_ms=train_ms,
                            predict_time_ms=predict_ms,
                        )
                    )

    return results


class _SequenceDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def finetune_sequence_classifier(
    df: pd.DataFrame,
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> BenchmarkEntry:
    """Executa fine-tuning supervisionado ponta a ponta com AutoModelForSequenceClassification."""
    torch.manual_seed(random_state)
    contents, encoder, y_encoded = _prepare_corpus(df)
    class_names = [str(c) for c in encoder.classes_]

    x_train, x_test, y_train, y_test = train_test_split(
        contents,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(class_names)
    )
    model.to(device)

    train_enc = tokenizer(
        list(x_train),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        list(x_test),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    train_loader = DataLoader(
        _SequenceDataset(train_enc, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        _SequenceDataset(test_enc, y_test), batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    t0 = perf_counter()
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            outputs.loss.backward()
            optimizer.step()
    train_ms = (perf_counter() - t0) * 1000.0

    model.eval()
    all_probs = []
    t1 = perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    predict_ms = (perf_counter() - t1) * 1000.0

    y_proba = np.vstack(all_probs)
    y_pred = np.argmax(y_proba, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    pol = _compute_polarization_errors(cm, class_names)

    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)

    short_model = model_name.split("/")[-1]
    return BenchmarkEntry(
        family="transformer_finetuned",
        name=f"{short_model}[finetuned-{epochs}ep]",
        model_name=model_name,
        pooling="sequence_cls_head",
        use_scaler=False,
        holdout_accuracy=float(accuracy_score(y_test, y_pred)),
        holdout_f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        holdout_log_loss=float(
            log_loss(y_test, y_proba, labels=list(range(len(class_names))))
        ),
        cv_accuracy_mean=None,
        cv_accuracy_std=None,
        cv_f1_macro_mean=None,
        cv_f1_macro_std=None,
        boa_as_ruim_count=pol["boa_classified_as_ruim_count"],
        boa_as_ruim_rate=pol["boa_classified_as_ruim_rate"],
        train_time_ms=train_ms,
        predict_time_ms=predict_ms,
    )


def export_benchmark_report(
    entries: Sequence[BenchmarkEntry],
    output_path: Path = ARTIFACTS_DIR / BENCHMARK_FILENAME,
) -> Dict[str, Any]:
    """Ordena resultados por F1 macro (CV ou holdout) e exporta JSON para o dashboard."""
    ranked = sorted(
        entries,
        key=lambda e: (
            e.cv_f1_macro_mean if e.cv_f1_macro_mean is not None else e.holdout_f1_macro,
            e.holdout_accuracy,
        ),
        reverse=True,
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_configurations": len(ranked),
        "best_configuration": asdict(ranked[0]) if ranked else None,
        "results": [asdict(entry) for entry in ranked],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def plot_benchmark_results(
    entries: Sequence[BenchmarkEntry],
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Gera gráficos comparativos de acurácia/F1 e tempos de treino/previsão."""
    if not entries:
        return

    names = [e.name for e in entries]
    f1_scores = [
        e.cv_f1_macro_mean if e.cv_f1_macro_mean is not None else e.holdout_f1_macro
        for e in entries
    ]
    train_times = [e.train_time_ms for e in entries]
    predict_times = [e.predict_time_ms for e in entries]

    charts = [
        ("F1-Macro (CV / Holdout)", "F1-Macro", f1_scores, "benchmark_f1.png"),
        ("Tempo de Treinamento (ms)", "Tempo (ms)", train_times, "benchmark_train_ms.png"),
        ("Tempo de Previsão (ms)", "Tempo (ms)", predict_times, "benchmark_predict_ms.png"),
    ]

    for title, ylabel, values, filename in charts:
        plt.figure(figsize=(12, 6))
        plt.bar(names, values)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / filename)
        if show:
            plt.show()
        plt.close()
