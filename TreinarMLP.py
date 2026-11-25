"""Script de treinamento do classificador MLP com embeddings Transformer."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from src.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DEFAULT_BERT_MODEL,
    DEFAULT_CLASSIFIER_TYPE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CV_SPLITS,
    DEFAULT_KEYWORDS,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_MAX_LENGTH,
    DEFAULT_POOLING,
    METRICS_FILENAME,
    VALID_CLASSIFIER_TYPES,
    VALID_POOLINGS,
)
from src.nlp.classifier import train_classifier
from src.nlp.dataset import load_dataset
from src.nlp.embedder import BertEmbedder
from src.nlp.preprocessing import combine_title_and_text


def plot_keyword_distribution(
    df, save_path: Optional[Path] = None, show: bool = True
) -> None:
    """Gera gráfico percentual de incidência de palavras-chave por classe."""
    working = df.copy()
    working["conteudo"] = [
        combine_title_and_text(row.titulo, row.texto)
        for row in working.itertuples(index=False)
    ]

    keywords = list(DEFAULT_KEYWORDS)
    for palavra in keywords:
        working[palavra] = working["conteudo"].str.count(palavra)

    ocorrencias = working.groupby("rotulo")[keywords].sum()
    ocorrencias_percentual = ocorrencias.div(ocorrencias.sum(axis=0), axis=1) * 100

    ocorrencias_percentual.plot(kind="bar", figsize=(12, 6))
    plt.title("Distribuição Percentual das Palavras-Chave por Tipo de Notícia")
    plt.ylabel("Porcentagem (%)")
    plt.xlabel("Classificação da Notícia")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina classificador com embeddings Transformer, CV estratificada e calibração."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--pooling", choices=VALID_POOLINGS, default=DEFAULT_POOLING)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--classifier",
        choices=VALID_CLASSIFIER_TYPES,
        default=DEFAULT_CLASSIFIER_TYPE,
    )
    parser.add_argument("--no-scaler", action="store_true")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD
    )
    parser.add_argument(
        "--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-plot", type=Path, default=None)
    args = parser.parse_args()

    df = load_dataset(args.data, require_valid=True)
    embedder = BertEmbedder(
        model_name=args.model_name,
        max_length=args.max_length,
        pooling=args.pooling,
    )

    classifier, metrics = train_classifier(
        df,
        embedder=embedder,
        artifacts_dir=args.artifacts_dir,
        classifier_type=args.classifier,
        use_scaler=not args.no_scaler,
        cv_splits=args.cv_splits,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )

    holdout = metrics["holdout"]
    cv = metrics["cross_validation"]
    pol = holdout["polarization_errors"]

    print(f"Modelo: {args.model_name} | Pooling: {args.pooling} | MaxLen: {args.max_length}")
    print(f"Classificador: {args.classifier} | StandardScaler: {not args.no_scaler}")
    print(f"Acurácia (Holdout): {holdout['accuracy']:.4f} | F1-Macro: {holdout['f1_macro']:.4f} | LogLoss: {holdout['log_loss']:.4f}")
    if not cv.get("skipped"):
        print(
            f"CV ({cv['n_splits']}-fold) Acurácia: {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f} | "
            f"F1-Macro: {cv['f1_macro_mean']:.4f} ± {cv['f1_macro_std']:.4f}"
        )
    print(
        f"Polarização Boa->Ruim: {pol['boa_classified_as_ruim_count']} "
        f"({pol['boa_classified_as_ruim_rate']:.1%})"
    )
    print("\nMatriz de Confusão:\n", holdout["confusion_matrix"])
    print("\nRelatório de Classificação:\n", metrics["classification_report"])
    print(f"Métricas salvas em: {args.artifacts_dir / METRICS_FILENAME}")

    sample_res = classifier.predict_detailed(
        "Pesquisadores criam método inovador",
        "para purificar água usando algas naturais.",
    )
    print(
        f"Classificação da nova notícia: {sample_res.label} "
        f"(confiança: {sample_res.confidence:.1%}, ambíguo: {sample_res.is_ambiguous})"
    )

    if not args.no_plot or args.save_plot is not None:
        plot_keyword_distribution(
            df, save_path=args.save_plot, show=not args.no_plot
        )


if __name__ == "__main__":
    main()
