"""Script de treinamento do classificador MLP com embeddings BERT."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import ARTIFACTS_DIR, DATA_PATH, DEFAULT_KEYWORDS
from src.nlp.classifier import train_classifier
from src.nlp.dataset import load_dataset
from src.nlp.preprocessing import combine_title_and_text


def plot_keyword_distribution(df, save_path: Path = None, show: bool = True) -> None:
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
        description="Treina o classificador MLP usando embeddings do BERT em português."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-plot", type=Path, default=None)
    args = parser.parse_args()

    df = load_dataset(args.data, require_valid=True)
    classifier, metrics = train_classifier(df, artifacts_dir=args.artifacts_dir)

    print("Acurácia:", metrics["accuracy"])
    print("\nMatriz de Confusão:\n", metrics["confusion_matrix"])
    print("\nRelatório de Classificação:\n", metrics["classification_report"])

    titulo = "Pesquisadores criam método inovador"
    texto = "para purificar água usando algas naturais."
    print("Classificação da nova notícia:", classifier.predict(titulo, texto))

    if not args.no_plot or args.save_plot is not None:
        plot_keyword_distribution(
            df, save_path=args.save_plot, show=not args.no_plot
        )


if __name__ == "__main__":
    main()
