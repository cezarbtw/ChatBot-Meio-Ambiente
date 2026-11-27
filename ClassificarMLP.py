"""Script de coleta multi-fonte (portais ambientais e redes sociais) e classificação."""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.collectors import (
    COLLECTOR_FACTORIES,
    CollectedItem,
    CollectorResult,
    G1Collector,
    collect_from_sources,
)
from src.config import ARTIFACTS_DIR
from src.nlp.classifier import NewsClassifier

NOTICIAS_MANUAIS: List[Tuple[str, str]] = [
    (
        "NO satélite que vai descobrir quanto a Amazônia 'pesa'",
        "Sistema de observação conta com um radar especial que vai medir a quantidade de gás armazenado pelas árvores nas florestas tropicais.",
    ),
    (
        "Financiamento, metas e combustíveis: veja três desafios que o governo Lula prevê nas negociações da COP30",
        "País sediará conferência do clima em novembro deste ano, em Belém (PA). Lula tem dito que planeta está 'farto' de promessas não cumpridas e que é o momento de agir.",
    ),
]

CORES_CLASSE = {
    "Boa": "green",
    "Ruim": "red",
    "Irrelevante": "gray",
}


def coletar_noticias_g1(limit: int = 10) -> List[CollectedItem]:
    """Compatibilidade: coleta notícias recentes do G1 Meio Ambiente."""
    return G1Collector().collect(limit=limit)


def plot_daily_counts(df_resultados: pd.DataFrame) -> None:
    """Plota contagem diária por classe para datas válidas."""
    valid = df_resultados[
        ~df_resultados["Data"].isin(["Data não disponível", "Não informado"])
    ]
    if valid.empty:
        return

    grupo = valid.groupby(["Data", "Classe"]).size().unstack(fill_value=0)
    for data, linha in grupo.iterrows():
        plt.figure(figsize=(10, 5))
        linha.plot(
            kind="bar",
            color=[CORES_CLASSE.get(classe, "gray") for classe in linha.index],
        )
        plt.title(f"Classificação das Notícias em {data}")
        plt.xlabel("Classificação")
        plt.ylabel("Quantidade")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        plt.close()


def main() -> None:
    source_choices = ("all", "manual") + tuple(sorted(COLLECTOR_FACTORIES))
    parser = argparse.ArgumentParser(
        description="Coleta notícias/posts ambientais e classifica com confiança."
    )
    parser.add_argument(
        "--source",
        choices=source_choices,
        default="g1",
        help="Fonte da coleta ('all' executa todos os portais e redes padrão).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Máximo de itens por fonte.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    classifier = NewsClassifier.load(args.artifacts_dir)

    reports: List[CollectorResult] = []
    if args.source == "manual":
        items = [
            CollectedItem(
                source="manual",
                source_type="manual",
                title=titulo,
                text=texto,
                url="Não informado",
                published_at=None,
                protocol="Não informado",
            )
            for titulo, texto in NOTICIAS_MANUAIS
        ]
    elif args.source == "all":
        items, reports = collect_from_sources(limit_per_source=args.limit)
    else:
        items, reports = collect_from_sources(
            sources=[args.source], limit_per_source=args.limit
        )

    if reports:
        print("=== Resumo da Coleta ===")
        for rep in reports:
            status = "OK" if rep.success else f"ERRO ({rep.error})"
            print(
                f"  [{rep.source_type}] {rep.source}: {len(rep.items)} item(ns) | "
                f"{status} | {rep.duration_ms:.0f} ms"
            )
        print()

    results = classifier.predict_many_detailed(
        [(item.title, item.text) for item in items]
    )

    dados_resultado = []
    for i, (item, pred) in enumerate(zip(items, results), start=1):
        data_label = (
            "Não informado" if item.source == "manual" else item.published_date
        )
        dados_resultado.append(
            {
                "Fonte": item.source,
                "Tipo": item.source_type,
                "Data": data_label,
                "Classe": pred.label,
                "Confiança": pred.confidence,
                "Ambíguo": pred.is_ambiguous,
                "Título": item.title,
                "Link": item.url,
            }
        )
        probs_fmt = ", ".join(
            f"{cls_name}: {prob:.1%}"
            for cls_name, prob in sorted(pred.probabilities.items())
        )
        print(f"Item {i} [{item.source}/{item.source_type}]:")
        print(f"  Título        : {item.title}")
        print(f"  Classe        : {pred.label} ({pred.confidence:.1%})")
        print(f"  Probabilidades: {probs_fmt}")
        if pred.is_ambiguous:
            print(f"  [AMBÍGUO]     : {pred.ambiguous_reason}")
        if item.author:
            print(f"  Autor         : {item.author}")
        print(f"  Link          : {item.url}")
        print(f"  Protocolo     : {item.protocol}")
        print(f"  Data          : {data_label}")

    if not args.no_plot and dados_resultado:
        plot_daily_counts(pd.DataFrame(dados_resultado))


if __name__ == "__main__":
    main()
