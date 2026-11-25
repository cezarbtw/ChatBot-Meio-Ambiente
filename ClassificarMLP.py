"""Script de coleta no G1 e classificação de notícias ambientais."""

import argparse
from pathlib import Path
import re
from typing import List, Tuple

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import requests

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


def extrair_data_do_link(link: str) -> str:
    """Extrai data YYYY-MM-DD a partir do padrão de URL do G1."""
    if not link:
        return "Data não disponível"
    match = re.search(r"/noticia/(\d{4}/\d{2}/\d{2})/", link)
    if match:
        return match.group(1).replace("/", "-")
    return "Data não disponível"


def coletar_noticias_g1() -> List[Tuple[str, str, str, str, str]]:
    """Coleta até 10 manchetes e resumos recentes da editoria Meio Ambiente do G1."""
    url = "https://g1.globo.com/meio-ambiente/"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    noticias_coletadas: List[Tuple[str, str, str, str, str]] = []
    artigos = soup.select(".feed-post")[:10]
    for artigo in artigos:
        titulo_tag = artigo.select_one(".feed-post-body-title")
        link_tag = artigo.select_one("a")
        texto_tag = artigo.select_one(".feed-post-body-resumo")

        if titulo_tag and link_tag:
            titulo = titulo_tag.get_text(strip=True)
            texto = texto_tag.get_text(strip=True) if texto_tag else ""
            link = link_tag.get("href") or ""
            protocolo = link.split(":")[0] if link else "desconhecido"
            data = extrair_data_do_link(link)
            noticias_coletadas.append((titulo, texto, link, protocolo, data))

    return noticias_coletadas


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
    parser = argparse.ArgumentParser(
        description="Classifica notícias coletadas do G1 ou exemplos manuais."
    )
    parser.add_argument(
        "--source",
        choices=("g1", "manual"),
        default="g1",
        help="Origem das notícias a classificar.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    classifier = NewsClassifier.load(args.artifacts_dir)

    if args.source == "manual":
        noticias = [
            (titulo, texto, "Não informado", "Não informado", "Não informado")
            for titulo, texto in NOTICIAS_MANUAIS
        ]
    else:
        noticias = coletar_noticias_g1()

    dados_resultado = []
    for i, (titulo, texto, link, protocolo, data) in enumerate(noticias, start=1):
        classe = classifier.predict(titulo, texto)
        dados_resultado.append(
            {
                "Data": data,
                "Classe": classe,
                "Título": titulo,
                "Link": link,
            }
        )
        print(f"Notícia {i}:")
        print(f"  Título    : {titulo}")
        print(f"  Classe    : {classe}")
        print(f"  Link      : {link}")
        print(f"  Protocolo : {protocolo}")
        print(f"  Data      : {data}")

    if not args.no_plot and dados_resultado:
        plot_daily_counts(pd.DataFrame(dados_resultado))


if __name__ == "__main__":
    main()
