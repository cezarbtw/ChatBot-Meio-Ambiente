"""Script de coleta multi-fonte, inspeção de protocolos de rede e classificação."""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.collectors import (
    COLLECTOR_FACTORIES,
    CollectedItem,
    CollectorResult,
    G1Collector,
    collect_from_sources,
)
from src.collectors.manager import build_telemetry_report, export_telemetry_report
from src.config import ARTIFACTS_DIR, TELEMETRY_FILENAME
from src.network import PacketCaptureSummary, capture_network_traffic
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
        description="Coleta notícias/posts ambientais, inspeciona protocolos e classifica com confiança."
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
    parser.add_argument(
        "--inspect-items",
        action="store_true",
        help="Inspeciona individualmente a URL final de cada item coletado.",
    )
    parser.add_argument(
        "--export-telemetry",
        type=Path,
        default=ARTIFACTS_DIR / TELEMETRY_FILENAME,
        help="Caminho JSON para salvar o resumo de telemetria de rede.",
    )
    parser.add_argument(
        "--capture-packets",
        action="store_true",
        help="Executa captura opcional de pacotes em interface local (requer tshark ou scapy).",
    )
    parser.add_argument("--capture-interface", type=str, default=None)
    parser.add_argument("--capture-seconds", type=int, default=5)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    classifier = NewsClassifier.load(args.artifacts_dir)

    capture_summary: Optional[PacketCaptureSummary] = None
    if args.capture_packets:
        capture_summary = capture_network_traffic(
            interface=args.capture_interface,
            duration_seconds=args.capture_seconds,
        )

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
        items, reports = collect_from_sources(
            limit_per_source=args.limit,
            inspect_items=args.inspect_items,
        )
    else:
        items, reports = collect_from_sources(
            sources=[args.source],
            limit_per_source=args.limit,
            inspect_items=args.inspect_items,
        )

    if reports:
        telemetry_report = build_telemetry_report(
            reports, items=items, capture_summary=capture_summary
        )
        export_telemetry_report(telemetry_report, output_path=args.export_telemetry)

        print("=== Resumo da Coleta & Protocolos de Rede ===")
        for rep in reports:
            status = "OK" if rep.success else f"ERRO ({rep.error})"
            tel = rep.source_telemetry or {}
            proto_str = ""
            if tel:
                ips = ", ".join((tel.get("resolved_ips") or [])[:2]) or "-"
                proto_str = (
                    f" | {tel.get('http_version') or '-'} "
                    f"(ALPN: {tel.get('alpn_protocol') or '-'}, "
                    f"TLS: {tel.get('tls_version') or '-'}, "
                    f"HTTP {tel.get('status_code') or '-'}, IP: {ips})"
                )
            print(
                f"  [{rep.source_type}] {rep.source}: {len(rep.items)} item(ns) | "
                f"{status} | {rep.duration_ms:.0f} ms{proto_str}"
            )
        summary = telemetry_report["summary"]
        print(
            f"  Resumo agregado -> ALPN: {summary['alpn_protocols']} | "
            f"TLS: {summary['tls_versions']} | HTTP: {summary['http_versions']} | "
            f"DNS médio: {summary['avg_dns_ms']} ms | TLS médio: {summary['avg_tls_ms']} ms"
        )
        print(f"  Telemetria salva em: {args.export_telemetry}\n")

    if capture_summary is not None:
        print("=== Captura de Pacotes (Modo Opcional) ===")
        if capture_summary.available and not capture_summary.error:
            print(
                f"  Backend: {capture_summary.backend} | Pacotes: {capture_summary.total_packets} | "
                f"Protocolos: {capture_summary.protocol_counts}\n"
            )
        else:
            print(f"  Aviso: {capture_summary.error}\n")

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
        if item.network_telemetry:
            tel = item.network_telemetry
            cert = tel.get("certificate") or {}
            print(
                f"  Rede/TLS      : IP={','.join((tel.get('resolved_ips') or [])[:2])} | "
                f"Cipher={tel.get('cipher_suite') or '-'} | "
                f"Emissor={cert.get('issuer_org') or cert.get('issuer_cn') or '-'} | "
                f"DNS={tel.get('dns_ms')}ms TLS={tel.get('tls_ms')}ms TTFB={tel.get('ttfb_ms')}ms"
            )
        print(f"  Data          : {data_label}")

    if not args.no_plot and dados_resultado:
        plot_daily_counts(pd.DataFrame(dados_resultado))


if __name__ == "__main__":
    main()
