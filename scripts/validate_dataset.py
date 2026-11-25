"""CLI para validar e opcionalmente higienizar o dataset de notícias."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import DATA_PATH  # noqa: E402
from src.nlp.dataset import clean_dataset, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida e opcionalmente limpa o CSV de notícias classificadas."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PATH,
        help="Caminho do arquivo CSV de entrada.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Aplica limpeza (strip, remoção de linhas vazias e duplicatas) no arquivo.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")

    if args.fix:
        cleaned, report = clean_dataset(df)
        cleaned.to_csv(args.input, index=False, encoding="utf-8", lineterminator="\n")
        print(f"Dataset limpo salvo em: {args.input}")
        print(
            f"Linhas originais: {report.total_rows} | "
            f"Válidas: {report.valid_rows} | "
            f"Duplicatas removidas: {report.duplicates_removed} | "
            f"Rótulos normalizados: {report.labels_stripped}"
        )
        print(f"Distribuição por classe: {report.class_counts}")
        return 0

    report = validate_dataset(df)
    print(f"Linhas avaliadas: {report.total_rows}")
    print(f"Distribuição por classe: {report.class_counts}")
    if report.is_valid:
        print("OK: dataset válido.")
        return 0

    print("ERRO: problemas encontrados no dataset:")
    for issue in report.issues:
        print(f" - {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
