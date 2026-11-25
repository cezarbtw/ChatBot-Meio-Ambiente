"""Benchmark comparativo de modelos (TF-IDF, embeddings Transformer e fine-tuning)."""

import argparse
from pathlib import Path
from typing import List

from src.config import ARTIFACTS_DIR, BENCHMARK_FILENAME, DATA_PATH, DEFAULT_BERT_MODEL
from src.nlp.benchmark import (
    BenchmarkEntry,
    export_benchmark_report,
    finetune_sequence_classifier,
    plot_benchmark_results,
    run_embedding_benchmark,
    run_tfidf_benchmark,
)
from src.nlp.dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa comparativo entre modelos TF-IDF e embeddings Transformer."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--output-json", type=Path, default=ARTIFACTS_DIR / BENCHMARK_FILENAME
    )
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_BERT_MODEL],
        help="Checkpoints HuggingFace avaliados quando --include-embeddings estiver ativo.",
    )
    parser.add_argument("--finetune-model", type=str, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-plots-dir", type=Path, default=None)
    args = parser.parse_args()

    df = load_dataset(args.data, require_valid=True)

    entries: List[BenchmarkEntry] = []
    entries.extend(run_tfidf_benchmark(df, cv_splits=args.cv_splits))

    if args.include_embeddings:
        entries.extend(
            run_embedding_benchmark(
                df,
                model_names=args.models,
                cv_splits=args.cv_splits,
            )
        )

    if args.finetune_model:
        entries.append(
            finetune_sequence_classifier(
                df,
                model_name=args.finetune_model,
                epochs=args.finetune_epochs,
            )
        )

    report = export_benchmark_report(entries, output_path=args.output_json)
    print(f"\nBenchmark salvo em: {args.output_json}")
    print(
        f"{'Configuração':<46} {'CV F1':>8} {'Holdout Acc':>12} {'Holdout F1':>11} {'Boa->Ruim':>10}"
    )
    print("-" * 92)
    for row in report["results"]:
        cv_f1 = (
            f"{row['cv_f1_macro_mean']:.4f}"
            if row["cv_f1_macro_mean"] is not None
            else "   N/A"
        )
        print(
            f"{row['name']:<46} {cv_f1:>8} {row['holdout_accuracy']:>12.4f} "
            f"{row['holdout_f1_macro']:>11.4f} {row['boa_as_ruim_count']:>10d}"
        )

    if not args.no_plot or args.save_plots_dir is not None:
        plot_benchmark_results(
            entries, save_dir=args.save_plots_dir, show=not args.no_plot
        )


if __name__ == "__main__":
    main()
