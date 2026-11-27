"""CLI para execução incremental e agendada do monitoramento no SQLite."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors import COLLECTOR_FACTORIES, DEFAULT_SOURCES  # noqa: E402
from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH  # noqa: E402
from src.pipeline import MonitoringCycleResult, run_scheduled_monitor  # noqa: E402
from src.storage import NewsRepository, create_db_engine, init_db  # noqa: E402


def _print_cycle_summary(cycle_num: int, result: MonitoringCycleResult) -> None:
    print(
        f"[Ciclo {cycle_num} | Run #{result.run_id}] "
        f"Coletados: {result.collected} | Inseridos: {result.inserted} | "
        f"Deduplicados: {result.deduplicated} | Erros de fonte: {result.errors_count} | "
        f"Duração: {result.duration_ms:.0f} ms"
    )


def _print_db_stats(repo: NewsRepository) -> None:
    overview = repo.get_overview_stats()
    daily = repo.get_daily_series()
    kw = repo.get_keyword_distribution()
    proto = repo.get_protocol_stats()

    print("\n=== Estatísticas do Banco Histórico ===")
    print(
        f"Total armazenado: {overview['total_items']} | "
        f"Ambíguos: {overview['ambiguous_items']} | "
        f"Por classe: {overview['by_label']}"
    )
    print(f"Por fonte: {overview['by_source']}")
    print(f"Série diária ({len(daily)} dia(s)): {daily}")
    print(f"Totais de palavras-chave: {kw['totals_by_keyword']}")
    print(
        f"Protocolos armazenados ({proto['total_records']} registro(s)): "
        f"ALPN={proto['alpn_protocols']} | TLS={proto['tls_versions']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa coleta incremental e agendada gravando no banco SQLite."
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=("all",) + tuple(sorted(COLLECTOR_FACTORIES)),
        default=["all"],
        help="Fontes a monitorar.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--inspect-items", action="store_true")
    parser.add_argument("--once", action="store_true", help="Executa apenas 1 ciclo.")
    parser.add_argument("--interval-minutes", type=float, default=60.0)
    parser.add_argument("--interval-seconds", type=float, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Exibe estatísticas consolidadas do banco após (ou sem) coleta.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Apenas exibe estatísticas do banco existente sem coletar.",
    )
    args = parser.parse_args()

    engine = create_db_engine(args.db_path)
    init_db(engine)
    repo = NewsRepository(engine)

    if args.stats_only:
        _print_db_stats(repo)
        return 0

    sources = list(DEFAULT_SOURCES) if "all" in args.source else list(args.source)
    max_runs = 1 if args.once else args.max_runs
    interval_sec = (
        args.interval_seconds
        if args.interval_seconds is not None
        else args.interval_minutes * 60.0
    )

    run_scheduled_monitor(
        sources=sources,
        limit_per_source=args.limit,
        interval_seconds=interval_sec,
        max_runs=max_runs,
        db_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
        inspect_items=args.inspect_items,
        repository=repo,
        on_cycle=_print_cycle_summary,
    )

    if args.stats:
        _print_db_stats(repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
