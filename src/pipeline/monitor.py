"""Pipeline incremental e agendado de coleta, classificação e persistência."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, List, Optional, Sequence, Union

from src.collectors import DEFAULT_SOURCES, BaseCollector, CollectorResult, collect_from_sources
from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH
from src.nlp.classifier import NewsClassifier
from src.storage.database import create_db_engine, init_db
from src.storage.repository import NewsRepository


@dataclass
class MonitoringCycleResult:
    """Resultado consolidado de uma execução do pipeline de monitoramento."""

    run_id: int
    collected: int
    inserted: int
    deduplicated: int
    errors_count: int
    duration_ms: float
    source_reports: List[CollectorResult]


def run_monitoring_cycle(
    sources: Optional[Sequence[str]] = None,
    limit_per_source: int = 10,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    inspect_items: bool = False,
    classifier: Optional[NewsClassifier] = None,
    collectors: Optional[Sequence[BaseCollector]] = None,
    repository: Optional[NewsRepository] = None,
) -> MonitoringCycleResult:
    """Executa 1 ciclo completo: coleta multi-fonte -> classificação -> gravação SQLite."""
    t0 = perf_counter()

    if repository is None:
        engine = create_db_engine(db_path)
        init_db(engine)
        repository = NewsRepository(engine)

    active_classifier = classifier or NewsClassifier.load(artifacts_dir)
    target_sources = (
        [c.source_name for c in collectors]
        if collectors is not None
        else (list(sources) if sources is not None else list(DEFAULT_SOURCES))
    )

    items, reports = collect_from_sources(
        sources=None if collectors is not None else target_sources,
        limit_per_source=limit_per_source,
        collectors=collectors,
        inspect_items=inspect_items,
    )

    predictions = (
        active_classifier.predict_many_detailed(
            [(item.title, item.text) for item in items]
        )
        if items
        else []
    )
    items_with_preds = list(zip(items, predictions))

    batch_result = repository.save_collection_batch(
        sources_requested=target_sources,
        items_with_predictions=items_with_preds,
        reports=reports,
    )

    return MonitoringCycleResult(
        run_id=batch_result.run_id,
        collected=batch_result.items_collected,
        inserted=batch_result.items_inserted,
        deduplicated=batch_result.items_deduplicated,
        errors_count=batch_result.errors_count,
        duration_ms=(perf_counter() - t0) * 1000.0,
        source_reports=reports,
    )


def run_scheduled_monitor(
    sources: Optional[Sequence[str]] = None,
    limit_per_source: int = 10,
    interval_seconds: float = 3600.0,
    max_runs: Optional[int] = None,
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    inspect_items: bool = False,
    classifier: Optional[NewsClassifier] = None,
    collectors_factory: Optional[Callable[[], Sequence[BaseCollector]]] = None,
    repository: Optional[NewsRepository] = None,
    on_cycle: Optional[Callable[[int, MonitoringCycleResult], None]] = None,
    sleep_fn: Callable[[float], None] = sleep,
) -> List[MonitoringCycleResult]:
    """Executa o pipeline em ciclos periódicos reutilizando modelo e banco em memória."""
    if repository is None:
        engine = create_db_engine(db_path)
        init_db(engine)
        repository = NewsRepository(engine)

    active_classifier = classifier or NewsClassifier.load(artifacts_dir)
    history: List[MonitoringCycleResult] = []
    cycle_num = 0

    try:
        while max_runs is None or cycle_num < max_runs:
            cycle_num += 1
            custom_collectors = collectors_factory() if collectors_factory else None
            result = run_monitoring_cycle(
                sources=sources,
                limit_per_source=limit_per_source,
                inspect_items=inspect_items,
                classifier=active_classifier,
                collectors=custom_collectors,
                repository=repository,
            )
            history.append(result)
            if on_cycle is not None:
                on_cycle(cycle_num, result)

            if max_runs is not None and cycle_num >= max_runs:
                break
            if interval_seconds > 0:
                sleep_fn(interval_seconds)
    except KeyboardInterrupt:
        pass

    return history
