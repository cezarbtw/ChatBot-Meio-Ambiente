"""Dependências compartilhadas da aplicação FastAPI (estado em app.state)."""

from pathlib import Path
from typing import Callable, Optional, Sequence

from fastapi import HTTPException, Request

from src.collectors import BaseCollector
from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH
from src.nlp.classifier import NewsClassifier
from src.storage import NewsRepository, create_db_engine, init_db


class AppContainer:
    """Mantém instâncias únicas de repositório SQLite e classificador em memória."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        artifacts_dir: Path = ARTIFACTS_DIR,
        classifier: Optional[NewsClassifier] = None,
        load_model: bool = True,
        collectors_factory: Optional[Callable[[], Sequence[BaseCollector]]] = None,
    ) -> None:
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self.engine = create_db_engine(db_path)
        init_db(self.engine)
        self.repository = NewsRepository(self.engine)
        self.collectors_factory = collectors_factory
        self.classifier: Optional[NewsClassifier] = classifier
        if self.classifier is None and load_model:
            self.classifier = NewsClassifier.load(artifacts_dir)
        if self.classifier is not None:
            embedder = getattr(self.classifier, "embedder", None)
            if embedder is not None and hasattr(embedder, "_ensure_loaded"):
                embedder._ensure_loaded()


def get_container(request: Request) -> AppContainer:
    container: Optional[AppContainer] = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=503, detail="Aplicação ainda não inicializada."
        )
    return container


def get_repository(request: Request) -> NewsRepository:
    return get_container(request).repository


def get_classifier(request: Request) -> NewsClassifier:
    container = get_container(request)
    if container.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classificador PLN não carregado no servidor.",
        )
    return container.classifier
