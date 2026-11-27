"""Fábrica da aplicação FastAPI com lifespan e CORS configurável."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Callable, Optional, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import AppContainer
from src.api.routes import router as api_router
from src.collectors import BaseCollector
from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH
from src.nlp.classifier import NewsClassifier

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def _resolve_cors_origins(explicit_origins: Optional[Sequence[str]]) -> Sequence[str]:
    if explicit_origins is not None:
        return list(explicit_origins)
    raw_env = os.getenv("CHATBOT_CORS_ORIGINS", "").strip()
    if raw_env:
        return [item.strip() for item in raw_env.split(",") if item.strip()]
    return list(DEFAULT_CORS_ORIGINS)


def create_app(
    db_path: Path = DEFAULT_DB_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    classifier: Optional[NewsClassifier] = None,
    load_model_on_startup: bool = True,
    cors_origins: Optional[Sequence[str]] = None,
    collectors_factory: Optional[Callable[[], Sequence[BaseCollector]]] = None,
) -> FastAPI:
    """Cria instância FastAPI configurada com OpenAPI, CORS e ciclo de vida único do modelo."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = AppContainer(
            db_path=db_path,
            artifacts_dir=artifacts_dir,
            classifier=classifier,
            load_model=load_model_on_startup,
            collectors_factory=collectors_factory,
        )
        yield

    app = FastAPI(
        title="Monitor Ambiental — API REST",
        description=(
            "API para coleta multi-fonte, inspeção de protocolos de rede/TLS, "
            "classificação de notícias ambientais (BERT + MLP) e curadoria."
        ),
        version="0.3.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_resolve_cors_origins(cors_origins)),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app
