"""Entrypoint ASGI padrão para execução via uvicorn src.api.main:app."""

import os
from pathlib import Path

from src.api.app import create_app
from src.config import ARTIFACTS_DIR, DEFAULT_DB_PATH

_db_path = Path(os.getenv("CHATBOT_DB_PATH", str(DEFAULT_DB_PATH)))
_artifacts_dir = Path(os.getenv("CHATBOT_ARTIFACTS_DIR", str(ARTIFACTS_DIR)))

app = create_app(db_path=_db_path, artifacts_dir=_artifacts_dir)
