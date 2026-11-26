"""Configuração de engine, criação de schema e sessões SQLAlchemy para SQLite."""

from pathlib import Path
from typing import Union

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config import DEFAULT_DB_PATH
from src.storage.models import Base


def build_sqlite_url(db_path: Union[str, Path] = DEFAULT_DB_PATH) -> str:
    """Converte caminho de arquivo ou URL SQLite em URL SQLAlchemy válida."""
    raw = str(db_path)
    if raw.startswith("sqlite:"):
        return raw
    if raw == ":memory:":
        return "sqlite:///:memory:"
    resolved = Path(raw).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved.as_posix()}"


def create_db_engine(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    echo: bool = False,
) -> Engine:
    """Cria Engine SQLAlchemy com foreign keys habilitadas e suporte multi-thread."""
    url = build_sqlite_url(db_path)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite:") else {}
    engine = create_engine(url, echo=echo, connect_args=connect_args, future=True)

    if url.startswith("sqlite:"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Cria todas as tabelas no banco se ainda não existirem."""
    Base.metadata.create_all(bind=engine)


def create_session_factory(engine: Engine) -> sessionmaker:
    """Retorna fábrica de sessões configurada para o engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)
