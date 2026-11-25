"""Funções compartilhadas de limpeza e normalização textual."""

import re
from typing import Any

_URL_PATTERN = re.compile(r"http\S+|www\S+", re.IGNORECASE)
_HTML_PATTERN = re.compile(r"<.*?>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """Remove URLs, tags HTML, normaliza espaços e converte para minúsculas."""
    if not isinstance(text, str):
        return ""
    cleaned = _URL_PATTERN.sub("", text)
    cleaned = _HTML_PATTERN.sub("", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned.lower()


def combine_title_and_text(title: Any, text: Any) -> str:
    """Normaliza e concatena título e corpo da notícia no formato esperado pelo modelo."""
    clean_title = clean_text(title)
    clean_body = clean_text(text)
    if clean_title and clean_body:
        return f"{clean_title}. {clean_body}"
    return clean_title or clean_body
