"""Canonicalização de URL, hash de conteúdo e extração de palavras-chave."""

import hashlib
from typing import Dict, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.config import DEFAULT_KEYWORDS
from src.nlp.preprocessing import clean_text, combine_title_and_text

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "igshid",
        "cmpid",
        "yclid",
    }
)


def canonicalize_url(url: str) -> str:
    """Normaliza URL removendo fragmentos, portas padrão e parâmetros de rastreamento."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if not parsed.scheme and not parsed.netloc:
        return raw

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if (scheme == "https" and netloc.endswith(":443")) or (
        scheme == "http" and netloc.endswith(":80")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    filtered_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
        and not any(k.lower().startswith(pref) for pref in _TRACKING_PARAM_PREFIXES)
    ]
    filtered_params.sort()
    query = urlencode(filtered_params)

    return urlunparse((scheme, netloc, path, "", query, ""))


def compute_content_hash(title: str, text: str) -> str:
    """Calcula SHA-256 sobre título e texto normalizados para deduplicação semântica."""
    normalized = f"{clean_text(title)}|{clean_text(text)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_keyword_counts(
    title: str,
    text: str,
    keywords: Sequence[str] = DEFAULT_KEYWORDS,
) -> Dict[str, int]:
    """Conta ocorrências das palavras-chave ambientais presentes no texto normalizado."""
    content = combine_title_and_text(title, text)
    counts: Dict[str, int] = {}
    for kw in keywords:
        cnt = content.count(kw.lower())
        if cnt > 0:
            counts[kw] = cnt
    return counts
