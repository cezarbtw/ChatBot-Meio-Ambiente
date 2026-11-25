"""Testes para validação e higienização do dataset de notícias."""

import pandas as pd
import pytest

from src.config import DATA_PATH, VALID_LABELS
from src.nlp.dataset import (
    DatasetValidationError,
    clean_dataset,
    load_dataset,
    validate_dataset,
)


def test_clean_dataset_strips_labels_and_removes_duplicates():
    df = pd.DataFrame(
        [
            {"titulo": "Notícia A", "texto": "Texto A", "rotulo": "Boa "},
            {"titulo": "Notícia A", "texto": "Texto A", "rotulo": "Boa"},
            {"titulo": "Notícia B", "texto": "Texto B", "rotulo": " Ruim"},
            {"titulo": "Notícia C", "texto": "Texto C", "rotulo": "Irrelevante"},
            {"titulo": None, "texto": None, "rotulo": None},
        ]
    )

    cleaned, report = clean_dataset(df)
    assert report.is_valid
    assert report.valid_rows == 3
    assert report.duplicates_removed == 1
    assert report.labels_stripped == 2
    assert set(cleaned["rotulo"].unique()) == set(VALID_LABELS)


def test_validate_dataset_flags_trailing_whitespace_and_unknown_label():
    df = pd.DataFrame(
        [
            {"titulo": "T1", "texto": "B1", "rotulo": "Boa "},
            {"titulo": "T2", "texto": "B2", "rotulo": "Ruim"},
            {"titulo": "T3", "texto": "B3", "rotulo": "Desconhecido"},
        ]
    )

    report = validate_dataset(df)
    assert not report.is_valid
    assert any("espaços nas bordas" in issue for issue in report.issues)
    assert any("Rótulos inválidos" in issue for issue in report.issues)


def test_validate_dataset_raises_on_missing_columns():
    df = pd.DataFrame([{"titulo": "T1", "texto": "B1"}])
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)


def test_repository_csv_is_valid_and_has_three_classes():
    df = load_dataset(DATA_PATH, require_valid=True)
    assert len(df) > 0
    assert df.duplicated(subset=["titulo", "texto"]).sum() == 0
    assert set(df["rotulo"].unique()) == set(VALID_LABELS)
