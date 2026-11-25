"""Carregamento, saneamento e validação do dataset de notícias classificadas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd

from src.config import DATA_PATH, VALID_LABELS

REQUIRED_COLUMNS = ("titulo", "texto", "rotulo")


class DatasetValidationError(ValueError):
    """Erro lançado quando o dataset falha nas verificações de integridade."""


@dataclass
class DatasetReport:
    """Resumo estruturado da validação ou limpeza de um dataset."""

    total_rows: int
    valid_rows: int
    duplicates_removed: int = 0
    labels_stripped: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DatasetValidationError(f"Colunas obrigatórias ausentes: {missing}")


def validate_dataset(df: pd.DataFrame) -> DatasetReport:
    """Valida integridade do DataFrame sem modificar seus dados."""
    _check_required_columns(df)
    issues: List[str] = []

    for col in REQUIRED_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            issues.append(f"Coluna '{col}' possui {null_count} valor(es) nulo(s).")

    non_null = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    for col in REQUIRED_COLUMNS:
        str_col = non_null[col].astype(str)
        empty_count = int((str_col.str.strip() == "").sum())
        if empty_count > 0:
            issues.append(f"Coluna '{col}' possui {empty_count} string(s) vazia(s).")

    raw_labels = non_null["rotulo"].astype(str)
    unstripped_count = int((raw_labels != raw_labels.str.strip()).sum())
    if unstripped_count > 0:
        issues.append(
            f"Coluna 'rotulo' possui {unstripped_count} valor(es) com espaços nas bordas."
        )

    dup_count = int(non_null.duplicated(subset=["titulo", "texto"]).sum())
    if dup_count > 0:
        issues.append(f"Dataset possui {dup_count} linha(s) duplicada(s) em titulo+texto.")

    observed_labels = set(raw_labels.unique())
    invalid_labels = sorted(observed_labels - set(VALID_LABELS))
    if invalid_labels:
        issues.append(f"Rótulos inválidos encontrados: {invalid_labels}")

    missing_labels = sorted(set(VALID_LABELS) - observed_labels)
    if missing_labels:
        issues.append(f"Classes esperadas ausentes no dataset: {missing_labels}")

    class_counts = {str(k): int(v) for k, v in raw_labels.value_counts().to_dict().items()}
    return DatasetReport(
        total_rows=len(df),
        valid_rows=len(non_null),
        class_counts=class_counts,
        issues=issues,
    )


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, DatasetReport]:
    """Limpa espaços em branco, remove linhas vazias/duplicadas e valida o resultado."""
    _check_required_columns(df)
    cleaned = df[list(REQUIRED_COLUMNS)].copy()

    # Descarta linhas completamente vazias antes de converter para string.
    cleaned = cleaned.dropna(how="all")

    raw_labels = cleaned["rotulo"].fillna("").astype(str)
    labels_stripped = int((raw_labels != raw_labels.str.strip()).sum())

    for col in REQUIRED_COLUMNS:
        cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    cleaned = cleaned[
        (cleaned["titulo"] != "")
        & (cleaned["texto"] != "")
        & (cleaned["rotulo"] != "")
    ]

    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["titulo", "texto"], keep="first")
    duplicates_removed = before_dedup - len(cleaned)
    cleaned = cleaned.reset_index(drop=True)

    validation = validate_dataset(cleaned)
    if not validation.is_valid:
        raise DatasetValidationError("; ".join(validation.issues))

    report = DatasetReport(
        total_rows=len(df),
        valid_rows=len(cleaned),
        duplicates_removed=duplicates_removed,
        labels_stripped=labels_stripped,
        class_counts=validation.class_counts,
        issues=[],
    )
    return cleaned, report


def load_dataset(
    path: Union[str, Path] = DATA_PATH,
    require_valid: bool = True,
) -> pd.DataFrame:
    """Lê o CSV de notícias e opcionalmente exige validação estrita."""
    df = pd.read_csv(path, encoding="utf-8")
    if require_valid:
        report = validate_dataset(df)
        if not report.is_valid:
            raise DatasetValidationError("; ".join(report.issues))
    return df
