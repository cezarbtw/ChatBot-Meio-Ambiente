"""Configurações centrais e caminhos padrão do projeto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "noticias_classificadas.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_BERT_MODEL = "neuralmind/bert-large-portuguese-cased"
DEFAULT_MAX_LENGTH = 128

VALID_LABELS = ("Boa", "Irrelevante", "Ruim")
DEFAULT_KEYWORDS = (
    "incêndios",
    "desmatamento",
    "aquecimento",
    "temperatura",
    "poluição",
)

MLP_FILENAME = "mlp_classifier.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
METADATA_FILENAME = "model_metadata.json"
