"""Configurações centrais e caminhos padrão do projeto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "noticias_classificadas.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_BERT_MODEL = "neuralmind/bert-large-portuguese-cased"
DEFAULT_MAX_LENGTH = 256
DEFAULT_POOLING = "cls"
VALID_POOLINGS = ("cls", "mean")

DEFAULT_CLASSIFIER_TYPE = "mlp"
VALID_CLASSIFIER_TYPES = ("mlp", "mlp_calibrated", "logreg", "svc")

DEFAULT_CV_SPLITS = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_MARGIN_THRESHOLD = 0.15

VALID_LABELS = ("Boa", "Irrelevante", "Ruim")
DEFAULT_KEYWORDS = (
    "incêndios",
    "desmatamento",
    "aquecimento",
    "temperatura",
    "poluição",
)
AMBIGUOUS_TERMS = (
    "aquecimento",
    "temperatura",
    "clima",
    "energia",
)

MLP_FILENAME = "mlp_classifier.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
METADATA_FILENAME = "model_metadata.json"
METRICS_FILENAME = "metrics_report.json"
BENCHMARK_FILENAME = "benchmark_report.json"
TELEMETRY_FILENAME = "network_telemetry.json"
