"""Subpacote de Processamento de Linguagem Natural e classificação."""

from src.nlp.classifier import NewsClassifier, PredictionResult, train_classifier
from src.nlp.embedder import BertEmbedder, TransformerEmbedder
from src.nlp.preprocessing import clean_text, combine_title_and_text

__all__ = [
    "BertEmbedder",
    "NewsClassifier",
    "PredictionResult",
    "TransformerEmbedder",
    "clean_text",
    "combine_title_and_text",
    "train_classifier",
]
