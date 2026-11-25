"""Extração de embeddings com modelo BERT pré-treinado (sem duplicação de pesos)."""

from typing import Optional, Sequence

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from src.config import DEFAULT_BERT_MODEL, DEFAULT_MAX_LENGTH


class BertEmbedder:
    """Carrega BERT pré-treinado sob demanda e gera embeddings do token [CLS]."""

    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Optional[BertTokenizer] = None
        self._model: Optional[BertModel] = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = BertModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def embed_batch(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Gera matriz 2D (N, hidden_size) para uma sequência de textos."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        batches = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            batches.append(cls_embeddings)

        return np.vstack(batches)

    def embed_one(self, text: str) -> np.ndarray:
        """Gera vetor 2D (1, hidden_size) para um único texto."""
        return self.embed_batch([text], batch_size=1)
