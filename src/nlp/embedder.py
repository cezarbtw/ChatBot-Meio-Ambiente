"""Extração de embeddings com Transformers pré-treinados (BERT, RoBERTa, etc.)."""

from typing import Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import DEFAULT_BERT_MODEL, DEFAULT_MAX_LENGTH, DEFAULT_POOLING, VALID_POOLINGS


class BertEmbedder:
    """Carrega modelo Transformer sob demanda e extrai embeddings [CLS] ou mean pooling."""

    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
        pooling: str = DEFAULT_POOLING,
        device: Optional[str] = None,
    ) -> None:
        if pooling not in VALID_POOLINGS:
            raise ValueError(
                f"Pooling inválido '{pooling}'. Valores aceitos: {VALID_POOLINGS}"
            )
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def _pool_outputs(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]

        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

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
            pooled = self._pool_outputs(
                outputs.last_hidden_state, inputs["attention_mask"]
            )
            batches.append(pooled.cpu().numpy())

        return np.vstack(batches)

    def embed_one(self, text: str) -> np.ndarray:
        """Gera vetor 2D (1, hidden_size) para um único texto."""
        return self.embed_batch([text], batch_size=1)


TransformerEmbedder = BertEmbedder
