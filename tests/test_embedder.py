"""Testes unitários para o módulo de extração de embeddings."""

import pytest
import torch

from src.nlp.embedder import BertEmbedder


def test_invalid_pooling_raises_value_error():
    with pytest.raises(ValueError):
        BertEmbedder(pooling="max")


def test_pool_outputs_cls_and_masked_mean():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    cls_embedder = BertEmbedder(pooling="cls")
    cls_out = cls_embedder._pool_outputs(hidden, mask)
    assert torch.allclose(cls_out, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    mean_embedder = BertEmbedder(pooling="mean")
    mean_out = mean_embedder._pool_outputs(hidden, mask)
    expected = torch.tensor([[2.0, 3.0], [7.0, 8.0]])
    assert torch.allclose(mean_out, expected)
