from __future__ import annotations

import numpy as np
import torch


def test_prepare_action_prompt_keeps_aligned_mask() -> None:
    from deploy.tensorrt.common import EMPTY_TOKEN_ID, prepare_action_prompt

    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    new_ids, new_mask, appended = prepare_action_prompt(ids, mask)
    assert appended
    assert new_ids.tolist() == [[1, 2, EMPTY_TOKEN_ID]]
    assert new_mask is not None and new_mask.tolist() == [[1, 1, 1]]


def test_action_token_range_has_256_symbols() -> None:
    from deploy.tensorrt.common import action_token_bounds

    class Config:
        n_action_bins = 256

    class Model:
        vocab_size = 32000
        config = Config()

    low, high = action_token_bounds(Model())
    assert (low, high) == (31744, 31999)
    assert high - low + 1 == 256


def test_top_action_symbol_clips_to_last_of_255_centers() -> None:
    bins = np.linspace(-1, 1, 256)
    centers = (bins[:-1] + bins[1:]) / 2
    discrete = np.clip(32000 - np.array([31744]) - 1, 0, centers.shape[0] - 1)
    assert discrete.tolist() == [254]
