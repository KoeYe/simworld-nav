"""Memory-efficient log-probs at selected positions.

The obvious implementation — ``log_softmax(model(**inputs).logits)`` —
materializes a ``[sequence, vocab]`` float32 tensor. For Qwen3-VL that vocab is
151,936, so a 1,500-token multimodal rollout needs about 0.9 GB for the logits
and another 0.9 GB for the log-softmax, before autograd keeps both for the
backward pass. That OOMs a 44 GB card on a 4B model, which is absurd: the loss
only ever touches the handful of positions where the policy sampled a token.

So the head is applied *after* selection. The inner model returns hidden states
(``[sequence, 2560]``, a few megabytes), the rows of interest are gathered, and
``lm_head`` runs on those rows alone, producing ``[selected, vocab]``. For a
20-token response that is 12 MB rather than 1.8 GB.

This is not an approximation. It computes exactly the same numbers.
"""

from __future__ import annotations

from typing import Any

import torch


def hidden_states(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    """Last-layer hidden states, without ever building the full logits tensor."""
    output = model.model(**inputs)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = output[0]
    return hidden[0]


def selected_token_logprobs(
    model: Any,
    inputs: dict[str, Any],
    positions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Log-probs of ``targets`` predicted at ``positions``.

    ``positions`` are the indices of the *predicted* tokens; the prediction for
    position ``i`` comes from the hidden state at ``i - 1``, so callers must pass
    positions greater than zero.
    """
    if positions.numel() == 0:
        raise ValueError("no positions selected")
    if int(positions.min()) < 1:
        raise ValueError("position 0 has no predecessor to predict it")

    hidden = hidden_states(model, inputs)
    rows = hidden.index_select(0, positions - 1)
    logits = model.lm_head(rows).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def build_model_inputs(
    sample_or_state: Any, *, device: str, dtype: Any, input_ids: list[int] | None = None
) -> dict[str, Any]:
    """Assemble the model kwargs shared by rollout, scoring, and training."""
    ids = input_ids if input_ids is not None else list(sample_or_state.input_ids)
    tensor = torch.tensor([ids], device=device)
    inputs: dict[str, Any] = {
        "input_ids": tensor,
        "attention_mask": torch.ones_like(tensor),
    }
    pixel_values = getattr(sample_or_state, "pixel_values", None)
    grid = getattr(sample_or_state, "image_grid_thw", None)
    if pixel_values:
        inputs["pixel_values"] = torch.cat([p.to(device, dtype) for p in pixel_values], dim=0)
        inputs["image_grid_thw"] = torch.cat([g.to(device) for g in grid], dim=0)
    return inputs
