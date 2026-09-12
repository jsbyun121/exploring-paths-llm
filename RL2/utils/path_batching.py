"""Padding-aware batching for single-device, positive-path training."""


def length_bucketed_partitions(lengths, token_budget):
    """Keep every sequence once, bounding the *padded* tokens per batch."""
    if not lengths:
        return []
    if token_budget < 1 or min(lengths) < 1 or max(lengths) > token_budget:
        raise ValueError("Each sequence must fit the positive token budget")
    batches, batch, longest = [], [], 0
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        if batch and longest * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch = []
        if not batch:
            longest = lengths[index]
        batch.append(index)
    if batch:
        batches.append(batch)
    return batches


def trim_right_padding(minibatch):
    """Remove only columns after the last real token in this minibatch."""
    length = int((minibatch["eos_mask"].argmax(-1) + 1).max().item())
    return {key: value[:, :length].contiguous() for key, value in minibatch.items()}


def causal_position_ids(minibatch):
    """Continue positions through right padding for independent causal rows.

    Transformers 5 interprets zero-reset padding positions as packed sequences
    and constructs a dense block mask. Continuing padded positions preserves
    every real token's position and allows the ordinary causal attention path.
    Use only for un-packed rows produced by get_tensor_dict (SP=1).
    """
    import torch
    return torch.arange(minibatch["states"].shape[1], device=minibatch["states"].device).expand_as(minibatch["states"])
