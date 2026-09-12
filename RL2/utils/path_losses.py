"""Token-level objectives for successful on-policy reasoning paths.

The functions in this module are intentionally independent of the trainer so
that their numerical and gradient behaviour can be unit tested on CPU.
Targets derived from the current policy are detached by default.  Without the
stop-gradient, a purported target moves with the prediction and changes the
objective substantially.
"""

from __future__ import annotations

import math

import torch


def _log1mexp(log_p: torch.Tensor) -> torch.Tensor:
    """Stable log(1 - exp(log_p)) for log_p <= 0."""

    log_p = torch.clamp(log_p, max=-torch.finfo(log_p.dtype).eps)
    cutoff = -math.log(2.0)
    return torch.where(
        log_p < cutoff,
        torch.log1p(-torch.exp(log_p)),
        torch.log(-torch.expm1(log_p)),
    )


def positive_cross_entropy(action_logps: torch.Tensor) -> torch.Tensor:
    """Rejection-sampling/SFT control: maximize every successful action."""

    return -action_logps


def bernoulli_max_kl(
    action_logps: torch.Tensor,
    max_logps: torch.Tensor,
    *,
    detach_target: bool,
) -> torch.Tensor:
    """KL(B(p_max) || B(p_action)) used in Sections 3.1 and 3.2.

    ``detach_target=False`` reproduces the legacy implementation.  It is an
    important ablation, not the recommended formulation: gradients then move
    both the maximum-token probability and the chosen-token probability.
    """

    target_logps = max_logps.detach() if detach_target else max_logps
    target_log1mps = _log1mexp(target_logps)
    action_log1mps = _log1mexp(action_logps)
    target_probs = torch.exp(target_logps)
    target_other_probs = torch.exp(target_log1mps)
    return (
        target_probs * (target_logps - action_logps)
        + target_other_probs * (target_log1mps - action_log1mps)
    )


def fixed_half_kl(
    action_logps: torch.Tensor,
    *,
    legacy_high_probability_only: bool,
) -> torch.Tensor:
    """KL(B(0.5) || B(p_action)).

    The thesis repository applied this only when ``p_action > 0.5``.  That
    legacy mask prevents the loss from promoting low-probability successful
    actions, so both corrected and legacy versions are exposed for the audit.
    """

    log_half = -math.log(2.0)
    losses = -0.5 * (action_logps + _log1mexp(action_logps)) + log_half
    if legacy_high_probability_only:
        losses = torch.where(action_logps > log_half, losses, 0.0)
    return losses


def rank_shift_jsd(
    logits: torch.Tensor,
    logsumexp: torch.Tensor,
    actions: torch.Tensor,
    *,
    rank_cap: int,
    divergence: str = "jsd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """JSD (default) or forward KL to a detached rank-shifted target.

    For a chosen action at rank r, the target rotates the probabilities over
    ranks 1..r: ``[q1, q2, ..., qr] -> [q2, ..., qr, q1]`` on the tokens in
    their original order.  All lower-ranked probabilities remain unchanged
    and therefore contribute exactly zero to the JSD.

    Exact computation is O(r) only when the chosen action is within
    ``rank_cap``.  Actions below the cap are skipped and reported through the
    returned coverage mask; silently pretending they have rank ``rank_cap``
    would implement a different, non-mass-preserving target.

    Returns:
        losses: per-token JSD, zero for rank-1 and out-of-cap actions.
        ranks: exact 1-indexed rank in-cap, otherwise ``rank_cap + 1``.
        in_cap: whether the chosen action was present in the top-k set.
    """

    if rank_cap < 2:
        raise ValueError(f"rank_cap must be >= 2, got {rank_cap}")

    cap = min(rank_cap, logits.shape[-1])
    top_logits, top_indices = torch.topk(logits, k=cap, dim=-1)
    top_logps = top_logits - logsumexp.unsqueeze(-1)
    top_probs = torch.exp(top_logps)

    matches = top_indices.eq(actions.unsqueeze(-1))
    in_cap = matches.any(dim=-1)
    rank_zero_based = matches.to(torch.int64).argmax(dim=-1)
    ranks = torch.where(
        in_cap,
        rank_zero_based + 1,
        torch.full_like(rank_zero_based, cap + 1),
    )

    positions = torch.arange(cap, device=logits.device)
    view_shape = (1,) * rank_zero_based.ndim + (cap,)
    positions = positions.view(view_shape)
    rank_position = rank_zero_based.unsqueeze(-1)

    # The target is a snapshot.  This stop-gradient is the key distinction
    # between rank promotion and merely making two moving distributions meet.
    detached_probs = top_probs.detach()
    shifted_probs = torch.cat(
        (detached_probs[..., 1:], detached_probs[..., -1:]), dim=-1
    )
    target_probs = torch.where(
        positions < rank_position,
        shifted_probs,
        torch.where(
            positions == rank_position,
            detached_probs[..., :1],
            detached_probs,
        ),
    )

    # Out-of-cap actions have no exact target.  Make their forward loss and
    # gradient zero by setting the target equal to the current snapshot.
    target_probs = torch.where(
        in_cap.unsqueeze(-1), target_probs, detached_probs
    )

    tiny = torch.finfo(torch.float32).tiny
    q = top_probs.to(torch.float32)
    p = target_probs.to(torch.float32)
    log_q = top_logps.to(torch.float32)
    log_p = torch.log(p.clamp_min(tiny))
    mixture = 0.5 * (p + q)
    log_mixture = torch.log(mixture.clamp_min(tiny))
    if divergence == "jsd":
        terms = 0.5 * (
            p * (log_p - log_mixture) + q * (log_q - log_mixture)
        )
    elif divergence == "kl":
        # Generalized KL has the same value on this mass-preserving rotation.
        # The -p+q correction is essential: omitting unchanged tail terms
        # from ordinary KL would otherwise give the WRONG gradient.
        terms = p * (log_p - log_q) - p + q
    else:
        raise ValueError(f"Unknown rank divergence: {divergence}")

    active_support = positions <= rank_position
    active_support = active_support & in_cap.unsqueeze(-1)
    losses = (terms * active_support).sum(dim=-1)
    return losses, ranks, in_cap


def compute_path_loss(
    objective: str,
    *,
    logits: torch.Tensor,
    logsumexp: torch.Tensor,
    actions: torch.Tensor,
    action_logps: torch.Tensor,
    max_logps: torch.Tensor,
    rank_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch a configured path objective and return common diagnostics."""

    ranks = torch.zeros_like(actions)
    in_cap = torch.zeros_like(actions, dtype=torch.bool)

    if objective == "positive_ce":
        losses = positive_cross_entropy(action_logps)
    elif objective == "bernoulli_kl_legacy":
        losses = bernoulli_max_kl(
            action_logps, max_logps, detach_target=False
        )
    elif objective == "bernoulli_kl_detached":
        losses = bernoulli_max_kl(
            action_logps, max_logps, detach_target=True
        )
    elif objective == "fixed_half_legacy":
        losses = fixed_half_kl(
            action_logps, legacy_high_probability_only=True
        )
    elif objective == "fixed_half":
        losses = fixed_half_kl(
            action_logps, legacy_high_probability_only=False
        )
    elif objective in {"rank_jsd", "rank_kl"}:
        losses, ranks, in_cap = rank_shift_jsd(
            logits, logsumexp, actions, rank_cap=rank_cap,
            divergence="kl" if objective == "rank_kl" else "jsd",
        )
    else:
        raise ValueError(f"Unknown path objective: {objective}")

    return losses, ranks, in_cap
