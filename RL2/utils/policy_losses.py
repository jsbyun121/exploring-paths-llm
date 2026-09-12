"""Paper-defined critic-free policy objectives; caller supplies frozen old logps.
Dr.GRPO: arXiv:2503.20783; SAPO: arXiv:2511.20347v2 Eqs.5-6.
Normalization uses the full optimizer batch, not individual memory microbatches.
"""
import torch


def policy_loss(name, logps, old_logps, advantages, mask, *, total_sequences,
                max_completion_length, clip=0.2, tau_pos=1.0, tau_neg=1.05):
    if max_completion_length <= 0 or total_sequences <= 0:
        raise ValueError('Positive length and batch size required')
    if name not in {'dr_grpo', 'sapo'}:
        raise ValueError(f'Unknown paper policy loss: {name}')
    mask = mask.to(logps.dtype)
    delta = logps - old_logps.detach()
    # Fail explicitly rather than silently changing a paper objective on overflow.
    ratio = delta.exp()
    if not torch.isfinite(ratio[mask.bool()]).all():
        raise FloatingPointError('Nonfinite importance ratio')
    adv = advantages.detach()
    if name == 'dr_grpo':
        values = -torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
        return (values * mask).sum() / (total_sequences * max_completion_length)
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError('SAPO temperatures must be positive')
    tau = torch.where(adv > 0, tau_pos, tau_neg)
    values = -(4 / tau) * torch.sigmoid(tau * (ratio - 1)) * adv
    lengths = mask.sum(-1).clamp_min(1)
    return ((values * mask).sum(-1) / lengths).sum() / total_sequences
