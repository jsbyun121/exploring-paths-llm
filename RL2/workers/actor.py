from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM
from RL2.workers import Worker
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.sequence_parallelism import sequence_parallelism_manager
from RL2.utils.functions import (
    compute_logsumexp,
    gather_action_logits,
    compute_entropy,
    aggregate_values
)
from RL2.utils.algorithms import compute_approx_kl
from RL2.utils.path_losses import compute_path_loss
from RL2.utils.offloading import model_offloading_manager
from RL2.utils.checkpointing import get_state_dict
from RL2.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_reduce,
    rank0_log
)
import math


class Actor(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)
        
        if config.use_liger_kernel:
            assert config.tp_size == 1, \
                "Liger kernel is not compatible with tensor parallelism."
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model_cls = AutoLigerKernelForCausalLM
        else:
            model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        self.prepare_model_optimizer()

    def compute_entropy(self, logits, logsumexp):
        """Compute entropy with gradients only when it affects the loss.

        Entropy is always reported as a metric, but most control runs use an
        exact zero entropy coefficient. Keeping that metric in the autograd
        graph retains a full-vocabulary probability tensor and makes backward
        traverse a mathematically zero-valued branch.
        """

        if self.config.entropy.coef != 0:
            return compute_entropy(logits, logsumexp, self.device_mesh["tp"])
        with torch.no_grad():
            return compute_entropy(logits, logsumexp, self.device_mesh["tp"])

    @sequence_parallelism_manager
    def forward_original(self, minibatch, return_entropy=False):

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.to(torch.float32) / getattr(
            self.config, "temperature", 1.0
        )
        # bfloat16 is unstable for the subsequent `logsumexp` operation.
        # See https://github.com/OpenRLHF/OpenRLHF/pull/634.
        
        logsumexp = compute_logsumexp(logits, self.device_mesh["tp"])
        action_logits = gather_action_logits(
            logits,
            minibatch["actions"],
            self.device_mesh["tp"]
        )
        logps = (action_logits - logsumexp) * minibatch["action_mask"]
        
        if return_entropy:
            entropy = self.compute_entropy(logits, logsumexp) * minibatch[
                "action_mask"
            ]
            return logps, entropy
        else:
            return logps

    @sequence_parallelism_manager
    def forward(self, minibatch, return_entropy=False, return_max=False):

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False
        ).logits.to(torch.float32) / getattr(
            self.config, "temperature", 1.0
        )
        # bfloat16 is unstable for the subsequent `logsumexp` operation.
        # See https://github.com/OpenRLHF/OpenRLHF/pull/634.

        logsumexp = compute_logsumexp(logits, self.device_mesh["tp"])
        action_logits = gather_action_logits(
            logits,
            minibatch["actions"],
            self.device_mesh["tp"]
        )
        logps = (action_logits - logsumexp) * minibatch["action_mask"]

        logits_for_others = logits.clone()
        action_indices = minibatch["actions"].unsqueeze(-1)
        logits_for_others.scatter_(-1, action_indices, float('-inf'))

        logsumexp_others = compute_logsumexp(logits_for_others, self.device_mesh["tp"])
        log1mp = (logsumexp_others - logsumexp) * minibatch["action_mask"]

        max_logps = None
        max_log1mp = None
        if return_max:
            max_indices = torch.argmax(logits, dim=-1)

            max_logits = gather_action_logits(
                logits,
                max_indices,
                self.device_mesh["tp"]
            )
            max_logps = (max_logits - logsumexp) * minibatch["action_mask"]

            # Reuse logits_for_others clone to save memory instead of creating new clone
            # Create a separate copy only for max token computation
            max_logits_for_others = logits.clone()
            max_indices_expanded = max_indices.unsqueeze(-1)
            max_logits_for_others.scatter_(-1, max_indices_expanded, float('-inf'))

            max_logsumexp_others = compute_logsumexp(max_logits_for_others, self.device_mesh["tp"])
            max_log1mp = (max_logsumexp_others - logsumexp) * minibatch["action_mask"]

        entropy = None
        if return_entropy:
            entropy = self.compute_entropy(logits, logsumexp) * minibatch[
                "action_mask"
            ]
            
        return log1mp, logps, max_logps, max_log1mp, entropy

    @sequence_parallelism_manager
    def forward_path(self, minibatch):
        """Compute a configured successful-path objective without logit clones."""

        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False,
        ).logits.to(torch.float32) / getattr(self.config, "temperature", 1.0)

        logsumexp = compute_logsumexp(logits, self.device_mesh["tp"])
        action_logits = gather_action_logits(
            logits, minibatch["actions"], self.device_mesh["tp"]
        )
        action_logps = action_logits - logsumexp
        max_logits = logits.max(dim=-1).values
        max_logps = max_logits - logsumexp
        entropy = self.compute_entropy(logits, logsumexp)

        losses, ranks, in_cap = compute_path_loss(
            self.config.path.objective,
            logits=logits,
            logsumexp=logsumexp,
            actions=minibatch["actions"],
            action_logps=action_logps,
            max_logps=max_logps,
            rank_cap=self.config.path.rank_cap,
        )
        mask = minibatch["action_mask"]
        return (
            losses * mask,
            entropy * mask,
            ranks * mask,
            in_cap.to(mask.dtype) * mask,
        )

    @time_logger("compute_logps")
    @model_offloading_manager
    @torch.no_grad()
    @data_manager(gather=True)
    def compute_logps(self, minibatches, step):
        prefix = "old" if self.train else "ref"

        self.model.eval()
        for minibatch in progress_bar(
            minibatches, desc=f"Compute {prefix} logps"
        ):
            minibatch[f"{prefix}_logps"] = self.forward_original(minibatch)
        
        return minibatches

    @time_logger("compute_logps")
    @model_offloading_manager
    @torch.no_grad()
    @data_manager(gather=True)
    def compute_probs(self, minibatches, step):
        prefix = "old" if self.train else "ref"

        self.model.eval()
        for minibatch in progress_bar(
            minibatches, desc=f"Compute {prefix} logps"
        ):
            log1mp, logps, max_logps, max_log1mp, _ = self.forward(minibatch, return_max=True)

            minibatch[f"{prefix}_log1mp"] = log1mp
            minibatch[f"{prefix}_logps"] = logps
            minibatch[f"{prefix}_max_logps"] = max_logps
            minibatch[f"{prefix}_max_log1mp"] = max_log1mp

        return minibatches

    @time_logger("update_actor")
    @model_offloading_manager
    @data_manager(pack_minibatches=True)
    def update_original(self, batches, step: int):
        if step < self.config.freeze_steps:
            self.state_dict = get_state_dict(self)
            return

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for batch in batches:
            
            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                self.device_mesh["dp"]
            )
            metric = defaultdict(list)
            for minibatch in batch:

                logps, entropy = self.forward_original(
                    minibatch, return_entropy=True
                )
                ratio = torch.exp(
                    logps - minibatch.get("old_logps", logps.detach())
                )
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip, 1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                losses = - torch.min(objective, clipped_objective)
                clip_ratios = objective > clipped_objective

                if self.config.tis_coef > 0:
                    # https://fengyao.notion.site/off-policy-rl
                    tis = torch.exp(
                        logps.detach() - minibatch["llm_logps"]
                    ).clamp(max=self.config.tis_coef)
                    losses *= tis
                    
                loss, clip_ratio, entropy = aggregate_values(
                    (losses, clip_ratios, entropy),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )
                if self.config.entropy.coef != 0:
                    loss = loss - self.config.entropy.coef * entropy
                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_approx_kl(
                        logps,
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                self.backward(loss)

                tbar.update()
                metric["actor/entropy"].append(entropy.item())
                metric["actor/loss"].append(loss.item())
                metric["actor/clip_ratio"].append(clip_ratio.item())

            grad_norm = self.optimizer_step()

            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"])
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)

        self.state_dict = get_state_dict(self)

    
    @time_logger("update_actor")
    @model_offloading_manager
    @data_manager(pack_minibatches=True)
    def update_0p5(self, batches, step: int):
        if step < self.config.freeze_steps:
            self.state_dict = get_state_dict(self)
            return

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                self.device_mesh["dp"]
            )
            metric = defaultdict(list)

            # Track total correct sequences for proper gradient scaling
            total_correct_sequences = 0
            eps = torch.finfo(torch.float32).eps

            for minibatch in batch:

                log1mp, logps, _, _, entropy = self.forward(
                    minibatch, return_entropy=True, return_max=True
                )

                log_05 = math.log(0.5)
                original_losses = - 0.5 * (logps + log1mp) + log_05

                losses = torch.where(logps > log_05, original_losses, 0)
                losses = losses * minibatch["action_mask"]  # Zero out masked positions

                # Debug: Token-level statistics
                valid_mask = minibatch["action_mask"] > 0
                valid_losses = losses[valid_mask]
                valid_advs = minibatch['advantages'][valid_mask]

                if valid_losses.numel() > 0:
                    print(f"[DEBUG] losses stats (masked): min={valid_losses.min().item():.2f}, max={valid_losses.max().item():.2f}, mean={valid_losses.mean().item():.2f}")
                    print(f"[DEBUG] advantages stats (masked): min={valid_advs.min().item():.2f}, max={valid_advs.max().item():.2f}, mean={valid_advs.mean().item():.2f}")

                # Filter correct sequences (positive total advantage)
                correct_mask = minibatch["advantages"].sum(-1) > 0
                num_correct = correct_mask.sum().item()

                print(f"[DEBUG] num correct seqs: {num_correct}/{len(minibatch['advantages'])}")

                if num_correct > 0:
                    if self.config.tis_coef > 0:
                        # https://fengyao.notion.site/off-policy-rl
                        tis = torch.exp(
                            logps.detach() - minibatch["llm_logps"]
                        ).clamp(max=self.config.tis_coef)
                        losses *= tis

                    # Compute per-sequence average loss and entropy
                    per_seq_loss = losses.sum(-1) / (minibatch["action_mask"].sum(-1) + eps)
                    per_seq_entropy = entropy.sum(-1) / (minibatch["action_mask"].sum(-1) + eps)

                    # Sum (not mean) for correct sequences - will be normalized later
                    minibatch_loss_sum = per_seq_loss[correct_mask].sum()
                    minibatch_entropy_sum = per_seq_entropy[correct_mask].sum()

                    # Backward on sum (releases computation graph immediately)
                    final_loss = minibatch_loss_sum - self.config.entropy.coef * minibatch_entropy_sum

                    if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                        # Note: KL term would need to be accumulated similarly if used
                        pass

                    self.backward(final_loss)

                    # Track sums for later averaging
                    metric["actor/entropy_sum"].append(minibatch_entropy_sum.item())
                    metric["actor/loss_sum"].append(minibatch_loss_sum.item())
                    total_correct_sequences += num_correct

                tbar.update()

            # Scale gradients by total correct sequences to get proper average
            if total_correct_sequences > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= total_correct_sequences

                print(f"[DEBUG] total correct seqs in batch: {total_correct_sequences}")

                # Compute average loss and entropy for logging
                total_loss_sum = sum(metric["actor/loss_sum"])
                total_entropy_sum = sum(metric["actor/entropy_sum"])
                avg_loss = total_loss_sum / total_correct_sequences
                avg_entropy = total_entropy_sum / total_correct_sequences

                # Log averages instead of sums
                metric["actor/loss"] = [avg_loss]
                metric["actor/entropy"] = [avg_entropy]
                del metric["actor/loss_sum"]
                del metric["actor/entropy_sum"]
            else:
                metric["actor/loss"] = [0.0]
                metric["actor/entropy"] = [0.0]

            grad_norm = self.optimizer_step()

            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"])
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)

        self.state_dict = get_state_dict(self)

    @time_logger("update_actor")
    @model_offloading_manager
    @data_manager(pack_minibatches=True)
    def update(self, batches, step: int):
        if step < self.config.freeze_steps:
            self.state_dict = get_state_dict(self)
            return

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]),
            desc="Update actor"
        )
        metrics = defaultdict(list)
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                self.device_mesh["dp"]
            )
            metric = defaultdict(list)

            # Track total correct sequences for proper gradient scaling
            total_correct_sequences = 0
            eps = torch.finfo(torch.float32).eps

            for minibatch in batch:

                log1mp, logps, max_logps, max_log1mp, entropy = self.forward(
                    minibatch, return_entropy=True, return_max=True
                )

                # Compute loss more memory-efficiently by computing each term separately
                # and avoiding intermediate tensor creation where possible
                losses = torch.exp(max_logps) * (max_logps - logps)
                losses.add_(torch.exp(max_log1mp) * (max_log1mp - log1mp))

                # Filter correct sequences (positive total advantage)
                correct_mask = minibatch["advantages"].sum(-1) > 0
                num_correct = correct_mask.sum().item()

                if num_correct > 0:
                    if self.config.tis_coef > 0:
                        # https://fengyao.notion.site/off-policy-rl
                        tis = torch.exp(
                            logps.detach() - minibatch["llm_logps"]
                        ).clamp(max=self.config.tis_coef)
                        losses *= tis

                    # Compute per-sequence average loss and entropy
                    per_seq_loss = losses.sum(-1) / (minibatch["action_mask"].sum(-1) + eps)
                    per_seq_entropy = entropy.sum(-1) / (minibatch["action_mask"].sum(-1) + eps)

                    # Sum (not mean) for correct sequences - will be normalized later
                    minibatch_loss_sum = per_seq_loss[correct_mask].sum()
                    minibatch_entropy_sum = per_seq_entropy[correct_mask].sum()

                    # Backward on sum (releases computation graph immediately)
                    final_loss = minibatch_loss_sum - self.config.entropy.coef * minibatch_entropy_sum

                    if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                        # Note: KL term would need to be accumulated similarly if used
                        pass

                    self.backward(final_loss)

                    # Track sums for later averaging
                    metric["actor/entropy_sum"].append(minibatch_entropy_sum.item())
                    metric["actor/loss_sum"].append(minibatch_loss_sum.item())
                    total_correct_sequences += num_correct

                tbar.update()

            # Scale gradients by total correct sequences to get proper average
            if total_correct_sequences > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= total_correct_sequences

                # Compute average loss and entropy for logging
                total_loss_sum = sum(metric["actor/loss_sum"])
                total_entropy_sum = sum(metric["actor/entropy_sum"])
                avg_loss = total_loss_sum / total_correct_sequences
                avg_entropy = total_entropy_sum / total_correct_sequences

                # Log averages instead of sums
                metric["actor/loss"] = [avg_loss]
                metric["actor/entropy"] = [avg_entropy]
                del metric["actor/loss_sum"]
                del metric["actor/entropy_sum"]
            else:
                metric["actor/loss"] = [0.0]
                metric["actor/entropy"] = [0.0]

            grad_norm = self.optimizer_step()

            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"])
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)

        self.state_dict = get_state_dict(self)

    @time_logger("update_actor")
    @model_offloading_manager
    @data_manager(pack_minibatches=True)
    def update_path(self, batches, step: int):
        """Optimize successful sequences with the configured path objective."""

        if step < self.config.freeze_steps:
            self.state_dict = get_state_dict(self)
            return
        if self.config.kl.coef > 0:
            raise NotImplementedError(
                "Reference-policy KL is not implemented for path objectives."
            )

        self.model.train()
        tbar = progress_bar(
            total=sum(len(batch) for batch in batches), desc="Update actor"
        )
        metrics = defaultdict(list)
        eps = torch.finfo(torch.float32).eps

        for batch in batches:
            _, total_sequences = count_total(
                batch, ("action_mask", "eos_mask"), self.device_mesh["dp"]
            )
            metric = defaultdict(list)
            total_correct_sequences = 0
            total_correct_tokens = 0
            total_selected_tokens = 0
            total_rank1_tokens = 0
            total_in_cap_tokens = 0
            total_rank_sum = 0.0
            total_rank_count = 0
            total_entropy_all_sum = 0.0
            total_entropy_all_sequences = 0

            for minibatch in batch:
                losses, entropy, ranks, in_cap = self.forward_path(minibatch)
                action_mask = minibatch["action_mask"].bool()
                correct_mask = minibatch["advantages"].sum(-1) > 0
                num_correct = int(correct_mask.sum().item())
                per_seq_entropy_all = entropy.sum(-1) / (
                    minibatch["action_mask"].sum(-1) + eps
                )
                valid_sequences = minibatch["eos_mask"].sum(-1) > 0
                total_entropy_all_sum += (
                    per_seq_entropy_all[valid_sequences].detach().sum().item()
                )
                total_entropy_all_sequences += int(valid_sequences.sum().item())

                if num_correct > 0:
                    correct_token_mask = action_mask & correct_mask.unsqueeze(-1)
                    selected_token_mask = correct_token_mask

                    entropy_quantile = self.config.path.token_entropy_quantile
                    if entropy_quantile > 0:
                        threshold = torch.quantile(
                            entropy[correct_token_mask].detach().to(torch.float32),
                            entropy_quantile,
                        )
                        selected_token_mask = correct_token_mask & (
                            entropy.detach() >= threshold
                        )
                        losses = losses * selected_token_mask

                    if self.config.tis_coef > 0:
                        tis = torch.exp(
                            minibatch.get("old_logps", minibatch["llm_logps"])
                            - minibatch["llm_logps"]
                        ).clamp(max=self.config.tis_coef)
                        losses = losses * tis

                    token_denominator = (
                        selected_token_mask.sum(-1)
                        if self.config.path.normalize_selected_tokens
                        else minibatch["action_mask"].sum(-1)
                    )
                    per_seq_loss = losses.sum(-1) / (token_denominator + eps)
                    per_seq_entropy = entropy.sum(-1) / (
                        minibatch["action_mask"].sum(-1) + eps
                    )

                    minibatch_loss_sum = per_seq_loss[correct_mask].sum()
                    minibatch_entropy_sum = per_seq_entropy[correct_mask].sum()
                    final_loss = (
                        minibatch_loss_sum
                        - self.config.entropy.coef * minibatch_entropy_sum
                    )
                    self.backward(final_loss)

                    metric["actor/loss_sum"].append(minibatch_loss_sum.item())
                    metric["actor/entropy_sum"].append(
                        minibatch_entropy_sum.item()
                    )
                    total_correct_sequences += num_correct

                    correct_tokens = int(correct_token_mask.sum().item())
                    selected_tokens = int(selected_token_mask.sum().item())
                    total_correct_tokens += correct_tokens
                    total_selected_tokens += selected_tokens
                    if self.config.path.objective == "rank_jsd":
                        rank1 = correct_token_mask & ranks.eq(1)
                        covered = correct_token_mask & in_cap.bool()
                        promoted = covered & ranks.gt(1)
                        total_rank1_tokens += int(rank1.sum().item())
                        total_in_cap_tokens += int(covered.sum().item())
                        total_rank_sum += ranks[promoted].sum().item()
                        total_rank_count += int(promoted.sum().item())

                tbar.update()

            if total_correct_sequences > 0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= total_correct_sequences

                metric["actor/loss"] = [
                    sum(metric.pop("actor/loss_sum"))
                    / total_correct_sequences
                ]
                metric["actor/entropy"] = [
                    sum(metric.pop("actor/entropy_sum"))
                    / total_correct_sequences
                ]
            else:
                metric["actor/loss"] = [0.0]
                metric["actor/entropy"] = [0.0]

            metric["path/correct_sequence_fraction"] = [
                total_correct_sequences / max(total_sequences, 1)
            ]
            metric["path/selected_token_fraction"] = [
                total_selected_tokens / max(total_correct_tokens, 1)
            ]
            metric["actor/entropy_all"] = [
                total_entropy_all_sum / max(total_entropy_all_sequences, 1)
            ]
            if self.config.path.objective == "rank_jsd":
                metric["path/rank1_fraction"] = [
                    total_rank1_tokens / max(total_correct_tokens, 1)
                ]
                metric["path/rank_cap_coverage"] = [
                    total_in_cap_tokens / max(total_correct_tokens, 1)
                ]
                metric["path/promoted_mean_rank"] = [
                    total_rank_sum / max(total_rank_count, 1)
                ]

            grad_norm = self.optimizer_step()
            for key, values in metric.items():
                metrics[key].append(
                    gather_and_reduce(values, self.device_mesh["dp"])
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        self.state_dict = get_state_dict(self)
