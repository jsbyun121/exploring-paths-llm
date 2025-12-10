# Changes from Original RL2 Library

This repository is based on [RL2 (Ray Less Reinforcement Learning)](https://github.com/ChenmienTan/RL2) by Chenmien Tan et al.

## Key Modifications for Research

### New Files

1. **`RL2/trainer/spo_0p5.py`**
   - Fixed soft target (0.5, 0.5) trainer implementation
   - Corresponds to Section 3.3 of the paper
   - Loss: `-0.5 * (log p + log(1-p)) + log(0.5)`

### Modified Files

1. **`RL2/workers/actor.py`**
   - **New `forward()` method**: Computes KL divergence components
     - Returns: `log1mp, logps, max_logps, max_log1mp, entropy`
     - Calculates log probabilities for chosen and max tokens
     - Computes complementary probabilities for binomial KL

   - **New `update()` method**: Main training loop with KL loss
     - Filters only correct sequences (reward > 0)
     - Loss: `exp(max_logps) * (max_logps - logps) + exp(max_log1mp) * (max_log1mp - log1mp)`
     - Gradient scaling by number of correct sequences

   - **New `update_0p5()` method**: Fixed 0.5 target training
     - Filters tokens where `logps > log(0.5)`
     - Loss: `-0.5 * (logps + log1mp) + log(0.5)`

   - **New `compute_probs()` method**: Computes all probability components for reference actor

   - **Renamed `forward()` to `forward_original()`**: Original PPO forward pass
   - **Renamed `update()` to `update_original()`**: Original PPO update

2. **`RL2/trainer/spo.py`**
   - Modified to use new advantage estimator `compute_spo_adv`
   - Calls `actor.update()` instead of `actor.update_original()`
   - Added `compute_probs()` for reference actor

3. **`RL2/trainer/ppo.py`**
   - Updated for Dr.GRPO baseline configuration
   - Uses `adv.estimator=reinforce` with 4 rollouts per prompt

4. **`RL2/utils/algorithms.py`**
   - **New function `compute_spo_adv()`**:
     - Simple advantage: `rewards * action_mask`
     - No baseline subtraction or normalization
     - Used in SPO trainer variants

5. **`RL2/trainer/config/spo.yaml`**
   - Updated default settings for SPO experiments
   - `adv.estimator: spo`
   - `actor.avg_level: sequence`

6. **`RL2/trainer/config/ppo.yaml`**
   - Updated for Dr.GRPO baseline
   - `adv.estimator: reinforce`
   - `adv.norm_var: false`

### New Experiment Scripts

All scripts in `examples/gsm8k_*.sh`:
- `gsm8k_reinforce.sh` - Dr.GRPO baseline
- `gsm8k_0.sh` - Section 3.1 (batch normalized)
- `gsm8k_0_1.sh` - Section 3.1 variant
- `gsm8k_0p5_0.sh` - Section 3.3 (fixed 0.5 target)
- `gsm8k_0p5_0_1.sh` - Section 3.3 variant
- `gsm8k_0p5_test.sh` - Testing script

### Environment

- **`envs/gsm8k.py`**: GSM8K answer extraction environment
  - Parses `</think>` tags for thinking models
  - Extracts answer after `####` marker
  - Binary reward based on exact match

## Mathematical Formulation

### KL Divergence Loss (Sections 3.1, 3.2)

The loss is based on KL divergence between two Bernoulli distributions:

**Target Distribution P**: `(p_max, 1 - p_max)`
**Model Distribution Q**: `(p_chosen, 1 - p_chosen)`

**Loss**:
```
L = KL(P || Q) = p_max * log(p_max / p_chosen) + (1 - p_max) * log((1 - p_max) / (1 - p_chosen))
  = exp(log_max) * (log_max - log_chosen) + exp(log1m_max) * (log1m_max - log1m_chosen)
```

### Fixed Target Loss (Section 3.3)

**Target Distribution**: `(0.5, 0.5)`
**Model Distribution**: `(p_chosen, 1 - p_chosen)`

**Loss**:
```
L = KL((0.5, 0.5) || (p, 1-p))
  = 0.5 * log(0.5 / p) + 0.5 * log(0.5 / (1-p))
  = -0.5 * (log p + log(1-p)) + log(0.5)
```

## Attribution

If you use this code, please cite:

1. **This research**:
```bibtex
@thesis{Byun2025ExploringPaths,
    author = {Junsoo Byun},
    title = {Exploring Paths in Probabilistic Graphs for Model Training},
    school = {Seoul National University},
    year = {2025}
}
```

2. **Original RL2 library**:
```bibtex
@misc{Tan2025RL2,
    author = {Chenmien Tan and others},
    title = {RL2: Ray Less Reinforcement Learning},
    howpublished = {\url{https://github.com/ChenmienTan/RL2}},
    year = {2025}
}
```
