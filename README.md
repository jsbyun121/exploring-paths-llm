# Exploring Paths in Probabilistic Graphs for Model Training

This repository contains the implementation and experimental code for the paper **"Exploring Paths in Probabilistic Graphs for Model Training"**.

## Overview

This work proposes a novel training algorithm that learns from diverse probabilistic reasoning paths by exclusively targeting correct states. The method applies an explicit token-level loss formulated as the expected Jensen-Shannon Divergence (JSD) or Kullback-Leibler (KL) divergence between the selected correct token's probability and the maximum probability token at each step.

**Key Results on GSM8K:**
- Consistent improvements in both accuracy and entropy
- Distinct training phases: exploration → consolidation → (potential collapse)
- More token-efficient reasoning compared to baseline Dr.GRPO

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Experiments](#experiments)
  - [Baseline: Dr.GRPO](#baseline-drgrpo)
  - [Section 3.1: KL Divergence with Max-Probability Target (Batch Normalized)](#section-31-kl-divergence-with-max-probability-target-batch-normalized)
  - [Section 3.2: KL Divergence with Max-Probability Target (Correct-Only Normalized)](#section-32-kl-divergence-with-max-probability-target-correct-only-normalized)
  - [Section 3.3: Fixed Soft Target (0.5, 0.5)](#section-33-fixed-soft-target-05-05)
- [Monitoring and Analysis](#monitoring-and-analysis)
- [Understanding the Results](#understanding-the-results)
- [Citation](#citation)

---

## Installation

### Prerequisites
- Python ≥ 3.12
- CUDA-capable GPU (tested on single GPU, scalable to multi-GPU)
- [uv](https://github.com/astral-sh/uv) package manager

### Step 1: Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### Step 2: Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/jsbyun121/exploring-paths-llm.git
cd exploring-paths-llm

# Create virtual environment and install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

**Note:** The installation includes:
- PyTorch 2.8.0 with Flash Attention 2
- Transformers 4.56.1
- SGLang 0.5.2 for async inference
- Additional dependencies: hydra-core, wandb, liger-kernel, etc.

---

## Dataset Preparation

The experiments use the **GSM8K dataset** from Hugging Face, which is automatically downloaded during training. The dataset format follows the RL2 convention:

```json
{
    "messages": [
        {"role": "user", "content": "What is 25 + 17?"}
    ],
    "extra_info": {
        "answer": "42"
    }
}
```

The answer extraction logic is implemented in `envs/gsm8k.py`:
- Extracts text after `</think>` tag (for thinking models)
- Parses answer after `####` marker
- Compares with ground truth for reward calculation

**No manual dataset preparation is required** - the training scripts automatically fetch `train@openai/gsm8k:main` and `test@openai/gsm8k:main`.

---

## Experiments

All experiments use the following **common hyperparameters**:
- **Base Model:** `Qwen/Qwen3-4B-Thinking-2507`
- **Batch Size:** 512 sequences per batch
- **Sampling Temperature:** 1.0 (training), 0.0 (evaluation/greedy)
- **Max New Tokens:** 8192
- **Number of Epochs:** 150
- **Test Frequency:** Every 5 steps
- **Checkpoint Frequency:** Every 20 steps
- **Learning Rate:** 1e-6 (default from config)
- **Environment:** `envs/gsm8k.py`

### Baseline: Dr.GRPO

**Paper Reference:** Baseline comparison (Figures 2, 4, 8, 12)

**Configuration:**
- Trainer: `RL2.trainer.ppo`
- Advantage Estimator: `reinforce`
- Prompts per rollout: 128
- Responses per prompt: 4
- **Total sequences per batch:** 128 × 4 = **512** (matched with proposed methods)

**Run Command:**
```bash
torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.ppo \
    train_data.path=train@openai/gsm8k:main \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=4 \
    test_data.path=test@openai/gsm8k:main \
    test_data.prompts_per_rollout=128 \
    test_data.responses_per_prompt=4 \
    actor.model_name=Qwen/Qwen3-4B-Thinking-2507 \
    actor.use_liger_kernel=true \
    actor.max_length_per_device=8576 \
    adv.estimator=reinforce \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/gsm8k.py \
    rollout.gpu_memory_utilization=0.3 \
    trainer.project=GSM8K \
    trainer.experiment_name=baseline_drgrpo \
    trainer.n_epochs=150 \
    trainer.save_freq=20 \
    trainer.test_freq=5
```

Or use the provided script:
```bash
bash examples/gsm8k_reinforce.sh
```

---

### Section 3.1: KL Divergence with Max-Probability Target (Batch Normalized)

**Paper Reference:** Section 3.1, Figures 2-5, Appendix A (Figure A1)

**Method:**
- Loss calculated for **all sequences** in batch (both correct and incorrect)
- Normalization: divided by total batch size
- Loss function:

```
L = E_t[p(max_t) * log(p(max_t)/p(chosen_t)) + p(not_max_t) * log(p(not_max_t)/p(not_chosen_t))]
```

**Implementation:** This is the **sequence-level** averaging mode in the SPO trainer.

**Key Observations from Paper:**
- Training loss increases over time (artifact of growing number of correct paths)
- Test accuracy improves consistently
- Average response length shows exploration → consolidation phases
- Training-inference discrepancy observed (lower training accuracy with T=1.0)

**Configuration:**
- Trainer: `RL2.trainer.spo`
- Advantage Estimator: `spo` (uses rewards directly)
- Loss averaging: `actor.avg_level=sequence`
- Prompts per rollout: 512
- Responses per prompt: 1

**Run Command:**
```bash
torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.spo \
    train_data.path=train@openai/gsm8k:main \
    train_data.prompts_per_rollout=512 \
    train_data.responses_per_prompt=1 \
    test_data.path=test@openai/gsm8k:main \
    test_data.prompts_per_rollout=512 \
    test_data.responses_per_prompt=1 \
    actor.model_name=Qwen/Qwen3-4B-Thinking-2507 \
    actor.use_liger_kernel=true \
    actor.max_length_per_device=8576 \
    actor.avg_level=sequence \
    actor.entropy.coef=0 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/gsm8k.py \
    rollout.gpu_memory_utilization=0.3 \
    trainer.project=GSM8K \
    trainer.experiment_name=section_3.1_batch_normalized \
    trainer.n_epochs=150 \
    trainer.save_freq=20 \
    trainer.test_freq=5
```

Or use the provided script:
```bash
bash examples/gsm8k_0.sh
```

---

### Section 3.2: KL Divergence with Max-Probability Target (Correct-Only Normalized)

**Paper Reference:** Section 3.2, Figures 6-9

**Method:**
- Loss calculated **only for correct sequences** (reward > 0)
- Normalization: divided by number of correct sequences
- Same KL divergence formulation as Section 3.1
- Gradient scaling applied to account for correct-only normalization

**Implementation:** Uses the main `update()` method in `actor.py` with correct sequence filtering.

**Key Observations from Paper:**
- Similar accuracy and entropy trends to Section 3.1
- Normalized loss still increases (divergence values increase with exploration)
- More token-efficient reasoning compared to baseline
- Training dynamics show similar exploration → consolidation phases

**Configuration:**
- Same as Section 3.1, but uses the default `update()` method which filters correct sequences

**Note:** The codebase currently uses this as the default implementation in `RL2.trainer.spo` when using the standard `actor.update()` method (see `RL2/workers/actor.py:373-452`).

---

### Section 3.3: Fixed Soft Target (0.5, 0.5)

**Paper Reference:** Section 3.3, Figures 10-13

**Method:**
- Target distribution fixed at P = (0.5, 0.5) instead of chasing max probability
- Stronger incentive to select correct paths in high-temperature settings
- Loss function:

```
L = -0.5 * (log p(chosen_t) + log(1 - p(chosen_t))) + log(0.5)
```

**Key Observations from Paper:**
- Faster initial improvement in test accuracy
- **Model collapse** observed after ~200 steps
- Training loss consistently decreases (unlike Sections 3.1-3.2)
- Most token-efficient at peak performance
- Suggests loss-metric misalignment or reward hacking

**Configuration:**
- Trainer: `RL2.trainer.spo_0p5`
- Fixed target: 0.5 for correct token probability
- All other settings same as Section 3.1

**Run Command:**
```bash
torchrun \
    --nproc_per_node=1 \
    -m RL2.trainer.spo_0p5 \
    train_data.path=train@openai/gsm8k:main \
    train_data.prompts_per_rollout=512 \
    train_data.responses_per_prompt=1 \
    test_data.path=test@openai/gsm8k:main \
    test_data.prompts_per_rollout=512 \
    test_data.responses_per_prompt=1 \
    actor.model_name=Qwen/Qwen3-4B-Thinking-2507 \
    actor.use_liger_kernel=true \
    actor.max_length_per_device=8576 \
    actor.avg_level=sequence \
    actor.entropy.coef=0 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/gsm8k.py \
    rollout.gpu_memory_utilization=0.3 \
    trainer.project=GSM8K \
    trainer.experiment_name=section_3.3_fixed_target \
    trainer.n_epochs=150 \
    trainer.save_freq=20 \
    trainer.test_freq=5
```

Or use the provided script:
```bash
bash examples/gsm8k_0p5_0.sh
```

---

## Monitoring and Analysis

### Weights & Biases Integration

All experiments automatically log to Weights & Biases (wandb). Key metrics tracked:

**Accuracy Metrics:**
- `test/score`: Test accuracy (greedy decoding, T=0)
- `train/score`: Training accuracy (sampled, T=1.0)

**Training Dynamics:**
- `actor/loss`: Training loss
- `actor/entropy`: Average token-level entropy
- `test/avg_response_length`: Average response length

**To monitor experiments:**
1. Set up wandb: `wandb login`
2. View results at: `https://wandb.ai/<your-username>/GSM8K`

### Local Checkpoints

Models are saved every 20 steps to:
```
/storage/junsoo/spo/<experiment_name>/  # For spo trainer
/storage/junsoo/ppo/<experiment_name>/  # For ppo trainer
```

You can modify the save directory in the config:
```bash
trainer.save_dir=/your/custom/path
```

---

## Understanding the Results

### Expected Trends

Based on the paper results, you should observe:

**1. Test Accuracy (All Methods)**
- Baseline (Dr.GRPO): Steady improvement
- Section 3.1 & 3.2: Consistent improvement, similar final performance → collapse slowly
- Section 3.3: Rapid improvement → peak → collapse fast

**2. Entropy**
- All proposed methods show **increasing entropy** (maintaining diversity)
- Baseline may show lower or decreasing entropy

**3. Response Length**
- Proposed methods: Initial increase (exploration) → decrease (consolidation)
- More token-efficient than baseline at similar accuracy levels

**4. Training Loss**
- Section 3.1 & 3.2: **Increasing** (more correct paths being optimized)
- Section 3.3: **Decreasing** (but model collapses)

---

## Multi-GPU Training (Optional)

To scale experiments to multiple GPUs:

```bash
# Single node, 4 GPUs
torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.spo \
    [... same arguments as above ...]

# Multi-node training
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=<master-node-ip> \
    --master_port=29500 \
    -m RL2.trainer.spo \
    [... same arguments as above ...]
```

Adjust `rollout.gpu_memory_utilization` based on GPU memory (0.3 for single GPU, can increase for multi-GPU setups).

---

## Proposed Future Work (Section 4.2-4.4)

The paper proposes several improvements to address stability issues:

### 1. Entropy Regularization (Section 4.2)
Add penalty term to dampen entropy growth:
```bash
actor.entropy.coef=-0.1  # or -0.5, -1.0 (Use negative coefficient to dampen entropy growth)
```

### 2. Rank-Aware Target with JSD (Section 4.3)
- Not yet implemented in current codebase
- Promotes chosen token to rank 1
- Uses Jensen-Shannon Divergence
- Computational efficiency: O(k) where k = rank of chosen token

### 3. Pre-training Application (Section 4.4)
- Apply greedy JSD loss during pre-training
- Encourage highest probability on correct tokens

---

## Citation

If you use this code or reproduce the experiments, please cite:

```bibtex
@thesis{Byun2025ExploringPaths,
    author = {Junsoo Byun},
    title = {Exploring Paths in Probabilistic Graphs for Model Training},
    school = {Seoul National University},
    year = {2025},
    type = {Bachelor's Thesis},
    department = {Mechanical Engineering}
}
```

For the RL2 library:
```bibtex
@misc{Tan2025RL2,
    author = {Chenmien Tan and Simon Yu and Lanbo Lin and Ze Zhang and Yuanwu Xu and Chenhao Jiang and Tianyuan Yang and Sicong Xie and Guannan Zhang},
    title = {RL2: Ray Less Reinforcement Learning},
    note = {GitHub repository},
    howpublished = {\url{https://github.com/ChenmienTan/RL2}},
    year = {2025}
}
```

---

## Acknowledgments

The implementation is built upon the RL2 library developed by Chenmien Tan and collaborators.

**Key Dependencies:**
- [RL2](https://github.com/ChenmienTan/RL2): Post-training library for LLMs
- [SGLang](https://github.com/sgl-project/sglang): Async inference engine
- [Qwen](https://github.com/QwenLM/Qwen): Base model (Qwen3-4B-Thinking-2507)
- [GSM8K](https://github.com/openai/grade-school-math): Evaluation dataset

---

## License

This project inherits the license from the RL2 library. Please refer to the [original RL2 repository](https://github.com/ChenmienTan/RL2) for licensing details.

---

## Contact

For questions about the paper or reproduction:
- **Author:** Junsoo Byun (jsbyun121@snu.ac.kr)
- **Institution:** Seoul National University

For questions about the RL2 library:
- See the [RL2 repository](https://github.com/ChenmienTan/RL2)
