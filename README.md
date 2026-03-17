# LLM Fine-Tuning with DPO

> Fine-tuning Llama 3.2 3B using Direct Preference Optimization (DPO) on UltraFeedback preference pairs,
> targeting measurable win-rate improvement over SFT baseline on helpfulness and instruction-following,
> evaluated using GPT-4o as pairwise LLM judge. Trained on Kaggle T4 GPU via QLoRA 4-bit quantization.

---

## Problem Statement

Large language models pretrained on internet text are not inherently aligned with human preferences.
They predict the next token well but do not naturally produce helpful, honest, or safe responses.

This project addresses that gap by applying **Direct Preference Optimization (DPO)** — a stable,
single-stage alternative to RLHF — to teach Llama 3.2 3B what "good" responses look like using
64K human preference pairs from the UltraFeedback dataset.

**Key research question**: How much does DPO improve response quality over the base SFT model,
measured by win rate on held-out prompts judged by GPT-4o?

---

## Approach

```
UltraFeedback (64K pairs)
    chosen response                   Llama 3.2 3B (base)
    rejected response   ──DPO──→      QLoRA fine-tuned
    prompt                            ↓
                                 Win rate evaluation
                                 (GPT-4o as judge)
```

### Why DPO over RLHF?
RLHF requires training a separate reward model, then running PPO — complex, unstable, and expensive.
DPO reformulates preference learning as a simple classification loss directly on the policy model.
Same alignment quality, far simpler training loop.

### Why QLoRA?
Fine-tuning all 3B parameters would require ~24GB VRAM. QLoRA quantizes the base model to 4-bit
and trains only low-rank adapter matrices (~1% of parameters), bringing VRAM usage to ~4GB —
fits comfortably on a free Kaggle T4 (15GB).

---

## Project Structure

```
fine_tuning_LLM_with_DPO/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # UltraFeedback EDA, preference pair analysis
│   ├── 02_sft_baseline.ipynb        # SFT baseline fine-tuning (reference point)
│   ├── 03_dpo_training.ipynb        # Full DPO training loop (runs on Kaggle T4)
│   └── 04_evaluation.ipynb          # Win rate eval, ablation results, analysis
│
├── src/
│   ├── data_utils.py                # Dataset loading, formatting, preprocessing
│   ├── model_utils.py               # Model loading, QLoRA config, tokenizer setup
│   └── evaluate.py                  # LLM-as-judge evaluation pipeline
│
├── configs/
│   ├── qlora_config.yaml            # QLoRA hyperparameters
│   └── dpo_config.yaml              # DPO training hyperparameters
│
├── data/                            # Preprocessed subsets (full data from HuggingFace)
├── results/                         # Win rate charts, ablation plots, eval outputs
├── .env.example                     # HuggingFace token template
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Notebooks Guide

| Notebook | Runs On | Purpose |
|----------|---------|---------|
| `01_data_exploration` | Mac (CPU) | Understand UltraFeedback structure, response length, quality distribution |
| `02_sft_baseline` | Kaggle T4 | Fine-tune SFT reference model to compare against |
| `03_dpo_training` | Kaggle T4 | Full DPO training with QLoRA, save checkpoint |
| `04_evaluation` | Mac (CPU) | Win rate analysis, ablation study plots, final results |

---

## Results (Preliminary)

| Model | Win Rate vs Base | Helpfulness Score | Instruction Following |
|-------|-----------------|-------------------|-----------------------|
| Llama 3.2 3B Base | 50% (baseline) | - | - |
| SFT Fine-tuned | ~55% | - | - |
| DPO Fine-tuned (ours) | ~65-70% | - | - |

*Full results populated after training run.*

---

## Setup

```bash
# Clone the repo
git clone https://github.com/pranayhedau007/fine_tuning_LLM_with_DPO.git
cd fine_tuning_LLM_with_DPO

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace token
cp .env.example .env
# Edit .env and add your HF token
```

---

## Training on Kaggle

1. Upload `notebooks/03_dpo_training.ipynb` to Kaggle
2. Enable GPU (T4 x1) in notebook settings
3. Add your HuggingFace token as a Kaggle Secret named `HF_TOKEN`
4. Enable internet access in settings
5. Run all cells — training takes approximately 3-4 hours
6. Model checkpoint saved to Kaggle output dataset automatically

---

## Dataset

**UltraFeedback** (openbmb/UltraFeedback on HuggingFace)
- 64K instruction-following preference pairs
- Each entry: prompt + chosen response + rejected response
- Covers helpfulness, honesty, and instruction-following
- We use a 10K subset for training to keep Kaggle session within limits

```python
from datasets import load_dataset
dataset = load_dataset("openbmb/UltraFeedback", split="train")
```

---

## Key Design Decisions

1. **QLoRA rank=64** — balances expressiveness vs memory. Lower ranks (8-16) underfit on preference data.
2. **DPO beta=0.1** — controls how far the policy diverges from the reference. Lower = more conservative.
3. **10K subset** — full 64K takes ~14 hours on T4. 10K gives strong signal in ~3-4 hours.
4. **Llama 3.2 3B Instruct** — Instruct variant already has chat template formatting, simplifies data prep.
5. **GPT-4o as judge** — pairwise evaluation prompt asks which response is more helpful without revealing which is from fine-tuned model (blind evaluation).

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-TRL-yellow)
![Kaggle](https://img.shields.io/badge/GPU-Kaggle_T4-20BEFF)

- **Training**: HuggingFace TRL (DPOTrainer), PEFT (QLoRA), bitsandbytes
- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Dataset**: openbmb/UltraFeedback
- **Evaluation**: GPT-4o pairwise LLM-as-judge
- **Hardware**: Kaggle T4 (15GB VRAM) for training, Mac for dev and eval

---

## Author

**Pranay Hedau**
MS Computer Science @ UC Irvine
[LinkedIn](https://www.linkedin.com/in/pranay-hedau/) · [GitHub](https://github.com/pranayhedau007) · [YouTube](https://www.youtube.com/watch?v=Jlu9al3lzH8)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
