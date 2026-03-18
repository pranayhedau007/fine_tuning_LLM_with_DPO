"""
Author: Pranay Hedau
data_utils.py
-------------
Purpose: Dataset loading, formatting, and preprocessing utilities.
Used by notebooks 02 (SFT) and 03 (DPO).
"""

import json
import random
from datasets import load_dataset, Dataset


def extract_assistant_text(field):
    """
    Extract plain assistant response text from chosen/rejected field.

    UltraFeedback stores responses as a list of {role, content} dicts:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    We extract only the assistant turn since that is what we train on.
    """
    if isinstance(field, list):
        for turn in field:
            if turn.get("role") == "assistant":
                return turn.get("content", "")
        # fallback: return last turn content
        return field[-1].get("content", "") if field else ""
    return str(field)


def format_sft(example):
    """
    Format a preference pair for SFT training.

    SFT trains only on chosen responses using cross-entropy loss.
    The model learns to predict the next token in a good response
    given the prompt. No preference signal yet, just instruction-following.

    Returns a dict with a single 'text' field containing the full
    formatted conversation string in Llama 3.2 Instruct chat format.
    """
    prompt   = example["prompt"]
    response = extract_assistant_text(example["chosen"])

    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}"
        "<|eot_id|>"
    )
    return {"text": text}


def format_dpo(example):
    """
    Format a preference pair for DPO training.

    DPOTrainer expects plain strings for prompt, chosen, and rejected.
    It applies the chat template internally via the tokenizer.

    Why plain strings and not pre-formatted?
    Pre-formatting causes double-formatting bugs in DPOTrainer.
    Always pass raw text and let DPOTrainer handle the template.
    """
    return {
        "prompt"  : example["prompt"],
        "chosen"  : extract_assistant_text(example["chosen"]),
        "rejected": extract_assistant_text(example["rejected"]),
    }


def load_ultrafeedback_subset(data_path, split_ratio=0.9, seed=42):
    """
    Load the 10K UltraFeedback subset and return train/val splits.

    Args:
        data_path   : path to ultrafeedback_10k.jsonl
        split_ratio : fraction used for training (default 0.9)
        seed        : random seed for reproducibility

    Returns:
        train_dataset, val_dataset
    """
    dataset = load_dataset("json", data_files=data_path, split="train")
    split   = dataset.train_test_split(test_size=1 - split_ratio, seed=seed)
    return split["train"], split["test"]


def prepare_sft_dataset(data_path, split_ratio=0.9, seed=42):
    """
    Load and format dataset for SFT training.

    Returns train/val datasets with a single 'text' field
    formatted in Llama 3.2 chat template style.
    """
    train_raw, val_raw = load_ultrafeedback_subset(data_path, split_ratio, seed)

    train_sft = train_raw.map(format_sft, remove_columns=train_raw.column_names)
    val_sft   = val_raw.map(format_sft,   remove_columns=val_raw.column_names)

    return train_sft, val_sft


def prepare_dpo_dataset(data_path, split_ratio=0.9, seed=42):
    """
    Load and format dataset for DPO training.

    Returns train/val datasets with prompt, chosen, rejected fields
    as plain strings ready for DPOTrainer.
    """
    train_raw, val_raw = load_ultrafeedback_subset(data_path, split_ratio, seed)

    train_dpo = train_raw.map(format_dpo, remove_columns=train_raw.column_names)
    val_dpo   = val_raw.map(format_dpo,   remove_columns=val_raw.column_names)

    return train_dpo, val_dpo


def load_held_out_prompts(full_dataset_name, train_indices_path, n=100, seed=42):
    """
    Load evaluation prompts that were NOT in the training subset.

    Used in notebook 04 to ensure win-rate evaluation is on
    truly unseen prompts.

    Args:
        full_dataset_name  : HuggingFace dataset name
        train_indices_path : path to subset_indices.json
        n                  : number of eval prompts to sample
        seed               : random seed

    Returns:
        list of prompt strings
    """
    dataset = load_dataset(full_dataset_name, split="train")

    with open(train_indices_path) as f:
        train_indices = set(json.load(f))

    held_out = list(set(range(len(dataset))) - train_indices)
    random.seed(seed)
    random.shuffle(held_out)

    return [dataset[i]["prompt"] for i in held_out[:n]]
