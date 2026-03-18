"""
Author: Pranay Hedau
model_utils.py
--------------
Purpose: Model loading, QLoRA configuration, and inference utilities.
Used by notebooks 02, 03, and 04.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


def get_bnb_config():
    """
    Build the 4-bit quantization config for QLoRA.

    NF4 (NormalFloat4) is chosen over INT4 because LLM weights
    follow a roughly normal distribution. NF4's quantization grid
    is designed for normal distributions, giving lower quantization
    error than INT4 for the same bit width.

    double_quant=True quantizes the quantization constants themselves,
    saving an extra 0.4 bits per parameter on average.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(r=64, lora_alpha=16, lora_dropout=0.05):
    """
    Build the LoRA adapter config.

    We attach adapters to q/k/v/o projection layers because these
    control what the model attends to and how it combines information.
    They carry the most signal for instruction-following behavior.

    Effective learning scale = lora_alpha / r = 16/64 = 0.25
    Lower scale = more conservative updates, less risk of catastrophic
    forgetting of pre-trained knowledge.

    Args:
        r           : LoRA rank (adapter expressiveness)
        lora_alpha  : scaling factor
        lora_dropout: dropout for regularization
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_base_model_4bit(model_id, hf_token):
    """
    Load a causal LM in 4-bit quantization for training on GPU.

    Sets use_cache=False because gradient checkpointing and KV cache
    are incompatible. The KV cache speeds up inference but during
    training we need full forward passes for gradient computation.
    """
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def load_tokenizer(model_id, hf_token, padding_side="right"):
    """
    Load tokenizer and set padding configuration.

    padding_side="right" for SFT (standard causal LM training)
    padding_side="left"  for DPO (ensures content aligns at end of sequence)

    Llama has no pad token by default so we assign eos_token as pad.
    This is safe because we mask pad positions during loss computation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_model_for_training(model_id, hf_token, lora_rank=64,
                             lora_alpha=16, lora_dropout=0.05):
    """
    Load base model in 4-bit and attach trainable LoRA adapters.
    Used for SFT training in notebook 02.

    Returns (model_with_lora, trainable_param_count, total_param_count)
    """
    base  = load_base_model_4bit(model_id, hf_token)
    lora  = get_lora_config(lora_rank, lora_alpha, lora_dropout)
    model = get_peft_model(base, lora)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    return model, trainable, total


def load_sft_adapter(base_model_id, adapter_dir, hf_token,
                     trainable=True):
    """
    Load base model and attach a saved SFT adapter.

    trainable=True  -> policy model for DPO training (weights updated)
    trainable=False -> reference model for DPO training (weights frozen)

    Both the policy and reference start from the same SFT checkpoint.
    DPO training only updates the policy's weights.
    """
    base  = load_base_model_4bit(base_model_id, hf_token)
    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=trainable
    )
    return model


def load_model_for_inference(base_model_id, adapter_dir, hf_token):
    """
    Load model for inference on Mac (no quantization needed).

    Uses MPS (Apple Silicon) if available, otherwise CPU.
    Runs in float16 to reduce memory footprint.
    """
    if torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
    else:
        device = "cpu"
        dtype  = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device,
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt,
                      max_new_tokens=256, temperature=0.7):
    """
    Generate a single response from model given a prompt string.

    Applies the Llama 3.2 Instruct chat template manually.
    Returns only the newly generated tokens (not the prompt).
    """
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + prompt
        + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(
        formatted, return_tensors="pt",
        truncation=True, max_length=512
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
