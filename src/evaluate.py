"""
Author: Pranay Hedau
evaluate.py
-----------
Purpose: LLM-as-judge evaluation pipeline.
GPT-4o as blind pairwise judge for win-rate and Best-of-N scoring.
Used by notebook 04.
"""

import time
import random
import numpy as np
from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI assistant responses.
Compare two responses to the same prompt and decide which is better.
Evaluate on: helpfulness, clarity, accuracy, and completeness.
Do not favor longer responses. Quality over quantity.
Respond with ONLY: A, B, or TIE"""

SCORER_SYSTEM_PROMPT = """Score this AI response 1-10 based on helpfulness,
clarity, and accuracy. Respond with ONLY a number from 1 to 10."""


def judge_responses(prompt, response_a, response_b, client,
                    model="gpt-4o", sleep=0.3):
    """
    Ask GPT-4o to judge which of two responses is better.

    Positions are NOT randomized here — randomization is handled
    by the caller (win_rate_eval) to keep this function pure.

    Args:
        prompt     : the original user prompt
        response_a : first response to compare
        response_b : second response to compare
        client     : OpenAI client
        model      : judge model (default gpt-4o)
        sleep      : seconds to wait after call (rate limit buffer)

    Returns:
        "A", "B", or "TIE"
    """
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Prompt: {prompt[:500]}\n\n"
                    f"Response A:\n{response_a[:600]}\n\n"
                    f"Response B:\n{response_b[:600]}\n\n"
                    "Which is better? Reply: A, B, or TIE"
                )},
            ],
            temperature=0,
            max_tokens=5,
        )
        verdict = r.choices[0].message.content.strip().upper()
        time.sleep(sleep)
        return verdict if verdict in ["A", "B", "TIE"] else "TIE"
    except Exception as e:
        print(f"Judge error: {e}")
        return "TIE"


def score_response(prompt, response, client, model="gpt-4o"):
    """
    Score a single response 1-10 using GPT-4o.
    Used by best_of_n to select the best candidate.

    Returns float score (1.0 to 10.0), defaults to 5.0 on error.
    """
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SCORER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Prompt: {prompt[:300]}\n\nResponse: {response[:500]}"
                )},
            ],
            temperature=0,
            max_tokens=5,
        )
        return float(r.choices[0].message.content.strip())
    except Exception:
        return 5.0


def win_rate_eval(prompts, model_a, model_b, tokenizer,
                  generate_fn, client, label_a="A", label_b="B",
                  sleep=0.3, seed=42):
    """
    Run blind pairwise win-rate evaluation between two models.

    Randomly flips which model appears as Response A vs B to
    eliminate position bias (some judges favor the first response
    regardless of quality).

    Args:
        prompts     : list of evaluation prompts
        model_a     : first model (e.g. SFT)
        model_b     : second model (e.g. DPO)
        tokenizer   : shared tokenizer
        generate_fn : function(model, tokenizer, prompt) -> str
        client      : OpenAI client
        label_a     : name for model_a in results
        label_b     : name for model_b in results
        sleep       : seconds between API calls
        seed        : random seed for flip randomization

    Returns:
        list of result dicts with keys:
        prompt, response_a, response_b, verdict, winner
    """
    random.seed(seed)
    results = []

    for prompt in prompts:
        resp_a = generate_fn(model_a, tokenizer, prompt)
        resp_b = generate_fn(model_b, tokenizer, prompt)

        # Randomize position to remove position bias
        flip = random.random() > 0.5
        if flip:
            ra, rb = resp_a, resp_b
            ma, mb = label_a, label_b
        else:
            ra, rb = resp_b, resp_a
            ma, mb = label_b, label_a

        verdict = judge_responses(prompt, ra, rb, client, sleep=sleep)
        winner  = ma if verdict == "A" else (mb if verdict == "B" else "tie")

        results.append({
            "prompt"    : prompt[:150],
            "winner"    : winner,
            "response_a": resp_a[:200],
            "response_b": resp_b[:200],
            "flipped"   : flip,
        })

    return results


def best_of_n(model, tokenizer, prompt, n, generate_fn, client,
              temperature=0.8):
    """
    Generate n candidate responses and return the highest-scoring one.

    This is the simplest form of test-time scaling:
    more inference compute -> better output quality.
    The same principle underlies OpenAI o1 and DeepSeek R1.

    Args:
        model       : the model to generate from
        tokenizer   : tokenizer
        prompt      : input prompt
        n           : number of candidates to generate
        generate_fn : function(model, tokenizer, prompt, temperature) -> str
        client      : OpenAI client for scoring (None = return first candidate)
        temperature : sampling temperature (higher = more diverse candidates)

    Returns:
        best_response (str)
    """
    candidates = [
        generate_fn(model, tokenizer, prompt, temperature=temperature)
        for _ in range(n)
    ]

    if client is None or n == 1:
        return candidates[0]

    scores  = [score_response(prompt, c, client) for c in candidates]
    best    = int(np.argmax(scores))
    return candidates[best]


def compute_win_rate_stats(results, label="dpo"):
    """
    Compute win rate with 95% confidence interval.

    Args:
        results : list of result dicts from win_rate_eval
        label   : model label to compute win rate for

    Returns:
        dict with win_rate, ci_low, ci_high, n_wins, n_total
    """
    n_total = len(results)
    n_wins  = sum(1 for r in results if r["winner"] == label)

    rate   = n_wins / n_total
    std_err = (rate * (1 - rate) / n_total) ** 0.5
    ci_low  = max(0, rate - 1.96 * std_err)
    ci_high = min(1, rate + 1.96 * std_err)

    return {
        "win_rate": rate * 100,
        "ci_low"  : ci_low * 100,
        "ci_high" : ci_high * 100,
        "n_wins"  : n_wins,
        "n_total" : n_total,
    }
