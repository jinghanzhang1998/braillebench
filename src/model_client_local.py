"""
Local model client for running small models (Qwen3-1.5B, Qwen3-4B, etc.) on GPU.

Requires: transformers, torch, accelerate
Install: pip install transformers torch accelerate

Usage:
    from model_client_local import invoke_local_model, LOCAL_MODELS

    response = invoke_local_model("qwen3-1.5b", "What is 2+3?")
"""

import json
import os
from pathlib import Path

LOCAL_MODELS = {
    "qwen3-1.5b": {
        "model_id": "Qwen/Qwen3-1.5B",
        "type": "causal_lm",
    },
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "type": "causal_lm",
    },
}

_loaded_models = {}


def load_model(model_name: str):
    """Load model and tokenizer into memory (cached)."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    if model_name not in LOCAL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LOCAL_MODELS.keys())}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    config = LOCAL_MODELS[model_name]
    model_id = config["model_id"]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    _loaded_models[model_name] = (model, tokenizer)
    print(f"Loaded {model_id} on {model.device}")
    return model, tokenizer


def invoke_local_model(model_name: str, prompt: str, max_tokens: int = 1024,
                       temperature: float = 0.0) -> str:
    """Run inference on a local model.

    Args:
        model_name: Key from LOCAL_MODELS
        prompt: User prompt text
        max_tokens: Max new tokens to generate
        temperature: Sampling temperature (0 = greedy)

    Returns:
        Generated response text
    """
    import torch

    model, tokenizer = load_model(model_name)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


def invoke_local_batch(model_name: str, prompts: list[str], max_tokens: int = 1024,
                       temperature: float = 0.0, batch_size: int = 8) -> list[str]:
    """Batch inference for efficiency on GPU."""
    import torch

    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_texts = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=2048).to(model.device)

        with torch.no_grad():
            if temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = output[input_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            results.append(response)

    return results


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.5b"
    print(f"Testing {model_name}...")

    response = invoke_local_model(model_name, "What is 2 + 3? Answer in one word.")
    print(f"Response: {response}")
