"""
Unified Bedrock model client for evaluation.

Handles different API formats for Claude, Llama, and Qwen models.
"""

import json
import time
import boto3

BEDROCK_PROFILE = "h100"
BEDROCK_REGION = "us-east-1"

MODELS = {
    "claude-opus-4.8": {
        "model_id": "us.anthropic.claude-opus-4-8",
        "provider": "anthropic",
    },
    "claude-sonnet-4.6": {
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "provider": "anthropic",
    },
    "claude-haiku-4.5": {
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "provider": "anthropic",
    },
    "llama4-maverick": {
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "provider": "meta",
    },
    "llama3.3-70b": {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
        "provider": "meta",
    },
    "llama3.1-8b": {
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
        "provider": "meta",
    },
    "qwen3-32b": {
        "model_id": "qwen.qwen3-32b-v1:0",
        "provider": "qwen",
    },
    "qwen3-next-80b": {
        "model_id": "qwen.qwen3-next-80b-a3b",
        "provider": "qwen",
    },
}

_client = None


def get_client():
    global _client
    if _client is None:
        from botocore.config import Config
        session = boto3.Session(profile_name=BEDROCK_PROFILE, region_name=BEDROCK_REGION)
        _client = session.client(
            "bedrock-runtime",
            config=Config(read_timeout=120, connect_timeout=10, retries={"max_attempts": 3}),
        )
    return _client


def invoke_model(model_name: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Send a prompt to a model and return the response text.

    Args:
        model_name: Key from MODELS dict (e.g. "claude-opus-4.8")
        prompt: The user prompt text
        max_tokens: Max response tokens
        temperature: Sampling temperature

    Returns:
        Model response as string
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    config = MODELS[model_name]
    client = get_client()
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "anthropic":
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature > 0:
            body_dict["temperature"] = temperature
        body = json.dumps(body_dict)
    elif provider == "meta":
        body = json.dumps({
            "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            "max_gen_len": max_tokens,
            "temperature": max(temperature, 0.01),  # Meta doesn't accept 0
        })
    elif provider == "qwen":
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
        })
    else:
        raise ValueError(f"Unknown provider: {provider}")

    resp = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())

    if provider == "anthropic":
        content = result.get("content", [])
        if not content:
            return ""
        return content[0].get("text", "")
    elif provider == "meta":
        return result.get("generation", "")
    elif provider == "qwen":
        choices = result.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")


def invoke_with_retry(model_name: str, prompt: str, max_tokens: int = 1024,
                      temperature: float = 0.0, max_retries: int = 3) -> str:
    """Invoke model with exponential backoff retry on throttling."""
    for attempt in range(max_retries):
        try:
            return invoke_model(model_name, prompt, max_tokens, temperature)
        except Exception as e:
            error_str = str(e)
            if "ThrottlingException" in error_str or "Too many requests" in error_str:
                wait = 2 ** attempt * 5
                print(f"  Throttled, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    return invoke_model(model_name, prompt, max_tokens, temperature)
