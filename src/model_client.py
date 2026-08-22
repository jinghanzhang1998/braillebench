"""Provider-neutral model adapter shipped in the public BrailleBench release.

Add model names to ``MODELS`` and implement ``invoke_with_retry`` for your API,
local inference server, or Transformers runtime. The benchmark intentionally does
not prescribe a provider or read credentials itself.
"""

MODELS = {
    # "my-model": {},
}


def invoke_with_retry(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Return one text completion for ``prompt``.

    Implement retry/backoff inside this function. Raise transient provider errors:
    BrailleBench recognizes common connection, timeout, throttling, and credential
    failures and pauses the run instead of recording them as model failures.

    Minimal OpenAI-compatible shape (after installing your provider SDK)::

        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    """
    raise NotImplementedError(
        "Connect your model backend in src/model_client.py before running BrailleBench."
    )
