# Model Configuration

## Evaluation Models

| Model | Provider | Size | Invoke ID | Speed | Notes |
|-------|----------|------|-----------|-------|-------|
| Claude Opus 4.8 | Anthropic | - | `us.anthropic.claude-opus-4-8` | ~5s/call | Flagship, strongest |
| Claude Haiku 4.5 | Anthropic | - | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | ~1.5s/call | Fast/cheap |
| Llama 3.3 70B | Meta | 70B dense | `us.meta.llama3-3-70b-instruct-v1:0` | ~2s/call | Best open dense |
| Llama 3.1 8B | Meta | 8B dense | `us.meta.llama3-1-8b-instruct-v1:0` | ~1s/call | Small open |
| Qwen3 32B | Alibaba | 32B dense | `qwen.qwen3-32b-v1:0` | ~12s/call | Thinking model |
| Qwen3 1.5B | Alibaba | 1.5B dense | local: `Qwen/Qwen3-1.5B` | local GPU | Tiny model |

## Bedrock Configuration

- **AWS Account**: 827410396081
- **Profile**: `h100`
- **Region**: `us-east-1`

## API Call Formats

### Claude (Anthropic)
```python
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "..."}]
}
# Note: temperature=0 omitted for Opus 4.8 (deprecated param)
```

### Llama (Meta)
```python
body = {
    "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "max_gen_len": 1024,
    "temperature": 0.01
}
```

### Qwen (Alibaba) - Bedrock
```python
body = {
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 1024,
    "temperature": 0.01
}
```

### Qwen3 1.5B - Local GPU
```python
from model_client_local import invoke_local_model
response = invoke_local_model("qwen3-1.5b", prompt, max_tokens=1024)
```

## Dropped Models (from earlier experiments)

| Model | Reason |
|-------|--------|
| Llama 4 Maverick 17B | Broken output format (GSM8K math_verify=13%, most QA=0%) |
| Qwen3 Next 80B | Too slow (17s/call, 45% of total time), AIME24=0%, redundant with Qwen3-32B |

## Time Estimates

| Task | Calls | Est. Time |
|------|-------|-----------|
| Baseline EN→EN (5 Bedrock models) | 27,850 | ~27h |
| One braille config (e.g. G1-EN ascii, 5 models) | 27,850 | ~27h |
| Full braille eval (6 configs × 3 formats) | 501,300 | ~486h |
| Recommended subset (P0+P1: 4 configs × ascii) | 111,400 | ~108h |
