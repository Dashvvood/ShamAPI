# ShamAPI
A sham plan, a true span.

See [docs/structure.md](docs/structure.md) for architecture.

## Quick start (Qwen3.5-2B)

Weights: `cache/Qwen3.5-2B`

```bash
uv sync --extra qwen
CONFIG=dev ./run.sh
# listens on :8001

curl -s http://127.0.0.1:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation":[{"role":"user","content":"你好"}],"max_new_tokens":64}'
```

Optional LiteLLM gateway (OpenAI clients → app):

```bash
litellm --config config/litellm.yaml --host 0.0.0.0 --port 8080
```

## Tests

```bash
pytest test/qwen3_5.py -v -m "not integration"
```
