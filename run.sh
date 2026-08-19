#!/usr/bin/env bash
# Start Qwen3.5-2B FastAPI app (transformers).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export CONFIG="${CONFIG:-dev}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"

exec uvicorn app.qwen3_5:app --host "$HOST" --port "$PORT"
