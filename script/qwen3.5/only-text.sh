#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 默认本地权重目录；覆盖示例: 
# MODEL_DIR=/data/models/Qwen3.5-2B ./script/qwen3.5/only-text.sh
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/cache/Qwen3.5-2B}"

# VLLM_USE_MODELSCOPE=true vllm serve /mnt/svp/Downloads/Qwen3.5-2B \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --max-model-len 32768 \
#   --enable-prefix-caching \
#   --trust-remote-code \
#   --dtype float16 


# VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-2B --port 8000 --tensor-parallel-size 1 --max-model-len 8196

VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3.5-2B --port 8000 --tensor-parallel-size 1 --max-model-len 8196 --language-model-only
