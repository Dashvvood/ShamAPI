#!/bin/bash

VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3-4B --enable-reasoning --reasoning-parser deepseek_r1