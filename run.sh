#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CONFIG=development \
gunicorn -c config/gunicorn.conf.py \
-w 1 app:qwen3_vl