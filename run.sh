#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CONFIG=dev \
gunicorn -c config/gunicorn.conf.py -w 1 app:qwen3_vl