vllm serve model/Qwen2.5-VL-3B-Instruct \
--trust-remote-code \
--host 0.0.0.0 \
--port 17702 \
--served-model-name VQA \
--gpu-memory-utilization 0.95