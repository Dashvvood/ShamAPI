vllm serve model/Qwen2.5-Omni-3B \
--trust-remote-code \
--host 0.0.0.0 \
--port 17702 \
--served-model-name VQA \
--gpu-memory-utilization 0.95