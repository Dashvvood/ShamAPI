from .. import QWEN3_CONFIG_PATH
from loguru import logger;logger.remove()


import os
import time
from omegaconf import OmegaConf

from fastapi import Depends, FastAPI, HTTPException
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.api_process import download_model, get_client_info

config = OmegaConf.load(QWEN3_CONFIG_PATH)
logger.add(**config["log"])

model_dir = config["model"]["model_dir"]
if not os.path.exists(model_dir):
    download_model(config["model"]["repo_id"], model_dir, provider="auto")

_tt = str(config["model"]["torch_type"]).lower()
_torch_dtype = "auto" if _tt == "auto" else getattr(torch, config["model"]["torch_type"])

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=config["model"]["device_map"],
    torch_dtype=_torch_dtype,
)

app = FastAPI(title="Qwen3", version=0.1)
logger.info("Model and tokenizer loaded.")


@app.get("/")
async def index(client_info: Dict[str, Any] = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"],
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info,
    }


@app.post("/chat")
def chat(
    message: Dict[str, Any],
    client_info: Dict[str, Any] = Depends(get_client_info),
):
    """
    Text-only chat. `conversation` is passed to `apply_chat_template` as-is;
    each message must use string `content` (Hugging Face chat format).

        {
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "..."},
            ],
            "max_new_tokens": 512
        }
    """
    conversation = message.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise HTTPException(
            status_code=422,
            detail={
                "api": "/chat",
                "error": "body must include non-empty list field 'conversation'",
            },
        )

    max_new_tokens = message.get("max_new_tokens", 2048)
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise HTTPException(
            status_code=422,
            detail={
                "api": "/chat",
                "error": "max_new_tokens must be a positive integer when provided",
            },
        )

    try:
        tic = time.time()
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        in_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][in_len:].tolist()
        reply = tokenizer.decode(output_ids, skip_special_tokens=True)
        toc = time.time()
        processed_time = round(toc - tic, 4)
        logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")
        return {
            "api": "/chat",
            "model": config["model"]["repo_id"],
            "model_output": {"text": reply},
            "processed_time": processed_time,
        }
    except Exception as e:
        logger.exception("Error processing /chat from IP {}: {}", client_info["ip"], e)
        raise HTTPException(
            status_code=500,
            detail={"api": "/chat", "error": str(e)},
        ) from e
