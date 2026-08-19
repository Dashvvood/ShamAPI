from .. import QWEN3_VL_CONFIG_PATH
from loguru import logger;logger.remove()


import os
import time
from omegaconf import OmegaConf

from fastapi import Depends, FastAPI, HTTPException
from typing import Any, Dict

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from utils.api_process import download_model, get_client_info

config = OmegaConf.load(QWEN3_VL_CONFIG_PATH)
logger.add(**config["log"])

model_dir = config["model"]["model_dir"]
if not os.path.exists(model_dir):
    download_model(config["model"]["repo_id"], model_dir, provider="auto")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    config["model"]["model_dir"],
    device_map=config["model"]["device_map"],
    torch_dtype=getattr(torch, config["model"]["torch_type"])
)
processor = AutoProcessor.from_pretrained(config["model"]["model_dir"])


app = FastAPI(title="Qwen3-VL", version=0.1)
logger.info("Model and Processor loaded.")


@app.get("/")
async def index(client_info: Dict[str, Any] = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"],
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info
    }


@app.post("/chat")
def chat(
    message: Dict[str, Any],
    client_info: Dict[str, Any] = Depends(get_client_info),
):
    """
    conversation = message["conversation"]
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": "video01.mp4",
                    "video_start": 0.0,
                    "video_end": 5.0,
                },
                {"type": "text", "text": "你看到了什么?"},
            ],
        }
    ]
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

    try:
        tic = time.time()
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        toc = time.time()
        processed_time = round(toc - tic, 4)
        logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")
        res = {
            "api": "/chat",
            "model": config["model"]["repo_id"],
            "model_output": {
                "text": output_text[0]
            },
            "processed_time": processed_time,
        }

    except Exception as e:
        logger.exception("Error processing /chat from IP {}: {}", client_info["ip"], e)
        raise HTTPException(
            status_code=500,
            detail={"api": "/chat", "error": str(e)},
        ) from e
