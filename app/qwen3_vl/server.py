from .. import QWEN3_VL_CONFIG_PATH
from loguru import logger;logger.remove()


import os
import time
import numpy as np
from omegaconf import OmegaConf

from fastapi import FastAPI, Depends
from typing import Dict, Any

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from utils.api_process import get_client_info
from qwen_vl_utils import process_vision_info

config = OmegaConf.load(QWEN3_VL_CONFIG_PATH)
logger.add(**config["log"])

model_dir = config["model"]["model_dir"]
if not os.path.exists(model_dir):
    # Load model directly
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=config["model"]["repo_id"], local_dir=model_dir)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    config["model"]["model_dir"],
    device_map=config["model"]["device_map"],
    torch_dtype=getattr(torch, config["model"]["torch_type"])
)
processor = AutoProcessor.from_pretrained(config["model"]["model_dir"])


app = FastAPI(title="Qwen3-VL", version=0.1)
logger.info("Model and Processor loaded.")


@app.get("/")  
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"],
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info
    }


@app.post("/chat")
async def chat(
    message: Dict,
    client_info: Dict = Depends(get_client_info)
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
    try:
        tic = time.time()
        conversation = message["conversation"]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([conversation], return_video_kwargs=True,
                                                                image_patch_size= 16,
                                                                return_video_metadata=True)
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128) # do_sample = T or F?
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
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
        logger.error(f"Error processing request from IP {client_info['ip']}: {e}")
        res = {
            "api": "/chat",
            "error": str(e),
        }
    return res
    
