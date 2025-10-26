import elinor
DOTENV = elinor.fast_loadenv_then_append_path(keys=["PROJECT_ROOT"])
import os
import time
import numpy as np
from omegaconf import OmegaConf
from loguru import logger
logger.remove()

from fastapi import FastAPI, Depends
from typing import Dict, Any
from rtmlib import Wholebody

import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from utils.api_process import get_client_info
from qwen_omni_utils import process_mm_info

o_d = elinor.o_d()
config = OmegaConf.load("./config.yaml")
logger.add(**config["log"])

app = FastAPI(title="Qwen2.5-Omni", version=0.1)

model_dir = "./model/Qwen2.5-Omni-3B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_dir=config["model"]["model_dir"],
    device_map=config["model"]["device_map"],
    torch_dtype=getattr(torch, config["model"]["torch_type"])
)
processor = Qwen2_5OmniProcessor.from_pretrained(config["model"]["model_dir"])


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
    """
    tic = time.time()
    conversation = message["conversation"]
    USE_AUDIO_IN_VIDEO = message.get("use_audio_in_video", False)

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    # Inference: Generation of the output text and audio
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    toc = time.time()
    processed_time = round(toc - tic, 4)
    logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")
    
    return {
        "api": "/chat",
        "model_output": {
            "text": text
        },
        "processed_time": processed_time,
    }
