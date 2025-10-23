import elinor
DOTENV = elinor.fast_loadenv_then_append_path(keys=["PROJECT_ROOT"])

import os
import time
import numpy as np
from omegaconf import OmegaConf

from loguru import logger;logger.remove()

from fastapi import FastAPI, Depends
from typing import Dict, Any
import whisper
from utils.api_process import get_client_info
from utils.audio_process import process_audio_info

O_D= elinor.O_D()
config = OmegaConf.load("./config.yaml")
logger.add(**config["log"])

model = whisper.load_model(**config["model"])
app = FastAPI(title="whisper", version=0.1)
logger.info("Model loaded.")


@app.get("/")
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"],
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info
    }



@app.post("/transcribe")
async def transcribe(
    message: Dict,
    client_info: Dict = Depends(get_client_info)
):
    """
    message = {
        "audio": "https://example.com/audio.mp3",
        "language": "en"
    }
    """
    tic = time.time()

    data = process_audio_info(message)
    language = message.get("language", None)
    result = model.transcribe(data, language=language)

    toc = time.time()
    processed_time = round(toc - tic, 4)

    logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")
    return {
        "processed_time": processed_time,
        "model_output": result
    }

