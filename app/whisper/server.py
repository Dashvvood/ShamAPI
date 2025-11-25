from .. import PROJECT_ROOT
from loguru import logger;logger.remove()

import os
import time
import whisper
from typing import Dict, Any
from omegaconf import OmegaConf

from fastapi import FastAPI, Depends
from utils.api_process import get_client_info
from utils.audio_process import process_audio_info


config = OmegaConf.load(os.path.join(PROJECT_ROOT, "config/whisper.yaml"))
logger.add(**config["log"])

model = whisper.load_model(**config["model"])
app = FastAPI(title="whisper", version=0.1)
logger.info("Model loaded.")


@app.get("/")
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"] if "APP_WORKER_ID" in os.environ else None,
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info
    }


@app.get("/help")
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "message" : {
            "audio": "https://example.com/audio.mp3",
            "language": "en"
        }
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
