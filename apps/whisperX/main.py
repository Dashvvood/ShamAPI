import elinor
DOTENV = elinor.fast_loadenv_then_append_path(keys=["PROJECT_ROOT"])

import os
import time
import numpy as np
from omegaconf import OmegaConf

from loguru import logger;logger.remove()

from fastapi import FastAPI, Depends
from typing import Dict, Any
import whisperx
from whisperx.diarize import DiarizationPipeline

from utils.api_process import get_client_info
from utils.audio_process import process_audio_info

O_D= elinor.O_D()
config = OmegaConf.load("./config.yaml")
logger.add(**config["log"])

model = whisperx.load_model(**config["model"])

align_models = {}

app = FastAPI(title="whisper", version=0.1)
logger.info("Model loaded.")


@app.get("/")
async def index(client_info: Dict = Depends(get_client_info)):
    logger.info(f"IP: {client_info['ip']}")
    return {
        "pid": os.getpid(),
        "worker_id": os.environ["APP_WORKER_ID"],
        "config": OmegaConf.to_container(config, resolve=True),
        "client_info": client_info,
        "model": {
            "asr": str(type(model)),
            "align": {lang: str(type(m[0])) for lang, m in align_models.items()}
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
    result = model.transcribe(data, batch_size=config["run"]["batch_size"], language=language)

    if result["language"] not in align_models:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=config["model"]["device"])
        align_models[result["language"]] = (model_a, metadata)
    else:
        model_a, metadata = align_models[result["language"]]

    # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=config["device"])

    result = whisperx.align(result["segments"], model_a, metadata, data, config["model"]["device"], return_char_alignments=False)

    toc = time.time()
    processed_time = round(toc - tic, 4)

    logger.info(f"IP: {client_info['ip']}; Time: {processed_time}s")
    return {
        "processed_time": processed_time,
        "model_output": result
    }
