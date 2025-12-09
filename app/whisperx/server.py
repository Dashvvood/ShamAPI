from .. import WHISPERX_CONFIG_PATH
from loguru import logger;logger.remove()

import os
import time
import whisperx
from typing import Dict, Any
from omegaconf import OmegaConf

from fastapi import FastAPI, Depends
from utils.api_process import get_client_info
from utils.audio_process import process_audio_info

config = OmegaConf.load(WHISPERX_CONFIG_PATH)
logger.add(**config["log"])
model = whisperx.load_model(**config["model"])
app = FastAPI(title="whisper", version=0.1)
logger.info(f"Model loaded: {type(model)}")

align_models = {}

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

if __name__ == "__main__":
    config = OmegaConf.load(os.path.join(PROJECT_ROOT, "config/whisper.yaml"))
    import uvicorn
    uvicorn.run(app, host=config["app"]["host"], port=config["app"]["port"])