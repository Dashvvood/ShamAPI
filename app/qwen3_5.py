from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common import (
    AppState,
    ChatRequest,
    OpenAIChatRequest,
    api_error,
    load_config,
    make_lifespan,
    register_health_route,
    setup_logger,
)
from model.qwen3_5 import chat as model_chat
from model.qwen3_5 import load_qwen35
from utils.api_process import get_client_info

_CONFIG_NAME = os.environ.get("CONFIG", "dev").lower()
_CONFIG_PATH = PROJECT_ROOT / "config" / _CONFIG_NAME / "qwen3_5.yaml"
_CONFIG = load_config(str(_CONFIG_PATH))


def _load() -> AppState:
    setup_logger(_CONFIG)
    logger.info("Loading Qwen3.5 from {}", _CONFIG["model"]["model_dir"])
    bundle = load_qwen35(
        model_dir=_CONFIG["model"]["model_dir"],
        repo_id=_CONFIG["model"]["repo_id"],
        device_map=_CONFIG["model"]["device_map"],
        torch_type=_CONFIG["model"]["torch_type"],
    )
    logger.info("Qwen3.5 ready.")
    return AppState(config=_CONFIG, bundle=bundle)


app = FastAPI(
    title="Qwen3.5",
    version="0.1",
    lifespan=make_lifespan(_load),
)
register_health_route(app, _CONFIG)


@app.post("/chat")
def chat(
    body: ChatRequest,
    client_info: dict[str, Any] = Depends(get_client_info),
):
    state: AppState = app.state.ml
    messages = [m.model_dump() for m in body.conversation]
    try:
        tic = time.time()
        reply = model_chat(
            state.bundle,
            messages,
            max_new_tokens=body.max_new_tokens,
        )
        processed_time = round(time.time() - tic, 4)
        logger.info("IP: {}; Time: {}s", client_info["ip"], processed_time)
        return {
            "api": "/chat",
            "model": state.bundle.repo_id,
            "model_output": {"text": reply},
            "processed_time": processed_time,
        }
    except Exception as e:
        logger.exception("Error /chat from IP {}: {}", client_info["ip"], e)
        raise api_error("/chat", str(e), status=500) from e


@app.post("/v1/chat/completions")
def openai_chat_completions(
    body: OpenAIChatRequest,
    client_info: dict[str, Any] = Depends(get_client_info),
):
    if body.stream:
        raise api_error(
            "/v1/chat/completions",
            "stream=true is not supported yet",
            status=400,
        )

    state: AppState = app.state.ml
    max_new_tokens = (
        body.max_tokens
        or body.max_new_tokens
        or int(_CONFIG["run"].get("max_new_tokens", 512))
    )
    messages = [m.model_dump() for m in body.messages]
    try:
        tic = time.time()
        reply = model_chat(
            state.bundle,
            messages,
            max_new_tokens=max_new_tokens,
        )
        processed_time = round(time.time() - tic, 4)
        logger.info(
            "IP: {}; Time: {}s; openai",
            client_info["ip"],
            processed_time,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model or state.bundle.repo_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except Exception as e:
        logger.exception(
            "Error /v1/chat/completions from IP {}: {}",
            client_info["ip"],
            e,
        )
        raise api_error(
            "/v1/chat/completions",
            str(e),
            status=500,
        ) from e
