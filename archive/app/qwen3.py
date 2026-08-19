from __future__ import annotations

import os
import time
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import QWEN3_CONFIG_PATH
from .common.bootstrap import (
    load_config,
    register_health_route,
    setup_logger,
)
from .common.lifespan import AppState, make_lifespan
from .common.schemas import ChatResponse, ErrorDetail, ModelOutputText
from utils.api_process import download_model, get_client_info

_CONFIG = load_config(QWEN3_CONFIG_PATH)


def _load_models() -> AppState:
    setup_logger(_CONFIG)
    logger.info("Loading Qwen3 model...")

    model_dir = _CONFIG["model"]["model_dir"]
    if not os.path.exists(model_dir):
        download_model(
            _CONFIG["model"]["repo_id"],
            model_dir,
            provider="auto",
        )

    tt = str(_CONFIG["model"]["torch_type"]).lower()
    if tt == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, _CONFIG["model"]["torch_type"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=_CONFIG["model"]["device_map"],
        torch_dtype=torch_dtype,
    )
    logger.info("Model and tokenizer loaded.")
    return AppState(config=_CONFIG, model=model, processor=tokenizer)


app = FastAPI(
    title="Qwen3",
    version="0.1",
    lifespan=make_lifespan(_load_models),
)
register_health_route(app, _CONFIG)


@app.post("/chat", response_model=ChatResponse)
def chat(
    message: dict[str, Any],
    client_info: dict[str, Any] = Depends(get_client_info),
):
    """
    Text-only chat. `conversation` is passed to `apply_chat_template` as-is;
    each message must use string `content` (Hugging Face chat format).

        {
            "conversation": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": "..."},
            ],
            "max_new_tokens": 512
        }
    """
    state: AppState = app.state.ml
    model, tokenizer, config = state.model, state.processor, state.config

    conversation = message.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(
                api="/chat",
                error=(
                    "body must include non-empty list field "
                    "'conversation'"
                ),
            ).model_dump(),
        )

    max_new_tokens = message.get("max_new_tokens", 2048)
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(
                api="/chat",
                error=(
                    "max_new_tokens must be a positive integer "
                    "when provided"
                ),
            ).model_dump(),
        )

    try:
        tic = time.time()
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(
            [text], return_tensors="pt"
        ).to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        in_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][in_len:].tolist()
        reply = tokenizer.decode(output_ids, skip_special_tokens=True)
        processed_time = round(time.time() - tic, 4)
        logger.info(
            "IP: {}; Time: {}s",
            client_info["ip"],
            processed_time,
        )
        return ChatResponse(
            api="/chat",
            model=config["model"]["repo_id"],
            model_output=ModelOutputText(text=reply),
            processed_time=processed_time,
        )
    except Exception as e:
        logger.exception(
            "Error processing /chat from IP {}: {}",
            client_info["ip"],
            e,
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(api="/chat", error=str(e)).model_dump(),
        ) from e
