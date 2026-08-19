"""
示例：用 app/common 重构 qwen3_vl 的写法（不替换现有 server.py，仅供对照）。

启动:
    CONFIG=dev uvicorn app.common.example_qwen3_vl:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from app import QWEN3_VL_CONFIG_PATH
from app.common.bootstrap import load_config, register_health_route, setup_logger
from app.common.lifespan import AppState, make_lifespan
from app.common.schemas import ChatResponse, ErrorDetail, ModelOutputText
from utils.api_process import download_model, get_client_info

_CONFIG = load_config(QWEN3_VL_CONFIG_PATH)


def _load_models() -> AppState:
    setup_logger(_CONFIG)
    logger.info("Loading Qwen3-VL model...")

    model_dir = _CONFIG["model"]["model_dir"]
    if not os.path.exists(model_dir):
        download_model(_CONFIG["model"]["repo_id"], model_dir, provider="auto")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        _CONFIG["model"]["model_dir"],
        device_map=_CONFIG["model"]["device_map"],
        torch_dtype=getattr(torch, _CONFIG["model"]["torch_type"]),
    )
    processor = AutoProcessor.from_pretrained(_CONFIG["model"]["model_dir"])
    logger.info("Model and processor loaded.")
    return AppState(config=_CONFIG, model=model, processor=processor)


app = FastAPI(
    title="Qwen3-VL (common example)",
    version="0.1",
    lifespan=make_lifespan(_load_models),
)
register_health_route(app, _CONFIG)


@app.post("/chat", response_model=ChatResponse)
def chat(
    message: dict[str, Any],
    client_info: dict[str, Any] = Depends(get_client_info),
):
    state: AppState = app.state.ml
    model, processor, config = state.model, state.processor, state.config

    conversation = message.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise HTTPException(
            status_code=422,
            detail=ErrorDetail(
                api="/chat",
                error="body must include non-empty list field 'conversation'",
            ).model_dump(),
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
        processed_time = round(time.time() - tic, 4)
        logger.info("IP: {}; Time: {}s", client_info["ip"], processed_time)
        return ChatResponse(
            api="/chat",
            model=config["model"]["repo_id"],
            model_output=ModelOutputText(text=output_text[0]),
            processed_time=processed_time,
        )
    except Exception as e:
        logger.exception("Error processing /chat from IP {}: {}", client_info["ip"], e)
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(api="/chat", error=str(e)).model_dump(),
        ) from e
