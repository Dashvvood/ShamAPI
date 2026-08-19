"""Shared helpers for ShamAPI FastAPI apps."""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from utils.api_process import get_client_info


def load_config(path: str) -> DictConfig:
    return OmegaConf.load(path)


def setup_logger(config: DictConfig):
    from loguru import logger

    logger.remove()
    logger.add(**config["log"])
    return logger


@dataclass
class AppState:
    config: DictConfig
    bundle: Any


def make_lifespan(loader: Callable[[], AppState]):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ml = loader()
        yield

    return lifespan


def register_health_route(app: FastAPI, config: DictConfig) -> None:
    @app.get("/")
    async def index(client_info: dict = Depends(get_client_info)):
        return {
            "pid": os.getpid(),
            "worker_id": os.environ.get("APP_WORKER_ID"),
            "config": OmegaConf.to_container(config, resolve=True),
            "client_info": client_info,
        }


def api_error(api: str, error: str, status: int = 422) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"api": api, "error": error},
    )


class ChatMessage(BaseModel):
    role: str
    content: Any  # str or multimodal list


class ChatRequest(BaseModel):
    conversation: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=512, ge=1)


class OpenAIChatMessage(BaseModel):
    role: str
    content: Any  # str or multimodal list


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_new_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
