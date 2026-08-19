import os
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI
from omegaconf import DictConfig, OmegaConf

from utils.api_process import get_client_info


def load_config(path: str) -> DictConfig:
    return OmegaConf.load(path)


def setup_logger(config: DictConfig):
    from loguru import logger

    logger.remove()
    logger.add(**config["log"])
    return logger


def register_health_route(
    app: FastAPI,
    config: DictConfig,
    *,
    get_config: Callable[[], DictConfig] | None = None,
) -> None:
    """注册 GET / 健康检查与配置快照。"""

    @app.get("/")
    async def index(client_info: dict[str, Any] = Depends(get_client_info)):
        cfg = get_config() if get_config else config
        return {
            "pid": os.getpid(),
            "worker_id": os.environ.get("APP_WORKER_ID"),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "client_info": client_info,
        }
