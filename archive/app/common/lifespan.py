from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import FastAPI
from omegaconf import DictConfig

T = TypeVar("T")


@dataclass
class AppState:
    """挂在 app.state 上的运行时状态。"""

    config: DictConfig
    model: Any
    processor: Any | None = None
    extra: dict[str, Any] | None = None


def make_lifespan(loader: Callable[[], T], *, state_key: str = "ml") -> Callable:
    """
    将阻塞的模型加载推迟到 worker 启动时，避免 import server 模块时就占 GPU。

    loader: 无参函数，返回 AppState 或任意对象，结果写入 app.state.<state_key>。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setattr(app.state, state_key, loader())
        yield

    return lifespan
