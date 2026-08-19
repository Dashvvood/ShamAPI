from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """与现有 HTTPException detail 字段对齐。"""

    api: str
    error: str


class ModelOutputText(BaseModel):
    text: str


class ChatResponse(BaseModel):
    api: str
    model: str | None = None
    model_output: ModelOutputText | dict[str, Any]
    processed_time: float


class HealthResponse(BaseModel):
    pid: int
    worker_id: str | None = None
    config: dict[str, Any]
    client_info: dict[str, Any]
