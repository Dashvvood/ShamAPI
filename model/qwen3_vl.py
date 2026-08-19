from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


@dataclass
class Qwen3VLBundle:
    model: Any
    processor: Any
    model_dir: str
    repo_id: str


def resolve_dtype(name: str):
    name = str(name).lower().strip()
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def load_qwen3_vl(
    *,
    model_dir: str,
    repo_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    device_map: str = "auto",
    torch_type: str = "bfloat16",
) -> Qwen3VLBundle:
    dtype = resolve_dtype(torch_type)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=device_map,
    )
    return Qwen3VLBundle(
        model=model,
        processor=processor,
        model_dir=model_dir,
        repo_id=repo_id,
    )


def chat(
    bundle: Qwen3VLBundle,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 512,
) -> str:
    """
    Multimodal chat (text / image / video).

    Example:
        [{"role": "user", "content": [
            {"type": "image", "image": "path_or_url"},
            {"type": "text", "text": "描述这张图"},
        ]}]
    """
    processor = bundle.processor
    model = bundle.model
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def openai_messages_to_qwen(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map OpenAI vision content parts to Qwen3-VL conversation."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = part.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                parts.append({"type": "image", "image": url})
            elif ptype == "image":
                parts.append(
                    {"type": "image", "image": part.get("image") or part.get("url")}
                )
            elif ptype in {"video_url", "video"}:
                url = part.get("video_url", part.get("video", {}))
                if isinstance(url, dict):
                    url = url.get("url", url.get("video", ""))
                video_part: dict[str, Any] = {"type": "video", "video": url}
                if "fps" in part:
                    video_part["fps"] = part["fps"]
                if "video_start" in part:
                    video_part["video_start"] = part["video_start"]
                if "video_end" in part:
                    video_part["video_end"] = part["video_end"]
                parts.append(video_part)
        out.append({"role": role, "content": parts})
    return out
