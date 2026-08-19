from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Qwen35Bundle:
    """Loaded Qwen3.5 causal-LM bundle."""

    model: Any
    tokenizer: Any
    model_dir: str
    repo_id: str


def resolve_dtype(name: str):
    name = str(name).lower().strip()
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def load_qwen35(
    *,
    model_dir: str,
    repo_id: str = "Qwen/Qwen3.5-2B",
    device_map: str = "auto",
    torch_type: str = "bfloat16",
) -> Qwen35Bundle:
    dtype = resolve_dtype(torch_type)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map=device_map,
        dtype=dtype,
    )
    return Qwen35Bundle(
        model=model,
        tokenizer=tokenizer,
        model_dir=model_dir,
        repo_id=repo_id,
    )


def chat(
    bundle: Qwen35Bundle,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
) -> str:
    """
    Text chat. Each message: {"role": "...", "content": "..."}.
    """
    tokenizer = bundle.tokenizer
    model = bundle.model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = model_inputs.to(model.device)
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    in_len = model_inputs.input_ids.shape[1]
    output_ids = generated[0][in_len:].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)
