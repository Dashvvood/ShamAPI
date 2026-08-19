"""
Smoke tests for Qwen3-VL app.

    pytest test/qwen3_vl.py -v -m "not integration"
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from app.common import AppState
from model.qwen3_vl import Qwen3VLBundle, openai_messages_to_qwen


def _mock_state() -> AppState:
    config = OmegaConf.create(
        {
            "model": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct"},
            "run": {"max_new_tokens": 64},
        }
    )
    bundle = Qwen3VLBundle(
        model=MagicMock(),
        processor=MagicMock(),
        model_dir="./cache/Qwen3-VL-2B-Instruct",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
    )
    return AppState(config=config, bundle=bundle)


@pytest.fixture
def client(monkeypatch):
    def fake_chat(bundle, messages, *, max_new_tokens=512):
        return f"vqa:{max_new_tokens}"

    monkeypatch.setattr("app.qwen3_vl.model_chat", fake_chat)
    monkeypatch.setattr(
        "app.qwen3_vl.load_qwen3_vl",
        lambda **kw: _mock_state().bundle,
    )
    monkeypatch.setattr("app.qwen3_vl.setup_logger", lambda c: None)

    from app.qwen3_vl import app

    with TestClient(app) as c:
        yield c


def test_openai_messages_to_qwen_image():
    msgs = openai_messages_to_qwen(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data/dummy/image01.jpg"},
                    },
                ],
            }
        ]
    )
    assert msgs[0]["content"][1]["type"] == "image"
    assert msgs[0]["content"][1]["image"] == "data/dummy/image01.jpg"


def test_openai_messages_to_qwen_video():
    msgs = openai_messages_to_qwen(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "data/dummy/video01.mp4"},
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "what happens?"},
                ],
            }
        ]
    )
    assert msgs[0]["content"][0]["type"] == "video"
    assert msgs[0]["content"][0]["video"] == "data/dummy/video01.mp4"
    assert msgs[0]["content"][0]["fps"] == 1.0


def test_native_image_chat(client: TestClient):
    r = client.post(
        "/chat",
        json={
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "data/dummy/image01.jpg",
                        },
                        {"type": "text", "text": "描述"},
                    ],
                }
            ],
            "max_new_tokens": 16,
        },
    )
    assert r.status_code == 200
    assert r.json()["model_output"]["text"] == "vqa:16"


def test_openai_image(client: TestClient):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data/dummy/image01.jpg"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "vqa:8"


@pytest.mark.integration
def test_live_image_vqa():
    url = os.environ.get("QWEN3VL_TEST_URL", "http://127.0.0.1:8002")
    try:
        r = requests.post(
            f"{url}/chat",
            json={
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "data/dummy/image01.jpg",
                            },
                            {
                                "type": "text",
                                "text": "用五个字描述这张图",
                            },
                        ],
                    }
                ],
                "max_new_tokens": 32,
            },
            timeout=300,
        )
    except requests.ConnectionError:
        pytest.skip(f"server not up at {url}")
    assert r.ok, r.text
    assert len(r.json()["model_output"]["text"].strip()) > 0
