"""
Smoke tests for Qwen3.5 app.

    pytest test/qwen3_5.py -v -m "not integration"
    QWEN35_TEST_URL=http://127.0.0.1:8001 pytest test/qwen3_5.py -v -m integration
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from app.common import AppState
from model.qwen3_5 import Qwen35Bundle


def _mock_state() -> AppState:
    config = OmegaConf.create(
        {
            "model": {"repo_id": "Qwen/Qwen3.5-2B"},
            "run": {"max_new_tokens": 64},
        }
    )
    bundle = Qwen35Bundle(
        model=MagicMock(),
        tokenizer=MagicMock(),
        model_dir="./cache/Qwen3.5-2B",
        repo_id="Qwen/Qwen3.5-2B",
    )
    return AppState(config=config, bundle=bundle)


@pytest.fixture
def client(monkeypatch):
    def fake_chat(bundle, messages, *, max_new_tokens=512):
        return f"echo:{messages[-1]['content']}:{max_new_tokens}"

    def fake_load_qwen35(**kwargs):
        return _mock_state().bundle

    monkeypatch.setattr("app.qwen3_5.model_chat", fake_chat)
    monkeypatch.setattr("app.qwen3_5.load_qwen35", fake_load_qwen35)
    monkeypatch.setattr("app.qwen3_5.setup_logger", lambda c: None)

    from app.qwen3_5 import app

    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "config" in r.json()


def test_native_chat(client: TestClient):
    r = client.post(
        "/chat",
        json={
            "conversation": [{"role": "user", "content": "hi"}],
            "max_new_tokens": 16,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["api"] == "/chat"
    assert body["model_output"]["text"] == "echo:hi:16"


def test_openai_chat(client: TestClient):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "echo:ping:8"


@pytest.mark.integration
def test_live_native_chat():
    url = os.environ.get("QWEN35_TEST_URL", "http://127.0.0.1:8001")
    try:
        r = requests.post(
            f"{url}/chat",
            json={
                "conversation": [
                    {"role": "user", "content": "Reply with exactly: OK"}
                ],
                "max_new_tokens": 16,
            },
            timeout=180,
        )
    except requests.ConnectionError:
        pytest.skip(f"server not up at {url}")
    assert r.ok, r.text
    assert len(r.json()["model_output"]["text"].strip()) > 0
