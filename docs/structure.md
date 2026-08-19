# ShamAPI 架构

基于 **Python + Transformers** 的简单部署；不使用 vLLM。  
LiteLLM 仅作为可选的 **OpenAI 兼容网关**，不进入 `model/` / `app/` 内部实现。

## 分层与目录

| 目录 | 职责 |
|------|------|
| `model/` | 各模型的 load 与基础调用（transformers） |
| `app/` | FastAPI 接入：单模型薄封装 / 增加业务动作（如复杂视频）/ 多模型 pipeline |
| `config/` | 统一 YAML 参数；含各 app 配置与可选的 LiteLLM 网关配置 |
| `utils/` | 通用工具函数 |
| `log/` | 按不同 app 分别输出日志 |

## LiteLLM 加在哪里

加在 **「OpenAI 客户端」与「各 FastAPI app」之间**，作为独立网关进程：

- 对外：统一端口，OpenAI 格式（如 `POST /v1/chat/completions`）
- 对内：按 `model` 名路由到一个或多个 `app` HTTP 地址
- 可选：同一逻辑模型的多副本负载均衡

**不**在 `model/` 里调用 LiteLLM；**不**用 LiteLLM 替代 FastAPI 业务编排。

```text
其他应用 / OpenAI SDK
        │  :8080  OpenAI 格式
        ▼
   LiteLLM Proxy          ← 网关：协议统一、路由、多副本
        │
        ├─ app A  :8001
        ├─ app A  :8002   （相同能力的副本）
        └─ app B  :8003   （增强能力或 pipeline）
              │
              ▼
         model/*          ← 纯 transformers load / generate

旁路：config/*.yaml 、 utils/ 、 log/<app>.log
```

## 两种调用路径

1. **需要 OpenAI / 同端口多 app** → Client → LiteLLM → `app/`
   - 要求对应 app 提供 OpenAI 兼容接口（推荐 `POST /v1/chat/completions`），内部再调 `model/`。
2. **自定义能力（视频、非 chat）** → Client → 直连 `app/`
   - 不必硬套 OpenAI；pipeline 仍在 `app/` 完成。

## app/ 的三种形态

- **薄封装**：几乎不改逻辑，加载单个 `model` 并对外服务
- **增强**：在单模型上增加动作（预处理、复杂视频等）
- **Pipeline**：组合多个 `model`；若最终输出是对话文本，可再挂到 LiteLLM 的一个 `model` 名

## config/ 约定

- `config/<env>/<app>.yaml`：该 app / model 的路径、device、dtype、日志等
- `config/litellm.yaml`（可选）：`model` 别名 → 各 `app` 的 `api_base` 列表

## 原则摘要

1. 推理只走 **transformers**（经 `model/`）。
2. 业务与 HTTP 只走 **`app/`（FastAPI）**。
3. **LiteLLM = 网关插件**，不是第四种模型后端。
4. 多副本同端口：多个相同 `app` 进程 + LiteLLM 同名 `model` 多条 upstream。
5. 日志按 app 落在 `log/`；网关可另有自身访问日志。

## 共享代码

- `app/common.py`：配置/日志/lifespan/健康检查/请求 schema/`api_error`（单文件，尽量薄）

## 参考实现：Qwen3.5-2B（文本）

| 路径 | 说明 |
|------|------|
| `model/qwen3_5.py` | load + `chat()` |
| `app/qwen3_5.py` | FastAPI：`/chat` + `/v1/chat/completions` |
| `config/dev/qwen3_5.yaml` | `./cache/Qwen3.5-2B`，默认端口 8001 |

## 参考实现：Qwen3-VL-2B（图像 / 视频 VQA）

| 路径 | 说明 |
|------|------|
| `model/qwen3_vl.py` | load + multimodal `chat()` + OpenAI→Qwen 映射 |
| `app/qwen3_vl.py` | FastAPI：`/chat` + `/v1/chat/completions` |
| `config/dev/qwen3_vl.yaml` | `./cache/Qwen3-VL-2B-Instruct`，默认端口 8002 |

```bash
CONFIG=dev uvicorn app.qwen3_vl:app --host 0.0.0.0 --port 8002

# 图像 VQA
curl -s http://127.0.0.1:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation":[{"role":"user","content":[
        {"type":"image","image":"data/dummy/image01.jpg"},
        {"type":"text","text":"描述这张图"}
      ]}],"max_new_tokens":64}'

# 视频 VQA
curl -s http://127.0.0.1:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation":[{"role":"user","content":[
        {"type":"video","video":"data/dummy/video01.mp4","fps":1.0},
        {"type":"text","text":"视频里发生了什么"}
      ]}],"max_new_tokens":64}'
```

## Qwen3.5 启动示例

```bash
# 启动 app
CONFIG=dev ./run.sh
# 或
CONFIG=dev uvicorn app.qwen3_5:app --host 0.0.0.0 --port 8001

# 原生问答
curl -s http://127.0.0.1:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation":[{"role":"user","content":"你好"}],"max_new_tokens":64}'

# OpenAI 兼容（供 LiteLLM / SDK）
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

# 可选网关
litellm --config config/litellm.yaml --host 0.0.0.0 --port 8080
```
