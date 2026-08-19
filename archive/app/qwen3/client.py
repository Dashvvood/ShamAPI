from elinor import o_d; o_d = o_d()
import argparse
import json
import os
import sys

import requests

parser = argparse.ArgumentParser(
    description="POST /chat to Qwen3 text API (string content per message, see server docstring).",
)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
    help="User message (maps to role=user).",
)
parser.add_argument(
    "-s",
    "--system",
    type=str,
    default="You are a helpful assistant.",
    help="System message; pass empty string to send only the user turn.",
)
parser.add_argument(
    "--url",
    type=str,
    default="http://127.0.0.1:8000/chat",
    help="Full URL of POST /chat.",
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=None,
    dest="max_new_tokens",
    metavar="N",
    help="Optional max_new_tokens; if omitted, server uses its default (2048).",
)
args = parser.parse_args()

conversation = []
if args.system.strip():
    conversation.append({"role": "system", "content": args.system})
conversation.append({"role": "user", "content": args.prompt})

message: dict = {"conversation": conversation}
if args.max_new_tokens is not None:
    message["max_new_tokens"] = args.max_new_tokens

response = requests.post(args.url, json=message)

try:
    body = response.json()
except json.JSONDecodeError:
    print(response.text, file=sys.stderr)
    sys.exit(1)

if not response.ok:
    print(body, file=sys.stderr)
    sys.exit(response.status_code)

# output_path = f"qwen3_{o_d.strftime('%Y%m%d-%H%M%S')}.json"
# out_dir = os.environ.get("QWEN3_CLIENT_OUT_DIR", ".")
# os.makedirs(out_dir, exist_ok=True)
# output_path = os.path.join(out_dir, os.path.basename(output_path))

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(body, f, indent=4, ensure_ascii=False)

print(body)
print(body.keys())
