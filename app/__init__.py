import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

config_dir = os.environ.get("CONFIG", "dev").lower()
print(f"{config_dir = }")

# Start apps via module path, e.g.:
#   uvicorn app.qwen3_5:app
#   uvicorn app.qwen3_vl:app --port 8002
#   gunicorn -c config/gunicorn.conf.py app.qwen3_5:app
