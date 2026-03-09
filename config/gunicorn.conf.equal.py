# RTFM -> http://docs.gunicorn.org/en/latest/settings.html#settings
import os
import sys
import subprocess

# Extract service name from app argument 
# (e.g., "app.qwen3_vl.server:app" -> "qwen3_vl")
service_name = 'app'
for arg in sys.argv:
    if arg.startswith('app.') and '.server:app' in arg:
        # Extract the part between "app." and ".server"
        parts = arg.split('.')
        if len(parts) >= 3 and parts[0] == 'app':
            service_name = parts[1]  # e.g., "qwen3_vl"
            break
pidfile = f"{service_name}_{os.getpid()}.pid"
print(f"{pidfile = }")

worker_class = "uvicorn.workers.UvicornWorker"
bind = '0.0.0.0:8000'
workers = 1
timeout = 2000

max_requests = 1000
max_requests_jitter = 100


def get_allowed_gpu_indices():
    """从 CUDA_VISIBLE_DEVICES 解析允许的 GPU 索引。None 表示不限制。"""
    val = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not val:
        return None
    indices = []
    for part in val.split(","):
        part = part.strip()
        if part:
            try:
                indices.append(int(part))
            except ValueError:
                pass
    return frozenset(indices) if indices else None


def get_gpu_indices_from_nvidia_smi():
    """从 nvidia-smi 获取所有 GPU 索引。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []


def select_gpu_with_least_workers(allocated_gpus, allowed_indices=None):
    """
    选出当前实例数最少的 GPU，保证均匀分配；新加 worker 也分配到实例最少的 GPU。
    Args:
        allocated_gpus: {worker_id: gpu_index}
        allowed_indices: CUDA_VISIBLE_DEVICES 范围；None 表示全部
    """
    gpus = allowed_indices if allowed_indices is not None else frozenset(get_gpu_indices_from_nvidia_smi())
    if not gpus:
        return None
    counts = {g: 0 for g in gpus}
    for gpu in allocated_gpus.values():
        if gpu in counts:
            counts[gpu] += 1
    min_count = min(counts.values())
    for gpu in sorted(gpus):
        if counts[gpu] == min_count:
            return gpu
    return None


def on_starting(server):
    """
    Attach a set of IDs that can be temporarily re-used.
    Used on reloads when each worker exists twice.
    Initialize GPU allocation tracking.
    """
    server._worker_id_overload = set()
    server._allocated_gpus = {}

 
 
def nworkers_changed(server, new_value, old_value):
    """
    Gets called on startup too.
    Set the current number of workers.  Required if we raise the worker count
    temporarily using TTIN because server.cfg.workers won't be updated and if
    one of those workers dies, we wouldn't know the ids go that far.
    """
    server._worker_id_current_workers = new_value
 
 
def _next_worker_id(server):
    """
    If there are IDs open for re-use, take one.  Else look for a free one.
    """
    if server._worker_id_overload:
        return server._worker_id_overload.pop()
 
    in_use = set(w._worker_id for w in server.WORKERS.values() if w.alive)
    free = set(range(1, server._worker_id_current_workers + 1)) - in_use
 
    return free.pop()
 
 
def on_reload(server):
    """
    Add a full set of ids into overload so it can be re-used once.
    Clear GPU allocations.
    """
    server._worker_id_overload = set(range(1, server.cfg.workers + 1))
    server._allocated_gpus = {}
 
 
def pre_fork(server, worker):
    """
    Attach the next free worker_id before forking off.
    将 worker 均匀分配到各 GPU；新加 worker 分配到当前实例最少的 GPU。
    """
    worker._worker_id = _next_worker_id(server)
    allowed = get_allowed_gpu_indices()
    selected_gpu = select_gpu_with_least_workers(server._allocated_gpus, allowed)
    if selected_gpu is not None:
        server._allocated_gpus[worker._worker_id] = selected_gpu
        worker._cuda_device = selected_gpu
    else:
        worker._cuda_device = None


def post_fork(server, worker):
    """
    Put the worker_id into an env variable. Set CUDA_VISIBLE_DEVICES to assigned GPU.
    """
    os.environ["APP_WORKER_ID"] = str(worker._worker_id)
    if hasattr(worker, "_cuda_device") and worker._cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker._cuda_device)


def worker_exit(server, worker):
    """Worker 退出时释放 GPU 分配记录。"""
    if hasattr(server, "_allocated_gpus") and worker._worker_id in server._allocated_gpus:
        server._allocated_gpus.pop(worker._worker_id)
