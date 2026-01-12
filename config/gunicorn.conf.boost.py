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


def get_gpu_memory_info():
    """
    获取所有 GPU 的显存信息
    返回: List[Dict] 每个字典包含 {'index': int, 'total': int, 'used': int, 'free': int}
    """
    try:
        # 使用 nvidia-smi 获取 GPU 信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpu_info.append({
                    'index': int(parts[0]),
                    'total': int(parts[1]),
                    'used': int(parts[2]),
                    'free': int(parts[3])
                })
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Failed to get GPU info: {e}")
        return []


def select_gpu_with_most_free_memory(exclude_indices=None):
    """
    选择空闲显存最大的 GPU
    Args:
        exclude_indices: 要排除的 GPU 索引列表
    Returns:
        GPU 索引，如果没有可用 GPU 则返回 None
    """
    exclude_indices = exclude_indices or []
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        return None
    
    # 过滤掉已分配的 GPU
    available_gpus = [gpu for gpu in gpu_info if gpu['index'] not in exclude_indices]
    
    if not available_gpus:
        return None
    
    # 按空闲显存降序排序，选择最大的
    best_gpu = max(available_gpus, key=lambda x: x['free'])
    return best_gpu['index']


def on_starting(server):
    """
    Attach a set of IDs that can be temporarily re-used.
    Used on reloads when each worker exists twice.
    Also, set pidfile to use the actual process ID.
    Initialize GPU allocation tracking.
    """
    server._worker_id_overload = set()
    server._allocated_gpus = {}  # 跟踪每个 worker_id 分配的 GPU

 
 
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
    Clear GPU allocations on reload.
    """
    server._worker_id_overload = set(range(1, server.cfg.workers + 1))
    server._allocated_gpus = {}
 
 
def pre_fork(server, worker):
    """
    Attach the next free worker_id before forking off.
    Allocate GPU with most free memory to this worker.
    """
    worker._worker_id = _next_worker_id(server)
    
    # 选择空闲显存最大的 GPU
    exclude_gpus = list(server._allocated_gpus.values())
    selected_gpu = select_gpu_with_most_free_memory(exclude_gpus)
    
    if selected_gpu is not None:
        server._allocated_gpus[worker._worker_id] = selected_gpu
        worker._cuda_device = selected_gpu
        print(f"Worker {worker._worker_id} assigned GPU {selected_gpu}")
    else:
        # 如果没有可用 GPU，使用默认值或抛出警告
        print(f"Warning: No GPU available for worker {worker._worker_id}, using default CUDA_VISIBLE_DEVICES")
        worker._cuda_device = None


def post_fork(server, worker):
    """
    Put the worker_id into an env variable for further use within the app.
    Set CUDA_VISIBLE_DEVICES to the allocated GPU.
    """
    os.environ["APP_WORKER_ID"] = str(worker._worker_id)
    
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    if hasattr(worker, '_cuda_device') and worker._cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker._cuda_device)
        print(f"Worker {worker._worker_id} PID {os.getpid()}: CUDA_VISIBLE_DEVICES={worker._cuda_device}")
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        # 如果没有分配 GPU 且环境变量未设置，保持原样或设置默认值
        pass


def worker_exit(server, worker):
    """
    Clean up GPU allocation when worker exits.
    """
    if hasattr(server, '_allocated_gpus') and worker._worker_id in server._allocated_gpus:
        gpu_id = server._allocated_gpus.pop(worker._worker_id)
        print(f"Worker {worker._worker_id} exited, released GPU {gpu_id}")
