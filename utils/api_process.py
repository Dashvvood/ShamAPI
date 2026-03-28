from fastapi import Request

def get_client_info(request: Request):
    """依赖项：获取客户端信息"""
    def get_ip():
        ip_address = request.headers.get("X-Real-IP")
        if not ip_address:
            ip_address = request.headers.get("X-Forwarded-For")
            if ip_address:
                ip_address = ip_address.split(",")[0].strip()
        if not ip_address:
            ip_address = request.client.host if request.client else "unknown"
        return ip_address
    
    def get_user_agent():
        return request.headers.get("User-Agent", "unknown")
    
    return {
        "ip": get_ip(),
        "user_agent": get_user_agent(),
        "method": request.method,
        "url": str(request.url)
    }

def _download_from_huggingface(model_name: str, local_dir: str | None = None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_name, local_dir=local_dir or model_name)

def _download_from_modelscope(model_name: str, local_dir: str | None = None):
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(model_id=model_name, local_dir=local_dir or model_name)


def download_model(
    model_name: str,
    local_dir: str | None = None,
    provider: str = "auto",
):
    provider = provider.lower().strip()
    if provider in {"auto", "hf", "huggingface"}:
        try:
            return _download_from_huggingface(model_name=model_name, local_dir=local_dir)
        except Exception:
            return _download_from_modelscope(model_name=model_name, local_dir=local_dir)
    if provider in {"ms", "modelscope"}:
        return _download_from_modelscope(model_name=model_name, local_dir=local_dir)
    raise ValueError("provider must be one of: auto, huggingface, modelscope")
