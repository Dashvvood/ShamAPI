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
