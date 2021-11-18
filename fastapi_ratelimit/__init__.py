import random
import string
from typing import Any

from starlette.requests import Request

from .errors import RateLimitExceededError, rate_limit_exceeded_handler
from .limiter import Limiter
from .middleware import RateLimitMiddleware


settings_limit: Any = None


def get_client_ip(request: Request) -> str:
    headers = request.headers
    if "x-forwarded-for" in headers:
        ip_list = headers["x-forwarded-for"]
        # Only get client ip
        ip: str = ip_list.split(",")[0]
    elif "x-real-ip" in headers:
        ip = headers["x-real-ip"]
    else:
        ip = request.client.host
    return ip


def _get_client_identity(request: Request) -> str:
    auth = request.headers.get("Authorization", "").replace(" ", "")
    ip = get_client_ip(request)
    return f"{ip}_{auth}"


def _get_key_prefix() -> str:
    key = "ratelimit"
    if settings_limit and settings_limit.TESTING:
        suffix = "".join(random.choice(string.ascii_uppercase) for _ in range(5))  # nosec
        key = f"{key}_{suffix}"
    return key


def set_settings(settings: Any) -> None:
    global settings_limit
    settings_limit = settings


limiter = Limiter(settings=settings_limit, key_func=_get_client_identity, headers_enabled=True, key_prefix=_get_key_prefix())