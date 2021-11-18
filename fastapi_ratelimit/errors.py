from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from .wrappers import Limit


class RateLimitExceededError(HTTPException):
    """
    exception raised when a rate limit is hit.
    """

    limit = None

    def __init__(self, limit: Limit) -> None:
        self.limit = limit
        if limit.error_message:
            description: str = limit.error_message if not callable(limit.error_message) else limit.error_message()
        else:
            description = str(limit.limit)
        super().__init__(status_code=429, detail=description)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceededError) -> JSONResponse:
    response = JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "result": dict(msg=f"Rate limit exceeded: {exc.detail}", type="429"),
        },
    )
    response = request.app.state.limiter.inject_headers(response, request.state.view_rate_limit)
    return response
