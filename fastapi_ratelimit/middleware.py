from typing import cast

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

from .errors import rate_limit_exceeded_handler
from .limiter import Limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        app: Starlette = request.app
        limiter: Limiter = app.state.limiter  # type: ignore
        handler = None

        for route in app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL and hasattr(route, "endpoint"):
                handler = route.endpoint  # type: ignore
                break

        # There's no handle for this request
        if handler is None:
            return await call_next(request)

        name = "%s.%s" % (handler.__module__, handler.__name__)
        # if exempt no need to check or decorator
        if name in limiter.exempt_routes or name in limiter.route_limits:
            return await call_next(request)

        # global rate limit check
        if limiter.auto_check and not getattr(request.state, "rate_limiting_complete", False):
            try:
                limiter.check_request_limit(request, handler, True)
            except Exception as e:  # pylint: disable=broad-except
                # handle the exception since the global exception handler won't pick it up if we call_next
                exception_handler = app.exception_handlers.get(type(e), rate_limit_exceeded_handler)  # type: ignore
                response = exception_handler(request, e)
                return cast(Response, response)
            response = await call_next(request)
            response = limiter.inject_headers(response, request.state.view_rate_limit)
            return cast(Response, response)
        return await call_next(request)
