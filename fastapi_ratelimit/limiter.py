# pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-branches,no-else-return
import asyncio
import datetime
import functools
import inspect
import itertools
import logging
import time
import os
from email.utils import formatdate, parsedate_to_datetime
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

from limits import RateLimitItem
from limits.errors import ConfigurationError
from limits.storage import MemoryStorage, Storage, storage_from_string
from limits.strategies import STRATEGIES, RateLimiter
from starlette.requests import Request
from starlette.responses import Response

from .errors import RateLimitExceededError
from .wrappers import Limit, LimitGroup

# used to annotate get_setting method
T = TypeVar("T")

AnyCallable = Callable[..., Any]

# Define an alias for the most commonly used type
StrOrCallableStr = Union[str, Callable[..., str]]


class Settings:
    RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", True)
    RATELIMIT_HEADERS_ENABLED = os.getenv("RATELIMIT_HEADERS_ENABLED", None)
    RATELIMIT_STORAGE_URL = os.getenv("RATELIMIT_STORAGE_URL", "")
    RATELIMIT_STORAGE_OPTIONS = os.getenv("RATELIMIT_STORAGE_OPTIONS", None)
    RATELIMIT_STRATEGY = os.getenv("RATELIMIT_STRATEGY", "fixed-window")
    RATELIMIT_GLOBAL = os.getenv("RATELIMIT_GLOBAL", None)
    RATELIMIT_APPLICATION = os.getenv("RATELIMIT_APPLICATION", None)
    RATELIMIT_IN_MEMORY_FALLBACK = os.getenv("RATELIMIT_IN_MEMORY_FALLBACK", None)
    RATELIMIT_IN_MEMORY_FALLBACK_ENABLED = os.getenv("RATELIMIT_IN_MEMORY_FALLBACK_ENABLED", None)


class C:
    ENABLED = "RATELIMIT_ENABLED"
    HEADERS_ENABLED = "RATELIMIT_HEADERS_ENABLED"
    STORAGE_URL = "RATELIMIT_STORAGE_URL"
    STORAGE_OPTIONS = "RATELIMIT_STORAGE_OPTIONS"
    STRATEGY = "RATELIMIT_STRATEGY"
    GLOBAL_LIMITS = "RATELIMIT_GLOBAL"
    APPLICATION_LIMITS = "RATELIMIT_APPLICATION"
    IN_MEMORY_FALLBACK = "RATELIMIT_IN_MEMORY_FALLBACK"
    IN_MEMORY_FALLBACK_ENABLED = "RATELIMIT_IN_MEMORY_FALLBACK_ENABLED"


class HEADERS:
    RESET = 1
    REMAINING = 2
    LIMIT = 3
    RETRY_AFTER = 4


MAX_BACKEND_CHECKS = 5


def get_setting(key: str, default_value: Optional[T] = None) -> T:
    # handle case on ci where settings.key == empty string
    settings = Settings()
    setting = getattr(settings, key, default_value) or default_value
    return cast(T, setting)

def _get_client_identity(request: Request) -> str:
    auth = request.headers.get("Authorization", "").replace(" ", "")
    ip = get_client_ip(request)
    return f"{ip}_{auth}"


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

class Limiter:
    """
    Initializes the limiter rate limiter.

    ** parameter **

    * **app**: `Starlette/FastAPI` instance to initialize the extension
     with.

    * **default_limits**: a variable list of strings or callables returning strings denoting global
     limits to apply to all routes. `ratelimit-string` for  more details.

    * **application_limits**: a variable list of strings or callables returning strings for limits that
     are applied to the entire application (i.e a shared limit for all routes)

    * **key_func**: a callable that returns the domain to rate limit by.

    * **headers_enabled**: whether ``X-RateLimit`` response headers are written.

    * **strategy:** the strategy to use. refer to `ratelimit-strategy`

    * **storage_uri**: the storage location. refer to `ratelimit-conf`

    * **storage_options**: kwargs to pass to the storage implementation upon
      instantiation.
    * **auto_check**: whether to automatically check the rate limit in the before_request
     chain of the application. default ``True``
    * **swallow_errors**: whether to swallow errors when hitting a rate limit.
     An exception will still be logged. default ``False``
    * **in_memory_fallback**: a variable list of strings or callables returning strings denoting fallback
     limits to apply when the storage is down.
    * **in_memory_fallback_enabled**: simply falls back to in memory storage
     when the main storage is down and inherits the original limits.
    * **key_prefix**: prefix prepended to rate limiter keys.
    * **config_filename**: name of the config file for Starlette from which to load settings
     for the rate limiter. Defaults to ".env".
    """

    def __init__(
        self,
        key_func: Callable[..., str] = _get_client_identity,
        default_limits: List[StrOrCallableStr] = [],
        application_limits: List[StrOrCallableStr] = [],
        headers_enabled: bool = False,
        enabled: bool = True,
        swallow_errors: bool = False,
        key_prefix: str = "ratelimit",
        auto_check: bool = True,
        in_memory_fallback_enabled: bool = False,
        strategy: Optional[str] = None,
        storage_uri: Optional[str] = None,
        storage_options: Dict[str, Any] = {},
        in_memory_fallback: List[StrOrCallableStr] = [],
        retry_after: Optional[str] = None,
        limit_default: Optional[str] = "60/minute"
    ) -> None:
        """
        Configure the rate limiter at app level
        """
        self.logger = logging.getLogger("limiter")
        self.enabled = enabled
        self.exempt_routes: Set[str] = set()
        self.route_limits: Dict[str, List[Limit]] = {}
        self.auto_check = auto_check

        self._default_limits = []
        self._application_limits = []
        self._in_memory_fallback: List[LimitGroup] = []
        self._in_memory_fallback_enabled = in_memory_fallback_enabled or len(in_memory_fallback) > 0
        self._request_filters: List[Callable[..., bool]] = []
        self._headers_enabled = headers_enabled
        self._header_mapping: Dict[int, str] = {}
        self._retry_after: Optional[str] = retry_after
        self._storage_options = storage_options or get_setting(C.STORAGE_OPTIONS, {})
        self._swallow_errors = swallow_errors
        self._key_func = key_func
        self._key_prefix = key_prefix
        self._strategy = strategy or get_setting(C.STRATEGY, "fixed-window")
        self._storage_uri = storage_uri or get_setting(C.STORAGE_URL, "memory://")
        self._update_header_map()

        for limit in set(default_limits):
            self._default_limits.extend([LimitGroup(limit, self._key_func)])
        for limit in application_limits:
            self._application_limits.extend([LimitGroup(limit, self._key_func, scope="global")])
        for limit in in_memory_fallback:
            self._in_memory_fallback.extend([LimitGroup(limit, self._key_func)])

        self._dynamic_route_limits: Dict[str, List[LimitGroup]] = {}
        # a flag to note if the storage backend is dead (not available)
        self._storage_dead: bool = False
        self._fallback_limiter = None
        self.__check_backend_count = 0
        self.__last_check_backend = time.time()
        self.__marked_for_limiting: Dict[str, List[AnyCallable]] = {}
        self._storage: Storage = storage_from_string(self._storage_uri, **self._storage_options)

        if self._strategy not in STRATEGIES:
            raise ConfigurationError("Invalid rate limiting strategy %s" % self._strategy)

        self._limiter: RateLimiter = STRATEGIES[self._strategy](self._storage)

        app_limits: Optional[StrOrCallableStr] = get_setting(C.APPLICATION_LIMITS, None)
        if not self._application_limits and app_limits:
            self._application_limits = [LimitGroup(app_limits, self._key_func, scope="global")]

        conf_limits: Optional[StrOrCallableStr] = limit_default
        if not self._default_limits and conf_limits:
            self._default_limits = [LimitGroup(conf_limits, self._key_func)]

        self._init_fallback_limiter()

    def _init_fallback_limiter(self) -> None:
        fallback_enabled = get_setting(C.IN_MEMORY_FALLBACK_ENABLED, False)
        fallback_limits: Optional[StrOrCallableStr] = get_setting(C.IN_MEMORY_FALLBACK, None)
        if not self._in_memory_fallback and fallback_limits:
            self._in_memory_fallback = [LimitGroup(fallback_limits, self._key_func)]
        if not self._in_memory_fallback_enabled:
            self._in_memory_fallback_enabled = fallback_enabled or len(self._in_memory_fallback) > 0

        if self._in_memory_fallback_enabled:
            self._fallback_storage = MemoryStorage()
            self._fallback_limiter = STRATEGIES[self._strategy](self._fallback_storage)

    def _update_header_map(self) -> None:
        self._header_mapping.update(
            {
                HEADERS.RESET: self._header_mapping.get(HEADERS.RESET, "X-RateLimit-Reset"),
                HEADERS.REMAINING: self._header_mapping.get(HEADERS.REMAINING, "X-RateLimit-Remaining"),
                HEADERS.LIMIT: self._header_mapping.get(HEADERS.LIMIT, "X-RateLimit-Limit"),
                HEADERS.RETRY_AFTER: self._header_mapping.get(HEADERS.RETRY_AFTER, "Retry-After"),
            }
        )

    def _should_check_backend(self) -> bool:
        if self.__check_backend_count > MAX_BACKEND_CHECKS:
            self.__check_backend_count = 0
        if time.time() - self.__last_check_backend > pow(2, self.__check_backend_count):
            self.__last_check_backend = time.time()
            self.__check_backend_count += 1
            return True
        return False

    def reset(self) -> None:
        """
        resets the storage if it supports being reset
        """
        try:
            self._storage.reset()
            self.logger.info("Storage has been reset and all limits cleared")
        except NotImplementedError:
            self.logger.warning("This storage type does not support being reset")

    @property
    def limiter(self) -> RateLimiter:
        """
        The backend that keeps track of consumption of endpoints vs limits
        """
        if self._storage_dead and self._in_memory_fallback_enabled:
            return self._fallback_limiter
        return self._limiter

    def inject_headers(self, response: Response, current_limit: Tuple[RateLimitItem, List[str]]) -> Response:
        if self.enabled and self._headers_enabled and current_limit is not None:
            if not isinstance(response, Response):
                logging.warning("parameter `response` must be an instance of starlette.responses.Response")
                return response

            try:
                window_stats: Tuple[int, int] = self.limiter.get_window_stats(current_limit[0], *current_limit[1])
                reset_in = 1 + window_stats[0]
                response.headers.append(self._header_mapping[HEADERS.LIMIT], str(current_limit[0].amount))
                response.headers.append(self._header_mapping[HEADERS.REMAINING], str(window_stats[1]))
                response.headers.append(self._header_mapping[HEADERS.RESET], str(reset_in))

                # response may have an existing retry after
                existing_retry_after_header = response.headers.get("Retry-After")
                # existing_retry_after_header = None
                if existing_retry_after_header is not None:
                    # might be in http-date format
                    retry_after = parsedate_to_datetime(existing_retry_after_header)

                    # parse_date failure returns None
                    if retry_after is None:
                        retry_after = time.time() + int(existing_retry_after_header)

                    if isinstance(retry_after, datetime.datetime):
                        retry_after_int: int = int(time.mktime(retry_after.timetuple()))

                    reset_in = max(retry_after_int, reset_in)

                response.headers[self._header_mapping[HEADERS.RETRY_AFTER]] = (
                    formatdate(reset_in) if self._retry_after == "http-date" else str(int(reset_in - time.time()))
                )
            except Exception:  # pylint: disable=broad-except
                if self._in_memory_fallback and not self._storage_dead:
                    self.logger.warning("Rate limit storage unreachable - falling back to" " in-memory storage")
                    self._storage_dead = True
                    response = self.inject_headers(response, current_limit)
                if self._swallow_errors:
                    self.logger.exception("Failed to update rate limit headers. Swallowing error")
                else:
                    raise
        return response

    def __evaluate_limits(self, request: Request, endpoint: str, limits: List[Limit]) -> None:
        failed_limit = None
        limit_for_header = None
        for lim in limits:
            limit_scope = lim.scope or endpoint
            if lim.is_exempt:
                continue
            if lim.methods is not None and request.method.lower() not in lim.methods:
                continue
            if lim.per_method:
                limit_scope += ":%s" % request.method

            if "request" in inspect.signature(lim.key_func).parameters.keys():
                limit_key = lim.key_func(request)
            else:
                limit_key = lim.key_func()

            args = [limit_key, limit_scope]
            if all(args):
                if self._key_prefix:
                    args = [self._key_prefix] + args
                if not limit_for_header or lim.limit < limit_for_header[0]:
                    limit_for_header = (lim.limit, args)
                if not self.limiter.hit(lim.limit, *args):
                    self.logger.warning(
                        "ratelimit %s (%s) exceeded at endpoint: %s",
                        lim.limit,
                        limit_key,
                        limit_scope,
                    )
                    failed_limit = lim
                    limit_for_header = (lim.limit, args)
                    break
            else:
                self.logger.error("Skipping limit: %s. Empty value found in parameters.", lim.limit)
                continue
        # keep track of which limit was hit, to be picked up for the response header
        request.state.view_rate_limit = limit_for_header

        if failed_limit:
            raise RateLimitExceededError(failed_limit)

    def check_request_limit(
        self,
        request: Request,
        endpoint_func: AnyCallable,
        in_middleware: bool = True,
    ) -> None:
        """
        Determine if the request is within limits
        """
        endpoint = request["path"] or ""
        view_func = endpoint_func
        name = "%s.%s" % (view_func.__module__, view_func.__name__) if view_func else ""

        # cases where we don't need to check the limits
        if not endpoint or not self.enabled or name in self.exempt_routes or any(fn() for fn in self._request_filters):
            return

        limits: List[Limit] = []
        dynamic_limits: List[Limit] = []

        if not in_middleware:
            limits = self.route_limits[name] if name in self.route_limits else []
            dynamic_limits = []
            if name in self._dynamic_route_limits:
                for lim in self._dynamic_route_limits[name]:
                    try:
                        dynamic_limits.extend(list(lim))
                    except ValueError as exc:
                        self.logger.error("failed to load rate limit for view function %s (%s)", name, exc)

        try:
            all_limits: List[Limit] = []
            if self._storage_dead and self._fallback_limiter:
                if not in_middleware or name not in self.__marked_for_limiting:
                    if self._should_check_backend() and self._storage.check():
                        self.logger.info("Rate limit storage recovered")
                        self._storage_dead = False
                        self.__check_backend_count = 0
                    else:
                        all_limits = list(itertools.chain(*self._in_memory_fallback))

            if not all_limits:
                route_limits: List[Limit] = limits + dynamic_limits
                all_limits = list(itertools.chain(*self._application_limits)) if in_middleware else []
                all_limits += route_limits
                combined_defaults = all(not limit.override_defaults for limit in route_limits)
                if not route_limits and not (in_middleware and name in self.__marked_for_limiting) or combined_defaults:
                    all_limits += list(itertools.chain(*self._default_limits))
            # actually check the limits, so far we've only computed the list of limits to check
            self.__evaluate_limits(request, endpoint, all_limits)

        except Exception as exc:  # pylint: disable=broad-except
            if isinstance(exc, RateLimitExceededError):
                raise
            # Other type of exception (data errors, connection errors)
            if self._in_memory_fallback_enabled and not self._storage_dead:
                self.logger.warning("Rate limit storage unreachable - falling back to" " in-memory storage")
                self._storage_dead = True
                self.check_request_limit(request, endpoint_func, in_middleware)
            else:
                if self._swallow_errors:
                    self.logger.exception("Failed to rate limit. Swallowing error")
                else:
                    raise

    def __limit_decorator(
        self,
        limit_value: StrOrCallableStr,
        key_func: Optional[Callable[..., str]] = None,
        shared: bool = False,
        scope: Optional[StrOrCallableStr] = None,
        per_method: bool = False,
        methods: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        exempt_when: Optional[Callable[..., bool]] = None,
        override_defaults: bool = True,
    ) -> AnyCallable:

        _scope = scope if shared else None

        def decorator(func: Callable[..., Response]) -> Callable[..., Response]:
            keyfunc = key_func or self._key_func
            name = f"{func.__module__}.{func.__name__}"
            dynamic_limit = None
            static_limits: List[Limit] = []
            if callable(limit_value):
                dynamic_limit = LimitGroup(
                    limit_value,
                    keyfunc,
                    _scope,
                    per_method,
                    methods,
                    error_message,
                    exempt_when,
                    override_defaults,
                )
            else:
                try:
                    static_limits = list(
                        LimitGroup(
                            limit_value,
                            keyfunc,
                            _scope,
                            per_method,
                            methods,
                            error_message,
                            exempt_when,
                            override_defaults,
                        )
                    )
                except ValueError as exc:
                    self.logger.error("Failed to configure throttling for %s (%s)", name, exc)
            self.__marked_for_limiting.setdefault(name, []).append(func)
            if dynamic_limit:
                self._dynamic_route_limits.setdefault(name, []).append(dynamic_limit)
            else:
                self.route_limits.setdefault(name, []).extend(static_limits)

            sig = inspect.signature(func)
            idx = -1
            for idx, parameter in enumerate(sig.parameters.values()):
                if parameter.name == "request" or parameter.name == "websocket":
                    break
            else:
                raise Exception(f'No "request" or "websocket" argument on function "{func}"')

            if asyncio.iscoroutinefunction(func):
                # Handle async request/response functions.
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Response:
                    # get the request object from the decorated endpoint function
                    request = kwargs.get("request", args[idx] if args else None)
                    if not isinstance(request, Request):
                        raise Exception("parameter `request` must be an instance of starlette.requests.Request")

                    if self.auto_check and not getattr(request.state, "rate_limiting_complete", False):
                        self.check_request_limit(request, func, False)
                        request.state.rate_limiting_complete = True
                    response: Response = await cast(Awaitable[Response], func(*args, **kwargs))
                    if not isinstance(response, Response):
                        # get the response object from the decorated endpoint function
                        _response = cast(Response, kwargs.get("response"))
                        self.inject_headers(_response, request.state.view_rate_limit)
                    else:
                        self.inject_headers(response, request.state.view_rate_limit)
                    return response

                return cast(Callable[..., Response], async_wrapper)

            # Handle sync request/response functions.
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Response:
                # get the request object from the decorated endpoint function
                request = kwargs.get("request", args[idx] if args else None)
                if not isinstance(request, Request):
                    raise Exception("parameter `request` must be an instance of starlette.requests.Request")

                if self.auto_check and not getattr(request.state, "rate_limiting_complete", False):
                    self.check_request_limit(request, func, False)
                    request.state.rate_limiting_complete = True

                response = func(*args, **kwargs)
                if not isinstance(response, Response):
                    # get the response object from the decorated endpoint function
                    self.inject_headers(kwargs.get("response"), request.state.view_rate_limit)
                else:
                    self.inject_headers(response, request.state.view_rate_limit)
                return response

            return sync_wrapper

        return decorator

    def limit(
        self,
        limit_value: Union[str, Callable[[str], str]],
        key_func: Optional[Callable[..., str]] = None,
        per_method: bool = False,
        methods: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        exempt_when: Optional[Callable[..., bool]] = None,
        override_defaults: bool = True,
    ) -> AnyCallable:
        """
        Decorator to be used for rate limiting individual routes.

        * **limit_value**: rate limit string or a callable that returns a string.
         :ref:`ratelimit-string` for more details.
        * **key_func**: function/lambda to extract the unique identifier for
         the rate limit. defaults to remote address of the request.
        * **per_method**: whether the limit is sub categorized into the http
         method of the request.
        * **methods**: if specified, only the methods in this list will be rate
         limited (default: None).
        * **error_message**: string (or callable that returns one) to override the
         error message used in the response.
        * **exempt_when**: function returning a boolean indicating whether to exempt
        the route from the limit
        * **override_defaults**: whether to override the default limits (default: True)
        """
        return self.__limit_decorator(
            limit_value,
            key_func,
            per_method=per_method,
            methods=methods,
            error_message=error_message,
            exempt_when=exempt_when,
            override_defaults=override_defaults,
        )

    def shared_limit(
        self,
        limit_value: Union[str, Callable[[str], str]],
        scope: StrOrCallableStr,
        key_func: Optional[Callable[..., str]] = None,
        error_message: Optional[str] = None,
        exempt_when: Optional[Callable[..., bool]] = None,
        override_defaults: bool = True,
    ) -> AnyCallable:
        """
        Decorator to be applied to multiple routes sharing the same rate limit.

        * **limit_value**: rate limit string or a callable that returns a string.
         :ref:`ratelimit-string` for more details.
        * **scope**: a string or callable that returns a string
         for defining the rate limiting scope.
        * **key_func**: function/lambda to extract the unique identifier for
         the rate limit. defaults to remote address of the request.
        * **per_method**: whether the limit is sub categorized into the http
         method of the request.
        * **methods**: if specified, only the methods in this list will be rate
         limited (default: None).
        * **error_message**: string (or callable that returns one) to override the
         error message used in the response.
        * **exempt_when**: function returning a boolean indicating whether to exempt
        the route from the limit
        * **override_defaults**: whether to override the default limits (default: True)
        """
        return self.__limit_decorator(
            limit_value,
            key_func,
            True,
            scope,
            error_message=error_message,
            exempt_when=exempt_when,
            override_defaults=override_defaults,
        )

    def exempt(self, obj: AnyCallable) -> AnyCallable:
        """
        Decorator to mark a view as exempt from rate limits.
        """
        name = "%s.%s" % (obj.__module__, obj.__name__)

        @wraps(obj)
        def _inner(*a: Any, **k: Any) -> Any:
            return obj(*a, **k)

        self.exempt_routes.add(name)
        return _inner
