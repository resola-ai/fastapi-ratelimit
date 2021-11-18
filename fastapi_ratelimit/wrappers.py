# pylint: disable=too-many-arguments,too-many-instance-attributes

from typing import Callable, Iterator, List, Optional, Union

from limits import RateLimitItem, parse_many

StrCallable = Callable[..., str]


class Limit:
    """
    simple wrapper to encapsulate limits and their context
    """

    def __init__(
        self,
        limit: RateLimitItem,
        key_func: StrCallable,
        scope: Optional[Union[str, StrCallable]] = None,
        per_method: bool = True,
        methods: Optional[List[str]] = None,
        error_message: Optional[Union[str, StrCallable]] = None,
        exempt_when: Optional[Callable[..., bool]] = None,
        override_defaults: bool = True,
    ) -> None:
        self.limit = limit
        self.key_func = key_func
        self.__scope = scope
        self.per_method = per_method
        self.methods = methods
        self.error_message = error_message
        self.exempt_when = exempt_when
        self.override_defaults = override_defaults

    @property
    def is_exempt(self) -> bool:
        """
        Check if the limit is exempt.
        Return True to exempt the route from the limit.
        """
        return self.exempt_when() if self.exempt_when is not None else False

    @property
    def scope(self) -> str:
        return ""


class LimitGroup:
    """
    represents a group of related limits either from a string or a callable that returns one
    """

    def __init__(
        self,
        limit_provider: Union[str, StrCallable],
        key_function: StrCallable,
        scope: Optional[Union[str, StrCallable]] = None,
        per_method: bool = False,
        methods: Optional[List[str]] = None,
        error_message: Optional[Union[str, StrCallable]] = None,
        exempt_when: Optional[Callable[..., bool]] = None,
        override_defaults: bool = False,
    ):
        self.__limit_provider = limit_provider
        self.__scope = scope
        self.key_function = key_function
        self.per_method = per_method
        self.methods = [m.lower() for m in methods] if methods else None
        self.error_message = error_message
        self.exempt_when = exempt_when
        self.override_defaults = override_defaults

    def __iter__(self) -> Iterator[Limit]:
        limit_items: List[RateLimitItem] = parse_many(
            self.__limit_provider() if callable(self.__limit_provider) else self.__limit_provider
        )
        for limit in limit_items:
            yield Limit(
                limit,
                self.key_function,
                self.__scope,
                self.per_method,
                self.methods,
                self.error_message,
                self.exempt_when,
                self.override_defaults,
            )
