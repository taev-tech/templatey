from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field

DEFAULT_MAX_ERRORS = 1000


@dataclass(slots=True, kw_only=True)
class ErrorCollector(list[Exception]):
    """The ``ErrorCollector`` is a list subclass with some minor
    improvements to help avoid getting into infinite loops due to
    faulty error handling.
    """

    # The lambda here is so that the DEFAULT_MAX_ERRORS can be changed at
    # runtime by library consumers
    max_errors: int = field(default_factory=lambda: DEFAULT_MAX_ERRORS)

    def append(self, obj: Exception):
        list.append(self, obj)

        if len(self) > self.max_errors:
            raise ExceptionGroup(
                'Max number of collected errors exceeded!',
                self)

    def extend(self, objs: Iterable[Exception]):
        list.extend(self, objs)

        if len(self) > self.max_errors:
            raise ExceptionGroup(
                'Max number of collected errors exceeded!',
                self)
