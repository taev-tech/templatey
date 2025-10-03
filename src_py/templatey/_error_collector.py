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


def capture_traceback[E: Exception](
        exc: E,
        from_exc: Exception | None = None) -> E:
    """This is a little bit hacky, but it allows us to capture the
    traceback of the exception we want to "raise" but then collect into
    an ExceptionGroup at the end of the rendering cycle. It does pollute
    the traceback with one extra stack level, but the important thing
    is to capture the upstream context for the error, and that it will
    do just fine.

    There's almost certainly a better way of doing this, probably using
    traceback from the stdlib. But this is quicker to code, and that's
    my current priority. Gracefulness can come later!
    """
    try:
        if from_exc is None:
            raise exc
        else:
            raise exc from from_exc

    except type(exc) as exc_with_traceback:
        return exc_with_traceback
