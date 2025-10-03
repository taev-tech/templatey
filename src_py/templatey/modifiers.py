from __future__ import annotations

import re
import typing
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from docnote import Note

if typing.TYPE_CHECKING:
    from templatey.parser import LiteralTemplateString
    from templatey.parser import TemplateInstanceContentRef
    from templatey.parser import TemplateInstanceDataRef
    from templatey.parser import TemplateInstanceVariableRef


@dataclass(slots=True, frozen=True)
class SegmentModifier:
    """Segment modifiers are a particularly powerful tool for reducing
    the verbosity of the actual template text. They allow you to modify
    literal string segments that match a particular pattern, replacing
    that pattern with a different segment sequence.

    For example, you could have a modifier that replaces every newline
    in the segment with a call to an ``add_indentation`` environment
    function:

    Note that if you want access to any part of the matched pattern,
    you must include it in a capture group:
    """
    pattern: re.Pattern
    modifier: Callable[
        [SegmentModifierMatch],
        Sequence[
            EnvFuncInvocationRef
            | TemplateInstanceContentRef
            | TemplateInstanceVariableRef
            | str]]

    def apply_and_flatten(
            self,
            literal_template_string: LiteralTemplateString
            ) -> tuple[
                bool,
                list[
                    str
                    | EnvFuncInvocationRef
                    | TemplateInstanceContentRef
                    | TemplateInstanceVariableRef]]:
        """This takes a literal template string segment and applies all
        modifiers. This gets it ready to be converted into the actual
        ParsedTemplateResource parts we need, but doesn't yet perform
        the final conversion.

        Returns a tuple of [had_matches, segments_after_modification]
        """
        after_modification: list[
            str
            | EnvFuncInvocationRef
            | TemplateInstanceContentRef
            | TemplateInstanceVariableRef] = []
        had_matches = False
        splits = self.pattern.split(literal_template_string)
        for split_segment_or_modification in SegmentModifierMatch.from_splits(
            self.pattern,
            splits
        ):
            if isinstance(split_segment_or_modification, str):
                # Here's where we filter out any empty strings. This also helps
                # us re-normalize the output from re.split, so that we recover
                # from the land of "let's use an empty string as a placeholder
                # for anything we did implicitly"
                if split_segment_or_modification:
                    after_modification.append(split_segment_or_modification)
            else:
                had_matches = True
                after_modification.extend(
                    self.modifier(split_segment_or_modification))

        return had_matches, after_modification


@dataclass(slots=True, kw_only=True)
class SegmentModifierMatch:
    """
    """
    captures: Annotated[
        list[str],
        Note(
            '''The captures in a segment modifier match correspond to
            the capture groups in the ``SegmentModifier`` pattern that
            resulted in the match. If the pattern had zero match groups,
            it will be an empty list. If it had one match group, it will
            be a single-item list. Etc.

            The ordering will be the same as the ordering of the groups
            in the original pattern. Named groups are not supported.
            ''')]

    @classmethod
    def from_splits(
            cls,
            src_pattern: re.Pattern,
            splits: list[str]
            ) -> Iterable[SegmentModifierMatch | str]:
        # Note: the +1 is because we need to account for the literal strings
        # in the pattern; otherwise the mod math doesn't work out! This also
        # has the added benefit of meaning we don't need to change behavior
        # between both cases.
        split_type_count = src_pattern.groups + 1

        current_captures = []
        for index, post_split_segment in enumerate(splits):
            # This is always literal text, even if it's an empty string.
            if not index % split_type_count:
                # Note that the zeroth and last index are always literal text,
                # and are never preceeded or followed by a capture. However,
                # we still need to yield the preceeding match when we reach
                # the end.
                if 0 < index:
                    yield cls(captures=current_captures)
                    current_captures = []

                # Note: we're saving the "cull empty strings" bit for later;
                # it makes the logic much cleaner to do one at a time
                yield post_split_segment

            else:
                current_captures.append(post_split_segment)


@dataclass(slots=True, init=False)
class EnvFuncInvocationRef:
    """Used to indicate that a segment modification needs to invoke an
    environment function. Instantiate these like partials:

    > Invocation example
        def my_env_func(foo: str, *, bar: int) -> list[str]:
            ...

        EnvFuncInvocationRef('my_env_func', 'foo', bar=3)
        EnvFuncInvocationRef(
            'my_env_func',
            # TemplateInstanceContentRef, TemplateInstanceDataRef, and
            # TemplateInstanceVariableRef are all supported, including nested
            # within containers
            TemplateInstanceVariableRef('foo'),
            bar=TemplateInstanceDataRef('bar'))

    Note that unfortunately, until/unless higher-kinded type support is
    added to python, these won't be able to type-check correctly.
    """
    name: str
    call_args: tuple[
        object
        | TemplateInstanceContentRef
        | TemplateInstanceDataRef
        | TemplateInstanceVariableRef, ...]
    call_kwargs: dict[
        str,
        object
        | TemplateInstanceContentRef
        | TemplateInstanceDataRef
        | TemplateInstanceVariableRef]

    def __init__(
            self,
            name: str,
            /,
            *call_args:
                object
                | TemplateInstanceContentRef
                | TemplateInstanceDataRef
                | TemplateInstanceVariableRef,
            **call_kwargs:
                object
                | TemplateInstanceContentRef
                | TemplateInstanceDataRef
                | TemplateInstanceVariableRef):
        self.name = name
        self.call_args = call_args
        self.call_kwargs = call_kwargs
