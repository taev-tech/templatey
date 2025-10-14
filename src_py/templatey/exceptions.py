from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from templatey._slot_tree import SlotTreeNode


class TemplateyException(Exception):
    """Base class for all templatey exceptions."""


class BlockedContentValue(TemplateyException):
    """Raised by content verifier functions if the content contains
    a blocked value -- for example, if an HTML verifier with an
    allowlist for certain tags detects a blocked tag.
    """


class InvalidTemplateInterface(TemplateyException):
    """Raised when something was wrong with the template interface
    definition.
    """


class InvalidTemplate(TemplateyException):
    """The most general form of "there's a problem with this template."
    """


class InvalidTemplateInterpolation(InvalidTemplate):
    """Raised when there was a specific problem with an interpolation
    within the template. That might be a typo, or it might be that you
    are trying to directly reference a value instead of putting it
    within a var/content/slot/etc namespace, or something else entirely.
    """


class DuplicateSlotName(InvalidTemplate):
    """Raised when a particular template has multiple slots with the
    same name.
    """


class MismatchedTemplateEnvironment(InvalidTemplate):
    """Raised when loading templates, if the template environment
    doesn't contain all of the template functions referenced by the
    template text.
    """


class MismatchedTemplateSignature(InvalidTemplate):
    """Raised when loading templates, if the template interface doesn't
    contain all of the contextuals (variables, slots, etc) referenced by
    the template text.

    May also be raised during rendering, if the template text attempts
    to reference a var as a slot, slot as content, etc.
    """


class OvercomplicatedSlotTree(InvalidTemplate):
    """Raised when loading templates if the template's slot tree is
    too complicated, as determined by the slot tree complexity limiter
    set on the render environment.

    This is generally an indication of too much mutual recursion in the
    slot tree, which can quickly result in explosive combinatorics.
    Rather than allowing the slot tree to grow to multiple megabytes or
    gigabytes in size, taking multiple minutes to generate, we instead
    raise this error.

    If you find yourself encountering this error in situations where
    the slot tree complexity is unavoidable, you can either raise the
    limits on the render environment, or -- far preferrably -- simply
    replace the most combinatorically expensive slots with
    ``DynamicClassSlot`` annotations instead of explicit slots. This
    will result in a very slight penalty in render speed, but
    dramatically reduce template load times.
    """
    partial_slot_tree: SlotTreeNode

    def __init__(self, *args, partial_slot_tree: SlotTreeNode, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_slot_tree = partial_slot_tree


class IncompleteTemplateParams(TypeError):
    """Raised when an ellipsis is still present in either slots or
    variables at render time.
    """


class TemplateFunctionFailure(Exception):
    """Raised when a requested template function raised an exception.
    Should always be raised ^^from^^ the raised exception, so that its
    traceback is preserved.
    """


class MismatchedRenderColor(Exception):
    """Raised when trying to access an async resource, especially an
    async environment function, from within a synchronous render call.
    """
