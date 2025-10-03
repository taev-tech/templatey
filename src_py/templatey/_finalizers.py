"""These are responsible for finalizing the template definition during
loading. We defer until loading so that forward refs on the templates
are resolved. Trust me, it's way, WAY easier this way (we used to try
to do it the other way and it was a massive headache.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import cast

from templatey._fields import NormalizedFieldset
from templatey._signature import TemplateSignature
from templatey._slot_tree import build_slot_tree
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey.parser import ParsedTemplateResource


@dataclass(slots=True)
class _TotalityRecursionGuard:
    root_template_cls: TemplateClass
    root_total_inclusions: set[TemplateClass]


def ensure_recursive_totality(
        signature: TemplateSignature,
        template_cls: TemplateClass,
        *,
        _recursion_guard: _TotalityRecursionGuard | None = None
        ) -> None:
    """This function constructs and populates the
    ``signature.fieldset`` and ``signature.total_inclusions``
    attributes if (and only if) either one of them is missing, and
    then recursively does the same for all non-dynamic nested slot
    classes.

    This is helpful for several reasons:
    ++  it makes sure that we have a full fieldset and descendant
        classes for the entire descendancy tree, so that we can
        construct a slot tree (and thereafter a prerender tree) for
        the class
    ++  if the attributes already exist, it does nothing, so we
        don't waste work on nested template classes
    ++  it allows parsed template resources to fall out of cache
        without discarding the work we did processing the signature
        itself

    This is meant to be called during loading (instead of, for example,
    during ``@template`` decoration time) because it maximizes the
    chances that all of the type hints are available (ie, no longer
    forward refs).

    Note that our approach here, though not computationally ideal, is
    much simpler than a more sophisticated/efficient approach. Instead
    of trying to construct all of the totalities at the same time, we
    simply construct them for the root class, and then proceed on to
    each one of its inclusions iteratively. Again, there's a lot of
    duplicate work here, but it makes it **much** easier to resolve
    recursion loops.

    This returns the total inclusions for the passed class, whether or
    not it was the root template class.
    """
    if not hasattr(signature, 'fieldset'):
        signature.fieldset = NormalizedFieldset.from_template_cls(
            template_cls)

    if not hasattr(signature, 'total_inclusions'):
        total_inclusions: set[TemplateClass]
        if _recursion_guard is None:
            total_inclusions = {template_cls}
            recursion_guard = _TotalityRecursionGuard(
                root_template_cls=template_cls,
                root_total_inclusions=total_inclusions)
        else:
            total_inclusions = _recursion_guard.root_total_inclusions
            # Note: this MUST be before the recursive call, or it won't protect
            # against recursion!
            total_inclusions.add(template_cls)
            recursion_guard = _recursion_guard

        # Note that in addition to saving us work (both by deduping
        # and by reusing the results from creating the fieldset),
        # this also flattens unions, aliases, etc.
        direct_inclusions: set[TemplateClass] = set()
        for _, nested_slot_cls in signature.fieldset.slotpaths:
            # Note that these do NOT include dynamic slot classes!
            direct_inclusions.add(nested_slot_cls)

        # Don't forget that templates can include themselves as a slot!
        # (Also, the total_classes already includes it).
        # This is another protection against infinite recursion.
        direct_inclusions.discard(template_cls)
        for nested_slot_cls in direct_inclusions:
            if nested_slot_cls not in total_inclusions:
                nested_signature = cast(
                    type[TemplateIntersectable], nested_slot_cls
                )._templatey_signature
                ensure_recursive_totality(
                    nested_signature,
                    nested_slot_cls,
                    _recursion_guard=recursion_guard)

        # We only want to set the value if we're being called on the root;
        # nothing else is definitely correct!
        if _recursion_guard is None:
            signature.total_inclusions = frozenset(total_inclusions)

            # But we do actually need to ensure recursive totality, so then
            # we need to follow up with each and every one of the inclusions.
            for nested_inclusion in total_inclusions:
                nested_signature = cast(
                    type[TemplateIntersectable], nested_inclusion
                )._templatey_signature
                # Note the difference: no _root_template_cls!
                ensure_recursive_totality(nested_signature, nested_inclusion)


@dataclass(slots=True)
class _TotalityFrame:
    """When calculating totality, we need to resolve recursion loops
    (again). ``_TotalityFrame`` objects maintain the state we need to
    use a stack-based approach for that instead of the naive recursion
    option, which ends up in infinite recursion when you get to
    nontrivial recursion loops.

    TODO: this needs a better description, that's less sloppy and
    confusing. Point is, we use this to calculate recursive totality.
    """
    slot_cls: TemplateClass
    # Note: this gets mutated during processing
    remaining_direct_inclusions: set[TemplateClass]
    signature: TemplateSignature
    total_classes: set[TemplateClass] = field(init=False)
    recursion_sources: list[tuple[_TotalityFrame, ...]] = field(
        default_factory=list)

    def __post_init__(self):
        self.total_classes = {self.slot_cls}

    @classmethod
    def from_slot_cls(cls, slot_cls: TemplateClass) -> _TotalityFrame:
        signature = cast(
            type[TemplateIntersectable], slot_cls)._templatey_signature
        # Note that in addition to saving us work (both by deduping
        # and by reusing the results from creating the fieldset),
        # this also flattens unions, aliases, etc.
        direct_inclusions: set[TemplateClass] = set()
        for _, nested_slot_cls in signature.fieldset.slotpaths:
            # Note that these do NOT include dynamic slot classes!
            direct_inclusions.add(nested_slot_cls)

        # Don't forget that templates can include themselves as a slot!
        # (Also, the total_classes already includes it).
        # This prevents infinite recursion.
        direct_inclusions.discard(slot_cls)

        return cls(
            slot_cls=slot_cls,
            remaining_direct_inclusions=direct_inclusions,
            signature=signature)

    @property
    def exhausted(self) -> bool:
        return not bool(self.remaining_direct_inclusions)

    def extract_recursion_loop(
            self,
            stack: list[_TotalityFrame],
            recursion_target: _TotalityFrame
            ) -> tuple[_TotalityFrame, ...]:
        """Given the current stack, and an up-stack recursion target,
        extracts out just the frames that are part of the recursion
        loop.
        """
        target_frames: list[_TotalityFrame] = []
        target_encountered = False
        for frame in stack:
            if frame is recursion_target:
                target_encountered = True
            elif target_encountered:
                target_frames.append(frame)

        if not target_encountered:
            raise ValueError(
                'Recursion target not found in stack!',
                stack, recursion_target)

        return tuple(target_frames)


def ensure_slot_tree(
        signature: TemplateSignature,
        template_cls: TemplateClass,):
    """After ensuring recursive totality, call this to make sure
    that the slot tree is defined on the signature.

    This is meant to be called during loading (instead of, for example,
    during ``@template`` decoration time) because it maximizes the
    chances that all of the type hints are available (ie, no longer
    forward refs).
    """
    if not hasattr(signature, 'slot_tree'):
        signature.slot_tree = build_slot_tree(template_cls)


def ensure_prerender_tree(
        signature: TemplateSignature,
        preload: dict[TemplateClass, ParsedTemplateResource]):
    """After fully loading the underlying template and all of its
    inclusions, call this to make sure that the prerender tree is
    defined on the signature.

    This is meant to be called during loading (instead of, for example,
    during ``@template`` decoration time) because it maximizes the
    chances that all of the type hints are available (ie, no longer
    forward refs).
    """
    if not hasattr(signature, 'prerender_tree'):
        signature.prerender_tree = signature.slot_tree.distill_prerender_tree(
            preload)
