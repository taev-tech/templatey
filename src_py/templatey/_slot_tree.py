from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence
from copy import copy
from dataclasses import KW_ONLY
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Self
from typing import cast
from typing import overload
from weakref import ref

from templatey._fields import ensure_normalized_fieldset
from templatey._fields import recursively_flatten_slot_annotations
from templatey._forwardrefs import ForwardRefLookupKey
from templatey._types import DYNAMIC_TEMPLATE_CLASS
from templatey._types import DynamicTemplateClass
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import create_templatey_id

type SlotPath = tuple[str, TemplateClass | DynamicTemplateClass]


# Note: the ordering here is to emphasize the fact that the slot
# name is on the ENCLOSING template, but the slot type is from the
# NESTED template
class PrerenderTreeRoute(tuple[str, TemplateClass, 'PrerenderTreeNode']):
    """An individual route on the prerender tree is defined by the attribute
    name for the slot, the slot type, and the subtree from the slot
    class.

    These are optimized for the non-union case. Traversing the prerender tree
    with union types will result in a bunch of unnecessary comparisons
    against slot names of different slot types.

    Note that prerender tree routes always have a concrete slot name and slot
    type, regardless of whether they're in a pending or concrete tree.
    The reason is simple: in a pending tree, all of the pending classes
    are dead-end nodes, and define their insertion points using just
    the string of the slot name, and nothing else.
    """
    @classmethod
    def new(
            cls,
            slot_name: str,
            slot_type: TemplateClass,
            subtree: PrerenderTreeNode,
            ) -> PrerenderTreeRoute:
        return cls((slot_name, slot_type, subtree))

    @property
    def subtree(self) -> PrerenderTreeNode:
        """This is slower than directly accessing the tuple values, but
        it makes for clearer code during tree building, where
        performance isn't quite so critical.
        """
        return self[2]

    @property
    def slot_path(self) -> SlotPath:
        return self[0:2]


@dataclass(slots=True)
class PrerenderTreeNode(list[PrerenderTreeRoute]):
    """The purpose of the prerender tree is to precalculate what sequences of
    getattr() calls we need to traverse to arrive at every instance of a
    particular slot type for a given template, including all nested
    templates.

    **These are optimized for rendering, not for template declaration.**
    Also note that these are optimized for slots that are not declared
    as type unions; type unions will result in a number of unnecessary
    comparisons against the routes of the other slot types in the union.

    The reason this is useful is for batching during rendering. This is
    important for function calls: it allows us to pre-execute all env
    func calls for a template before we start rendering it. In the
    future, it will also serve the same role for discovering the actual
    template types for dynamic slots, allowing us to load the needed
    template types in advance.

    An individual node on the prerender tree is a list of all possible
    attribute names (as ``PrerenderTreeRoute``s) that a particular search
    pass needs to check for a given instance. Note that **all** of the
    attributes must be searched -- hence using an iteration-optimized
    list instead of a mapping.
    """
    routes: InitVar[Iterable[PrerenderTreeRoute] | None] = None

    _: KW_ONLY
    # We use this to limit the number of entries we need in the transmogrifier
    # lookup during tree merging/copying
    is_recursive: bool = False

    # Set to true if the corresponding template has any environment functions.
    requires_precall: bool = False
    # Set to true if the corresponding template is a dynamic template class
    # that needs to get added to the preload.
    requires_preload: bool = False

    id_: int = field(default_factory=create_templatey_id)

    # We use this to make the logic cleaner when merging trees, but we want
    # the faster performance of the tuple when actually traversing the tree
    _index_by_slot_path: dict[SlotPath, int] = field(init=False, repr=False)

    def __post_init__(
            self,
            routes: Iterable[PrerenderTreeRoute] | None):
        # Explicit instead of super because... idunno, we're breaking things
        # somehow
        if routes is None:
            list.__init__(self)
        else:
            list.__init__(self, routes)

        self._index_by_slot_path = {
            route.slot_path: index for index, route in enumerate(self)}

    def empty_clone(
            self,
            *,
            fields_to_skip: set[str] | frozenset[str] = frozenset()
            ) -> Self:
        """This creates a clone of the node without any routes. Useful
        for merging and copying, where you need to do some manual
        transform of the content.

        Note that this is almost the same as dataclasses.replace, with
        the exception that we create shallow copies of attributes
        instead of preserving them.
        """
        kwargs = {}
        for dc_field in fields(self):
            if dc_field.init and dc_field.name not in fields_to_skip:
                # Note: the copy here is important for any mutable values,
                # notably the insertion_slot_names.
                kwargs[dc_field.name] = copy(getattr(self, dc_field.name))

        return type(self)(**kwargs)

    def merge_fields_only(
            self,
            other: PrerenderTreeNode,
            *,
            fields_to_skip: set[str] | frozenset[str] = frozenset()):
        """Updates the current node, merging in all non-init field
        values from other, using |=. Only merges values that exist on
        the current node, allowing for transformation between pending
        and concrete node types.

        Leaves the ID of the current node unchanged.
        """
        # Always skip ``id_`` -- always preserve the original ID!
        fields_to_skip = fields_to_skip | {'id_'}
        missing = object()

        for dc_field in fields(self):
            if dc_field.init and dc_field.name not in fields_to_skip:
                current_value = getattr(self, dc_field.name)
                other_value = getattr(other, dc_field.name, missing)

                if other_value is not missing:
                    setattr(
                        self,
                        dc_field.name,
                        operator.ior(current_value, other_value))

    @property
    def requires_transmogrification(self) -> bool:
        """This determines whether or not copies of the tree require
        some post-processing to make sure that the tree STRUCTURE is
        the same. It is used by both copying and merging trees to make
        sure that the transmogrification lookup is as sparse as
        possible.

        For the base class, we simply wrap ``is_recursive``, but for
        the pending tree derived class, we also check other stuff --
        hence the wrapping.
        """
        return self.is_recursive

    def append(self, route: PrerenderTreeRoute):
        """This has slightly different semantics to normal list
        appending if you're trying to append a duplicate slot path:
        ++  if it's identical to the existing path (ie, it targets the
            same node), we do nothing
        ++  if it's not identical, we error
        """
        slot_path = (route[0], route[1])
        slot_index = self._index_by_slot_path.get(slot_path)

        if slot_index is not None:
            if self[slot_index].subtree is route.subtree:
                return
            else:
                raise ValueError(
                    'Templatey internal error: attempt to append duplicate '
                    + 'slot name for same slot type, but a different subtree! '
                    + 'Please search for / report issue to github along with '
                    + 'a traceback.')

        list.append(self, route)
        self._index_by_slot_path[slot_path] = len(self) - 1

    def has_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass
            ) -> bool:
        return (slot_name, slot_type) in self._index_by_slot_path

    def get_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass
            ) -> PrerenderTreeRoute:
        return self[self._index_by_slot_path[(slot_name, slot_type)]]

    def rewrite_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass,
            new_route: PrerenderTreeRoute
            ) -> None:
        dest_index = self._index_by_slot_path[(slot_name, slot_type)]
        list.__setitem__(self, dest_index, new_route)

    def stringify(
            self,
            *,
            depth=0,
            _encountered_ids: frozenset[int] = frozenset()
            ) -> str:
        """Creates a pretty-print-style string representation of the
        node and all its nested nodes, recursively.
        """
        indentation = '    ' * depth

        to_join = []
        for dc_field in fields(self):
            if dc_field.init:
                field_name = dc_field.name
                to_join.append(
                    f'{indentation}{field_name}: {getattr(self, field_name)}')

        if self.id_ in _encountered_ids:
            to_join.append(
                f'{indentation}... Recursion detected; omitting subtrees.')

        else:
            for route in self:
                to_join.append(
                    f'{indentation}++  {route[0:2]}')
                to_join.append(route[2].stringify(
                    depth=depth + 1,
                    _encountered_ids=_encountered_ids | {self.id_}))

        return '\n'.join(to_join)

    def is_equivalent(
            self,
            other: PrerenderTreeNode,
            *,
            _previous_encounters: dict[int, int] | None = None
            ) -> bool:
        """This compares two prerender tree nodes recursively, ignoring IDs.
        If they are otherwise identical -- including the structure of
        any recursive loops -- returns True.

        Note that this is optimized for maintainability and not
        performance. Its primary intended use is in testing.
        """
        previous_encounters: dict[int, int]
        if _previous_encounters is None:
            previous_encounters = {}
        else:
            # Note that the copy here is important because otherwise we'd be
            # mutating state even in other tree branches which might not have
            # encountered us (since we reuse this in the recursive case)
            previous_encounters = {**_previous_encounters}

        # These two cases are trivial: first, if the types aren't the same,
        # they're not equivalent, period. Second, if we already encountered
        # that node's ID, that means we're in a recursive loop, and the two
        # IDs must match.
        if type(self) is not type(other):
            return False
        if other.id_ in previous_encounters:
            return previous_encounters[other.id_] == self.id_

        # First check all the fields, since this should in theory be quick.
        # Yes we could do a bool for this, but doing it as a dict makes it
        # easier to add in temporary print debugs if required
        fields_match: dict[str, bool] = {}
        for dc_field in fields(self):
            field_name = dc_field.name
            if dc_field.init and field_name != 'id_':
                fields_match[field_name] = (
                    getattr(self, field_name) == getattr(other, field_name))
        if not all(fields_match.values()):
            return False

        # Now make sure the routes are the same. This lets us simplify the
        # recursive comparison logic; we don't need any checks to make sure
        # that there weren't any leftover routes on either self or other.
        # It also lets us short-circuit without recursion if they don't match,
        # but that's not the primary motivation behind it.
        self_routes = set(self._index_by_slot_path)
        other_routes = set(other._index_by_slot_path)
        if self_routes != other_routes:
            return False

        # Now finally we're getting to recursive territory.
        previous_encounters[other.id_] = self.id_
        for slot_path in self._index_by_slot_path:
            self_index = self._index_by_slot_path[slot_path]
            self_subtree = self[self_index].subtree
            other_index = other._index_by_slot_path[slot_path]
            other_subtree = other[other_index].subtree

            # Recursion is glorious as long as you don't need to worry about
            # performance! Too bad we can't use it during rendering (because
            # it's way too slow)
            if not self_subtree.is_equivalent(
                other_subtree,
                _previous_encounters=previous_encounters
            ):
                return False

        return True

    def extend(self, routes: Iterable[PrerenderTreeRoute]):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __delitem__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __setitem__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def clear(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def copy(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def insert(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def pop(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def remove(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def reverse(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __imul__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __iadd__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')


@overload
def _copy_prerender_tree[T: PrerenderTreeNode](
        src_tree: T,
        ) -> T: ...
@overload
def _copy_prerender_tree[T: PrerenderTreeNode](
        src_tree: PrerenderTreeNode,
        *,
        with_node_type: type[T],
        ) -> T: ...
def _copy_prerender_tree[ST: PrerenderTreeNode, NT: PrerenderTreeNode](
        src_tree: ST,
        *,
        with_node_type: type[NT] | None = None
        ) -> ST | NT:
    """This creates a copy of an existing prerender tree. We use it when
    merging nested prerender trees into enclosers; otherwise, we end up with
    a huge mess of "it's not clear what object holds which prerender tree"
    that is very difficult to reason about. This is slightly more memory
    intensive, but... again, this is much, much easier to reason about.

    Take special note that this preserves reference cycles, which is a
    bit of a tricky thing.

    Note: if ``into_tree`` is provided, this copies inplace and returns
    the ``into_tree``. Otherwise, a new tree is created and returned.
    In both cases, we also return a lookup from
    ``{old_node.id_: copied_node}``.
    """
    copied_tree: ST | NT
    if with_node_type is None:
        node_type = type(src_tree)
    else:
        node_type = with_node_type

    copied_tree = node_type()
    copied_tree.merge_fields_only(src_tree)

    # This converts ``old_node.id_`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, ST | NT] = {src_tree.id_: copied_tree}
    copy_stack: \
        list[_PrerenderTreeTraversalFrame[ST | NT, ST]] = [
            _PrerenderTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=copied_tree,
                insertion_subtree=src_tree,
                first_encounters={})]

    while copy_stack:
        current_stack_frame = copy_stack[-1]
        if current_stack_frame.exhausted:
            copy_stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        next_subtree_id = next_subtree.id_
        already_copied_node = transmogrified_nodes.get(next_subtree_id)
        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if already_copied_node is None:
            new_subtree = node_type()
            new_subtree.merge_fields_only(next_subtree)

            if next_subtree.requires_transmogrification:
                transmogrified_nodes[next_subtree_id] = new_subtree

            current_stack_frame.existing_subtree.append(
                PrerenderTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    new_subtree,))
            copy_stack.append(_PrerenderTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=new_subtree,
                insertion_subtree=next_subtree,
                # Note that, though this isn't correct, it also
                # doesn't matter for purely COPYING a prerender tree.
                first_encounters={}))

        # We've hit a recursive subtree -- one that we've already copied --
        # which means we don't need to copy it again; instead, we just need to
        # transmogrify the reference so that the nested route refers back to
        # the original copied node.
        else:
            current_stack_frame.existing_subtree.append(
                PrerenderTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    already_copied_node,))

    return copied_tree


def merge_into_prerender_tree[T: PrerenderTreeNode](
        existing_tree_template_cls: TemplateClass,
        # Note: this is only used for determining ``is_terminus``.
        # Also note: None for pending slots. These can never be a terminus
        # anyways, so it doesn't matter.
        existing_tree_slot_type: TemplateClass | None,
        existing_tree: T,
        to_merge: T
        ) -> None:
    """This traverses the existing tree, merging in the slot_name and
    its subtrees into the correct locations in the existing prerender tree,
    recursively.

    This will cull any recursion loops to their minimum possible size.
    Note that this can result in different prerender tree recursion loop
    "phases", based on which of the two slot types is encountered first.

    Also note that this merges all public prerender tree node attributes,
    with three exceptions:
    ++  the ``id_`` of the existing tree is always preserved
    ++  ``is_recursive`` will always be calculated based on the
        recursion loop culling
    ++  ``is_terminus`` will always be calculated based on node types
        matching the passed ``existing_tree_slot_type``.
    """
    node_type = type(existing_tree)
    transmogrified_nodes: dict[int, T] = {to_merge.id_: existing_tree}
    # Note: we want our internal culling logic to handle ``is_recursive``.
    # Also note: the existing tree usually knows whether or not it's already
    # recursive, but there's an edge case where it doesn't:
    # 1.. have a recursive loop between two templates
    # 2.. when resolving the back-ref of the second template, it finds a
    #     forward ref to itself
    # 3.. it now immediately resolves the forward ref to itself
    # In this situation, the short-circuited self-referential forward ref
    # results in is_terminus being False, which we correct here.
    existing_tree.merge_fields_only(
        to_merge, fields_to_skip={'is_recursive', 'is_terminus'})
    existing_tree.is_terminus = (
        existing_tree_template_cls is existing_tree_slot_type)

    # Counterintuitive: since we're MERGING trees, the existing_subtree is
    # actually the DESTINATION, and the insertion_subtree the source!
    merge_stack: list[_PrerenderTreeTraversalFrame[T, T]] = [
        _PrerenderTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=existing_tree,
            insertion_subtree=to_merge,
            first_encounters={existing_tree_template_cls: existing_tree})]
    # Yes, in theory, this one specific operation of merging trees would be
    # faster if the trees were dicts instead of iterative structures. But
    # we're not optimizing for tree merging; we're optimizing for rendering!
    # And in that case, we're better off with a simple iterative structure.
    while merge_stack:
        current_stack_frame = merge_stack[-1]
        if current_stack_frame.exhausted:
            merge_stack.pop()
            continue

        existing_subtree = current_stack_frame.existing_subtree
        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow (also
        # because we want to use a continue statement in a second)
        current_stack_frame.next_subtree_index += 1

        next_subtree_id = next_subtree.id_
        already_merged_node = transmogrified_nodes.get(next_subtree_id)

        # For merging, we're going to handle the recursive subtree case first,
        # because it makes the rest of the logic cleaner
        if already_merged_node is not None:
            # Note that appending is idempotent as long as the
            # already_merged_node is the same object as any hypothetical
            # already_existing_already_merged_node that could be found there.
            # So this will only error if the slot path already exists, AND
            # it points to a different node.
            current_stack_frame.existing_subtree.append(
                PrerenderTreeRoute.new(
                    next_slot_name, next_slot_type, already_merged_node))

            already_merged_node.merge_fields_only(
                next_subtree,
                # Note: we want our internal culling logic to handle
                # ``is_recursive``. Otherwise, we might result in something
                # being reported as recursive, which actually isn't -- because
                # the ... phase, I guess you could say ... of the recursion
                # loop is different with a different root node.
                # Also note: we don't want to preserve the terminus value of
                # a tree we're merging in; we want the terminus to be purely
                # calculated within the merge function. This prevents bugs
                # when merging trees across slot types, which happens a lot
                # with pending prerender trees.
                fields_to_skip={'is_recursive', 'is_terminus'})
            # This might seem redundant, but there are some weird edge cases
            # when merging a fully-defined tree (for example, adding in
            # self-referential recursion loops to a pending tree) that require
            # it to be set.
            # (This is because we skip it when merging the fields on the
            # existing tree; this recovers the correct state).
            already_merged_node.is_recursive = True
            continue

        # This accomplishes two things: first, it culls an extra cycle from the
        # to_merge tree that would ultimately have the same effect. Secondly,
        # it ensures correct recursion when we're merging in the pending tree
        # from a class that used us as a forward reference, since the other
        # class won't yet be resolved.
        # Note that, by definition, this won't happen the first time we
        # encounter a node of any particular type, including the type of the
        # rootwards-most node for the whole tree. This is expected! We first
        # need to encounter the slot type before we can recurse to it. **We
        # will never recurse back to the root of the prerender tree!**
        if next_slot_type in current_stack_frame.first_encounters:
            # Note that we want to avoid modifying the already-encountered
            # tree. This is mostly because we copy trees willy nilly and want
            # to avoid them accidentally getting out of sync. However, it does
            # require us to be very careful to ensure that we've corrently
            # copied all needed values over prior to merging trees, if we've
            # just created the existing tree from scratch.
            next_existing_subtree = current_stack_frame.first_encounters[
                next_slot_type]
            next_existing_subtree.is_recursive = True

            next_existing_route = PrerenderTreeRoute.new(
                next_slot_name,
                next_slot_type,
                next_existing_subtree)
            if existing_subtree.has_route_for(next_slot_name, next_slot_type):
                existing_subtree.rewrite_route_for(
                    next_slot_name, next_slot_type, next_existing_route)
                # raise RecursionError(
                #     'Non-culled infinite recursion while merging templatey '
                #     + 'prerender trees!', next_slot_name, next_slot_type)

            else:
                existing_subtree.append(next_existing_route)

            # Note that we don't need to update transmogrification for two
            # reasons: first, because we added the root node at the very
            # beginning, and second, because we -- by definition -- cannot
            # have any deeper references to this part of the destination tree,
            # because  we're culling the depthwise-rest of the source tree
            continue

        # The existing subtree -- the one we're merging INTO -- has a route for
        # this slot name and type already, so we need to merge them together
        # instead of simply copy/transmogrify/cull
        elif existing_subtree.has_route_for(next_slot_name, next_slot_type):
            next_existing_route = existing_subtree.get_route_for(
                    next_slot_name, next_slot_type)
            __, __, next_existing_subtree = next_existing_route
            next_existing_subtree.merge_fields_only(
                next_subtree,
                # Note: we want our internal culling logic to handle
                # ``is_recursive``. Otherwise, we might result in something
                # being reported as recursive, which actually isn't -- because
                # the ... phase, I guess you could say ... of the recursion
                # loop is different with a different root node.
                # Also note: we don't want to preserve the terminus value of
                # a tree we're merging in; we want the terminus to be purely
                # calculated within the merge function. This prevents bugs
                # when merging trees across slot types, which happens a lot
                # with pending prerender trees.
                fields_to_skip={'is_recursive', 'is_terminus'})

        # The existing subtree doesn't have any existing routes for this, so
        # we don't need to worry about merging things together -- but we still
        # need to worry about transmogrification and culling.
        # Also note that there might be identically-named slots for different
        # slot types in the case of a union, but that will be handled on a
        # different iteration of the merge stack while loop.
        else:
            # Note that the next_subtree might have a different node type
            # than the existing tree!
            next_existing_subtree = node_type()
            next_existing_subtree.merge_fields_only(
                next_subtree,
                fields_to_skip={'is_recursive', 'is_terminus'})
            # Because we do lots and lots of merging, with offsets and
            # transforms etc, it's easiest to just reset this every time.
            # This helps prevent weird bugs with copying.
            next_existing_subtree.is_terminus = (
                # Note: what matters here is the slot type, not the template
                # class!
                next_slot_type is existing_tree_slot_type)
            next_existing_route = PrerenderTreeRoute.new(
                next_slot_name,
                next_slot_type,
                next_existing_subtree)
            existing_subtree.append(next_existing_route)

        if next_existing_subtree.requires_transmogrification:
            transmogrified_nodes[next_subtree_id] = next_existing_subtree

        if next_subtree:
            next_first_encounters = {**current_stack_frame.first_encounters}
            next_first_encounters.setdefault(
                next_slot_type, next_existing_subtree)
            merge_stack.append(_PrerenderTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=next_existing_subtree,
                insertion_subtree=next_subtree,
                first_encounters=next_first_encounters))


type _SlotTreeRoute = tuple[
    str, TemplateClass | DynamicTemplateClass, SlotTreeNode]
type _PendingNodeRegistry = dict[
    TemplateClass | DynamicTemplateClass, list[SlotTreeNode]]


@dataclass(slots=True, weakref_slot=True)
class SlotTreeNode(list[_SlotTreeRoute]):
    """Slot trees paint the full picture of all slots on all
    children of a particular root template class. They are not culled,
    and therefore not optimized for rendering. However, they can be
    generated as soon as template recursion totality is achieved -- ie,
    before actually loading the template bodies (in contrast to the
    prerender tree, which needs to know function invocations, and
    therefore requires template loading).

    These are used as an intermediate step to building a prerender tree.
    The challenge here is that we need to avoid infinite recursion when
    resolving mutually-recursive loops of slot dependencies. So we need
    a way that we can short-circuit tree population any time we
    encounter a slot with an already encountered slot class.

    These frames allow for recursion detection by reducing the problem
    to something (metaphorically) similar to this:
    >
    __embed__: 'code/python'
        def recursively_add_slot_classes(
                parent,
                all_slots: set[TemplateClass]):
            for slot_class in parent:
                if slot_class in all_slots:
                    continue
                else:
                    recursively_add_slot_classes(slot_class, all_slots)

    Although this requires a little bit of duplicated effort when we
    then need to build the trees for other slot classes along the
    recursion path, it massively simplifies the code to build them.

    These are not meant to be created directly; instead, use the two
    helpers, ``make_root`` and ``make_child``.
    """
    # Note: these are (deliberately) redundant with the nodepath that led us
    # here, and included purely for convenience
    slot_name: str
    slot_cls: TemplateClass | DynamicTemplateClass

    # Note that this does NOT include the current node
    _nodepath_parents: tuple[tuple[*SlotPath, ref[SlotTreeNode]], ...]

    # These are always self-referential, so we use a weakref to avoid literally
    # always needing complicated GC
    first_encounters: dict[TemplateClass, ref[SlotTreeNode]]

    # Both recursion fields are updated by children as they are created
    nonrecursive_descendants: set[TemplateClass | DynamicTemplateClass] = \
        field(default_factory=set)
    # These are deeper nodes, so no weakref is needed (the refs still flow in
    # the same direction)
    # Note that the ordering of the tuple here is deliberately different
    # than in other places, because it's also semantically different. In other
    # places, it's the slot path FOR the next-door node; in this case, it's
    # the slot path WITHIN the next-door node!
    recursion_loop_sources: list[tuple[SlotTreeNode, SlotPath]] = \
        field(default_factory=list)

    id_: int = field(default_factory=create_templatey_id)

    def __post_init__(self):
        for _, _, parent_pending_node in self.nodepath_parents:
            parent_pending_node.nonrecursive_descendants.add(self.slot_cls)

    def get_all_nonrecursive_inclusions(
            self
            ) -> Iterable[TemplateClass | DynamicTemplateClass]:
        """This provides a view into the slot classes that
        INCLUDES the current class.
        """
        for slot_cls in self.nonrecursive_descendants:
            yield slot_cls
        yield self.slot_cls

    def get_all_nodepath_inclusions(
            self,
            offset: SlotTreeNode | None = None
            ) -> Iterable[tuple[*SlotPath, SlotTreeNode]]:
        """Provides a view into all nodepath parents, plus the current
        node, in order. If offset is passed, start immediately after the
        passed node.
        """
        if offset is None:
            yield from self.nodepath_parents
        else:
            offset_encountered = False
            for slot_name, slot_cls, pending_node in self.nodepath_parents:
                if offset_encountered:
                    yield (slot_name, slot_cls, pending_node)

                if pending_node is offset:
                    offset_encountered = True

            if not offset_encountered:
                raise ValueError(
                    'Nodepath inclusion offset not found within parents!',
                    self, offset)

        yield (self.slot_name, self.slot_cls, self)

    @property
    def nodepath_parents(
            self
            ) -> tuple[tuple[*SlotPath, SlotTreeNode], ...]:
        """This dereferences the nodepath parents.
        """
        # Note: the cast() here is because pyright doesn't pick up on the
        # any() we do in just a second to fix it, and the explicit typecast
        # is because otherwise pyright assigns the type too broadly and it
        # no longer matches the return signature
        retval: tuple[tuple[*SlotPath, SlotTreeNode], ...] = tuple(
            (slot_name, slot_cls, cast(SlotTreeNode, noderef()))
            for slot_name, slot_cls, noderef in self._nodepath_parents)
        if any(segment[2] is None for segment in retval):
            raise RuntimeError(
                'Impossible branch: prematurely GCd nodepath parents!',
                self)

        return retval

    def check_recursion(
            self,
            template_cls: TemplateClass
            ) -> SlotTreeNode | None:
        """Checks for the passed ``template_cls`` in the history. If
        found, returns a it (after dereferencing). If not found, returns
        None. If the reference is no longer valid, raises.
        """
        if template_cls in self.first_encounters:
            node = self.first_encounters[template_cls]()
            if node is None:
                raise RuntimeError(
                    'Impossible branch: prematurely GCd slot tree '
                    + 'node!', self)

            return node

        return None

    def make_child(
            self,
            slot_name: str,
            slot_cls: TemplateClass | DynamicTemplateClass
            ) -> SlotTreeNode:
        new_child = SlotTreeNode(
            slot_name=slot_name,
            slot_cls=slot_cls,
            first_encounters={**self.first_encounters},
            _nodepath_parents=(
                *self._nodepath_parents, (slot_name, slot_cls, ref(self))),)

        if (
            slot_cls is not DYNAMIC_TEMPLATE_CLASS
            and slot_cls not in new_child.first_encounters
        ):
            new_child.first_encounters[slot_cls] = ref(new_child)

        return new_child

    @classmethod
    def make_root(
            cls,
            template_cls: TemplateClass,
            ) -> SlotTreeNode:
        new_root = cls(
            # Hacky, but... easier than needing to always check for Nones.
            slot_name='',
            slot_cls=template_cls,
            first_encounters={},
            _nodepath_parents=(),)
        new_root.first_encounters[template_cls] = ref(new_root)
        return new_root

    @staticmethod
    def update_postrecursion_descendants(
            pending_node: SlotTreeNode,
            postrecursion_descendants:
                dict[int, set[TemplateClass | DynamicTemplateClass]]
            ) -> set[TemplateClass | DynamicTemplateClass]:
        """Updates the postrecursion descendants for all children of
        the current ``pending_node``. Returns the total descendants for
        the current node.

        This relies upon us traversing the pending tree from shallowest
        to deepest, but it is independent of depth-first vs
        breadth-first, since we're updating specific descendants and the
        total descendants will never change for a particular node
        (again, as long as we traverse from shallowest to deepest).
        """
        total_descendants = (
            postrecursion_descendants[pending_node.id_]
            | pending_node.nonrecursive_descendants)
        for recursion_loop_src, _ in pending_node.recursion_loop_sources:
            for (
                _, _, descendant_node
            ) in recursion_loop_src.get_all_nodepath_inclusions(
                offset=pending_node
            ):
                postrecursion_descendants[descendant_node.id_].update(
                    total_descendants)

        return total_descendants

    def convert_to_prerender_tree(
            self,
            slot_cls: TemplateClass | DynamicTemplateClass
            ) -> PrerenderTreeNode:
        """Call this on the root of the pending tree to create a slot
        tree for the passed ``slot_cls``. This will correctly cull all
        non-relevant slot paths while keeping recursion loops intact.

        Overview of the algorithm:
        ++  iterate over the whole pending tree, from shallowest node to
            deepest node. this allows you to guarantee that you're finding
            the deepest recursions first.
        ++  keep track of the ``postrecursion_descendants`` separately.
            these get updated during the transformation.
            probably do this as an {id: set[]} construct, that way it's
            really easy to update.
        ++  every time you encounter a pending node with recursion loop
            sources:
            ++  for each recursion loop source
            ++  ``get_all_nodepath_inclusions`` (probably will want to add
                an offset parameter) FROM the recursion target UNTIL the
                recursion source (inclusive) -- ie, all intermediate nodes
                within the recursion loop
            ++  add the current pending node's postrecursion descendants
                (plus the slot class for the pending node) to that
                intermediate node's postrecursion descendants
        ++  every time you encounter a node, add the nonrecursive descendants
            to the postrecursion descendants to get the total descendants
            for that node
        ++  if the current slot class matches the target, include, regardless
            of how many children it has.
        ++  for each child:
            ++  if the target slot class isn't included in the child's
                total descendants, cull the child (including all its children)
        ++  if the current slot class does not match the target, AND all
            its children were culled, cull the current pending node in its
            entirety
        """
        # Note that this cannot be part of the stack frame, because it needs
        # to be specific to particular nodes -- not an entire tree depth.
        postrecursion_descendants: \
            dict[int, set[TemplateClass | DynamicTemplateClass]] = \
            defaultdict(set)
        slotnode_by_id: dict[int, PrerenderTreeNode] = {}
        visited_node_ids: set[int] = {self.id_}

        # Total descendants include every template class that could possibly be
        # encountered by any children of this node, including within any
        # recursion loops (even those that recurse higher up the tree than the
        # current frame).
        root_total_descendants = self.update_postrecursion_descendants(
            self, postrecursion_descendants)

        # Separating the frame from the stack creation gets pyright to stop
        # complaining that frame is possibly unbound
        frame = _PrerenderTreeBuilderFrame(
            pending_node=self,
            total_descendants=root_total_descendants)
        stack: list[_PrerenderTreeBuilderFrame] = [frame]

        while stack:
            frame = stack[-1]

            # When we exhaust all of the children in a frame, we need to first
            # check the node associated with the frame itself before advancing
            if frame.exhausted:
                # If we're at the root node, we don't need to check anything;
                # root nodes are always included.
                if len(stack) > 1:
                    parent_frame = stack[-2]

                    if (
                        # First case: the slot class of the current frame
                        # matches the slot class we're building a tree for.
                        # Always include!
                        frame.pending_node.slot_cls is slot_cls
                        # Second case: the slot class doesn't match, but it has
                        # non-culled children. Also include!
                        or frame.child_slot_routes
                    ):
                        new_node = PrerenderTreeNode(
                            frame.child_slot_routes,
                            # This is partly for convenience (by keeping the
                            # IDs the same it's easier to see what's going on)
                            # but is also critical for resolving recursion
                            # (see below)
                            id_=frame.pending_node.id_,)
                        slotnode_by_id[new_node.id_] = new_node

                        parent_frame.child_slot_routes.append(
                            # Note: this will work; we'll get to the leaf
                            # node, which will have no children. That will
                            # therefore be immediately exhausted, and the
                            # node created -- with an empty subtree.
                            # Then we'll proceed back up the tree from
                            # there.
                            PrerenderTreeRoute.new(
                                frame.pending_node.slot_name,
                                frame.pending_node.slot_cls,
                                new_node))

                        # Finally, we need to restore any recursion loops,
                        # which got lost because we needed to descend to the
                        # leafmost nodes for culling to work.
                        for (
                            recursion_src, recursive_slotpath
                        ) in frame.pending_node.recursion_loop_sources:
                            recursive_leaf = slotnode_by_id[recursion_src.id_]
                            recursive_leaf.append(
                                PrerenderTreeRoute.new(
                                    *recursive_slotpath, new_node))

                    # Implicit third case: the slot class doesn't match, AND
                    # all of its children (if it had any) were culled. In this
                    # case, the current frame should be culled as well.

                stack.pop()
                continue

            _, _, nested_pending_node = frame.advance()
            nested_total_descendants = self.update_postrecursion_descendants(
                nested_pending_node, postrecursion_descendants)

            # If the nested child's total descendants (ie, everything,
            # including any recursion loops, even to shallower levels) doesn't
            # include the target slot_cls, we can cull the entire branch (by
            # skipping this block).
            # Note that this doesn't automatically cull the node associated
            # with the frame itself -- we handle that after exhausting the
            # frame (see above).
            if (
                slot_cls in nested_total_descendants
                # This prevents infinite recursion, but it also relies upon
                # the exhaustion block to fixup the recursion in recursion
                # sources.
                and nested_pending_node.id_ not in visited_node_ids
            ):
                # Note: we can't rely upon the logic in the exhaustion block
                # to handle this, because exhaustion works its way back out
                # from the leaf nodes. Therefore we need something extra so
                # that the RECURSIVE leaf nodes don't continue on back to the
                # recursion target.
                visited_node_ids.add(nested_pending_node.id_)
                # NOTE: we're not adding this to the child slot routes in the
                # current frame yet! That gets handled when the child frame
                # is exhausted.
                stack.append(_PrerenderTreeBuilderFrame(
                    pending_node=nested_pending_node,
                    total_descendants=nested_total_descendants))

        # Okay, finally we need to fix recursion and termini. The easiest way
        # to do this is just to merge the tree into a fresh one.
        last_frame = frame
        root = PrerenderTreeNode()

        merge_into_prerender_tree(
            self.slot_cls,
            existing_tree_slot_type=slot_cls,
            existing_tree=root,
            to_merge=PrerenderTreeNode(routes=last_frame.child_slot_routes))

        return root

    def extend(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __delitem__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __setitem__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def clear(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def copy(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def insert(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def pop(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def remove(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def reverse(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __imul__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __iadd__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')


@dataclass(slots=True)
class _PrerenderTreeBuilderFrame:
    """``_PrerenderTreeBuilderFrame`` instances are responsible for
    building up a prerender tree from a slot tree.
    """
    pending_node: SlotTreeNode
    total_descendants: set[TemplateClass | DynamicTemplateClass]
    child_slot_routes: list[PrerenderTreeRoute] = field(default_factory=list)
    child_index: int = 0

    @property
    def exhausted(self) -> bool:
        return self.child_index >= len(self.pending_node)

    def advance(self) -> _SlotTreeRoute:
        next_route = self.pending_node[self.child_index]
        self.child_index += 1
        return next_route


@dataclass(slots=True)
class _SlotTreeBuilderFrame:
    """``_SlotTreeBuilderFrame`` instances are used to build up a
    slot tree, which is then used to construct the prerender tree.
    """
    slot_cls: TemplateClass
    pending_node: SlotTreeNode
    remaining_nested_slots: list[SlotPath]

    @classmethod
    def from_slot_cls(
            cls,
            slot_cls: TemplateClass,
            pending_node: SlotTreeNode,
            ) -> _SlotTreeBuilderFrame:
        """Constructs a new slot tree builder frame for the passed
        template class. **Note that the pending node is for the passed
        slot_cls!**
        """
        fieldset = cast(
            type[TemplateIntersectable], slot_cls
        )._templatey_signature.fieldset
        return cls(
            slot_cls=slot_cls,
            pending_node=pending_node,
            remaining_nested_slots=list(fieldset.slotpaths))

    @property
    def exhausted(self) -> bool:
        return not self.remaining_nested_slots


def build_slot_tree(
        template_cls: TemplateClass
        ) -> SlotTreeNode:
    """
    """
    # First we need to build up the pending slot tree, which contains all of
    # the nested template classes with no filtering or culling
    root_node = SlotTreeNode.make_root(template_cls)
    stack: list[_SlotTreeBuilderFrame] = [
        _SlotTreeBuilderFrame.from_slot_cls(template_cls, root_node)]

    while stack:
        frame = stack[-1]
        if frame.exhausted:
            stack.pop()
            continue

        nested_slot_name, nested_slot_cls = frame.remaining_nested_slots.pop()

        # We treat dynamic template classes like any other template class in
        # terms of the data structures we use to store them, but we do need
        # to short circuit any further checks on them.
        # Note that this differs from the non-recursive case in that we don't
        # add anything to the stack.
        if nested_slot_cls is DYNAMIC_TEMPLATE_CLASS:
            frame.pending_node.append((
                nested_slot_name,
                nested_slot_cls,
                frame.pending_node.make_child(
                    nested_slot_name, nested_slot_cls)))

        # In the simple recursion case -- a template defines a slot of its
        # own class -- we can immediately create a reference loop without
        # any pomp nor circumstance.
        elif nested_slot_cls is frame.slot_cls:
            trivially_recursive_slot_route = (
                nested_slot_name,
                nested_slot_cls,
                frame.pending_node)
            frame.pending_node.append(trivially_recursive_slot_route)
            # Note that we don't need to add this to the recursion sources,
            # because trivial recursion can't influence the culling of the tree

        # In the slightly more complicated recursion case -- a template defines
        # a slot of an already-encountered class -- we just need to retrieve
        # the previous node.
        elif (
            first_encounter
                := frame.pending_node.check_recursion(nested_slot_cls)
        ) is not None:
            mutually_recursive_slot_route = (
                nested_slot_name,
                nested_slot_cls,
                first_encounter)
            frame.pending_node.append(mutually_recursive_slot_route)
            first_encounter.recursion_loop_sources.append(
                (frame.pending_node, (nested_slot_name, nested_slot_cls)))

        # In the non-recursive case, we need to descend deeper into the
        # dependency graph.
        else:
            next_node = frame.pending_node.make_child(
                nested_slot_name, nested_slot_cls)
            next_frame = _SlotTreeBuilderFrame.from_slot_cls(
                nested_slot_cls, next_node)
            stack.append(next_frame)

    return root_node


def _convert_prerender_trees():
    # Now, we need to distill the pending slot tree into individual prerender trees,
    # one for each of the encountered template classes
    retval: dict[TemplateClass | DynamicTemplateClass, PrerenderTreeNode] = {}
    for slot_cls in root_node.get_all_nonrecursive_inclusions():
        retval[slot_cls] = root_node.convert_to_prerender_tree(slot_cls)

    return retval


def update_encloser_with_trees_from_slot(
        enclosing_cls: TemplateClass,
        enclosing_prerender_tree_lookup: dict[TemplateClass, ConcretePrerenderTreeNode],
        nested_slot_type: TemplateClass,
        nested_slot_name: str,
        ) -> None:
    """This function updates the enclosing class' concrete and pending
    prerender trees with the concrete and pending prerender trees from the
    nested slot.

    Note that it can be called during initial signature assembly, hence
    needing explicit arguments instead of relying upon the lookups
    already being defined on the class.
    """
    nested_slot_xable = cast(type[TemplateIntersectable], nested_slot_type)
    nested_lookup = nested_slot_xable._templatey_signature._prerender_tree_lookup

    # First and foremost, we can't forget to add a route for the
    # actual slot itself!
    _apply_concrete_insertions(
        enclosing_cls,
        enclosing_prerender_tree_lookup,
        nested_slot_type,
        nested_lookup,
        nested_slot_type,
        # Note that this is actually supposed to be the nested slot offset
        # in both cases; the function itself gets the appropriate tree from
        # the nested_lookup
        nested_slot_offset)

    # Now the CONCRETE nested slots.
    # IMPORTANT: all of the nested concrete slots need to already
    # be merged in before starting the nested pending refs! (More
    # info below).
    for doubly_nested_slot_type in nested_lookup:
        _apply_concrete_insertions(
            enclosing_cls,
            enclosing_prerender_tree_lookup,
            nested_slot_type,
            nested_lookup,
            doubly_nested_slot_type,
            # Note that this is actually supposed to be the nested slot offset
            # in both cases; the function itself gets the appropriate tree from
            # the nested_lookup
            nested_slot_offset)

    # And now, finally, the nested PENDING slots.
    # IMPORTANT: all of the nested concrete slots need to already
    # be merged in before starting the nested pending refs!
    # If we reversed the order, then we wouldn't be able to
    # guarantee that we'd already discovered every possible
    # terminus node for the nested slot type (due to potential
    # recursion loops). This could then cause us to miss out on
    # a lot of the combinatorics of the offset pending trees.
    # That being said, note that this is just a transform-and-merge
    # operation; we're not adding any additional insertion points
    # on top of what the nested slot defines. That happens later,
    # when we deal with the pending ref defs for the enclosing
    # template class (the one we're currently constructing a
    # signature for).

    # Note: we want to special-case a nested forward ref to the
    # enclosing class (ie, a recursion loop) for exactly the
    # same reason: to guarantee it's resolved before we start on
    # the other nested ones. Hence copy-and-mutate.
    unresolved_nested_forward_refs = {**nested_pending_refs}

    # Remember that we're potentially in the middle of constructing the
    # signature for a new template class. If the nested class
    # (from the slot) was depending on the class we're still
    # constructing, it hasn't yet been updated with the
    # resolved class. Therefore, instead of needing to come
    # back and fix up any recursive forward refs later, we can
    # simply do them right here, right now.
    # Also note that we'll NEVER have an existing pending tree
    # for this, because we're adding it directly and
    # immediately into the actual prerender tree.
    encloser_referencing_pending_container = (
        unresolved_nested_forward_refs.pop(
            enclosing_cls_ref_lookup_key, None))
    if encloser_referencing_pending_container is not None:
        # This is effectively a special-case of pending prerender tree extension
        # where we immediately apply the concrete insertions. But we still need
        # to merge the two pending trees (which handles the combinatorics --
        # remember that each insertion point for the nested class will also
        # get N insertion points from its references to the enclosing cls).
        _extend_pending_prerender_tree(
            enclosing_cls=enclosing_cls,
            enclosing_prerender_tree_lookup=enclosing_prerender_tree_lookup,
            enclosing_pending_ref_lookup=enclosing_pending_ref_lookup,
            nested_cls=nested_slot_type,
            nested_pending_container=encloser_referencing_pending_container,
            nested_pending_ref=enclosing_cls_ref_lookup_key,)
        offset_pending_container = enclosing_pending_ref_lookup.pop(
            enclosing_cls_ref_lookup_key)
        _apply_concrete_insertions(
            enclosing_cls,
            enclosing_prerender_tree_lookup,
            # These are counter-intuitive, but correct. We've divorced this
            # from the nested slot entirely; all the insertion points for it
            # now are pointed back at ourselves.
            enclosing_cls,
            enclosing_prerender_tree_lookup,
            enclosing_cls,
            offset_pending_container.pending_root_node)

    # Okay, after all of that meticulous work, we can now guarantee
    # that all of the concrete classes we know about up until this
    # point have been fully resolved and incorporated into the
    # prerender tree. That means we can FINALLY start merging the
    # pending items from the nested slot.
    for (
        nested_forward_ref_key, nested_pending_container
    ) in unresolved_nested_forward_refs.items():
        # Okay, so: technically, we should be checking the concrete prerender tree
        # for any reference loops to the enclosing class before doing this.
        # The problem is that this can change literally every time a pending
        # ref is resolved. Therefore, we do that THEN -- and quite literally.
        # Every time a pending slot is resolved, after resolution is complete,
        # we go through every tree and merge in the recursive one.
        _extend_pending_prerender_tree(
            enclosing_cls=enclosing_cls,
            enclosing_prerender_tree_lookup=enclosing_prerender_tree_lookup,
            enclosing_pending_ref_lookup=enclosing_pending_ref_lookup,
            nested_cls=nested_slot_type,
            nested_pending_container=nested_pending_container,
            nested_pending_ref=nested_forward_ref_key,)


def _apply_concrete_insertions(
        enclosing_cls: TemplateClass,
        enclosing_prerender_tree_lookup: dict[TemplateClass, ConcretePrerenderTreeNode],
        nested_slot_type: TemplateClass,
        # Note: may or may not be different from insertion_cls; see note in
        # docstring
        nested_prerender_tree_lookup: dict[TemplateClass, ConcretePrerenderTreeNode],
        insertion_slot_type: TemplateClass,
        insertion_tree: PendingPrerenderTreeNode
        ) -> None:
    """This takes an insertion tree (a pending prerender tree node), applies
    a resolved, concrete insertion class at each of the insertion
    points on the insertion tree, and then merges the result into the
    prerender tree lookup for the enclosing class.

    Note that this is meant to be used both with the nested class
    itself, as well as its (doubly-)nested concrete slots. In the former
    case, the prerender tree lookup will match the insertion class. In the
    latter case, the insertion class will instead be the (doubly-)nested
    concrete slot.
    """
    root_after_insertion = PrerenderTreeNode()
    tree_to_insert = nested_prerender_tree_lookup.get(
        insertion_slot_type,
        PrerenderTreeNode(is_terminus=True))

    # This converts ``old_node.id_`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, PrerenderTreeNode] = {
        insertion_tree.id_: root_after_insertion}
    stack: \
        list[_PrerenderTreeTraversalFrame[PrerenderTreeNode, PendingPrerenderTreeNode]] = [
        _PrerenderTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=root_after_insertion,
            insertion_subtree=insertion_tree,
            first_encounters={enclosing_cls: root_after_insertion})]

    while stack:
        current_stack_frame = stack[-1]
        src_subtree = current_stack_frame.insertion_subtree
        target_subtree = current_stack_frame.existing_subtree

        if current_stack_frame.exhausted:
            # This is counter-intuitive. Note that:
            # ++  the stack is for checking nested (deeper) routes only. it
            #     does not check for insertions on the current stack frame.
            # ++  the current stack frame might, though, actually have some
            #     insertions!
            # ++  we only want those insertions to be applied ONCE, and not
            #     once per subtree
            # ++  ordering of the insertion application doesn't matter
            # Therefore, we apply insertions to the current frame immediately
            # before discarding the frame for being exhausted.
            for nested_slot_type_slot_name in src_subtree.insertion_slot_names:
                target_subtree.append(
                    PrerenderTreeRoute.new(
                        nested_slot_type_slot_name,
                        nested_slot_type,
                        _copy_prerender_tree(tree_to_insert)))

            stack.pop()
            continue

        next_slot_route = src_subtree[current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        # Note that this will still get merged into the actual full prerender tree
        # for the enclosing template, which will cull any extra links in
        # recursive reference cycles, so we don't need to worry about that
        # here.
        next_subtree_id = next_subtree.id_
        transmogrified_dest_subtree = transmogrified_nodes.get(next_subtree_id)

        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if transmogrified_dest_subtree is None:
            dest_subtree = PrerenderTreeNode()
            dest_subtree.merge_fields_only(next_subtree)

            if next_subtree.requires_transmogrification:
                transmogrified_nodes[next_subtree_id] = dest_subtree

            # Note that, since we're building a new tree from scratch, we don't
            # need to worry about this already existing.
            target_subtree.append(
                PrerenderTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    dest_subtree))
            next_first_encounters = {**current_stack_frame.first_encounters}
            next_first_encounters.setdefault(next_slot_type, dest_subtree)
            stack.append(_PrerenderTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=dest_subtree,
                insertion_subtree=next_subtree,
                first_encounters=next_first_encounters))

        # We've hit a recursive subtree -- one that we've already copied --
        # which means we don't need to copy it again; instead, we just need to
        # transmogrify the reference so that the nested route refers back to
        # the original copied node.
        else:
            dest_subtree = transmogrified_dest_subtree
            # Note: is_recursive was already set!
            target_subtree.append(
                PrerenderTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    transmogrified_dest_subtree))

    enclosing_root = enclosing_prerender_tree_lookup.get(insertion_slot_type)
    if enclosing_root is None:
        enclosing_root = enclosing_prerender_tree_lookup[insertion_slot_type] = (
            PrerenderTreeNode())

    merge_into_prerender_tree(
        enclosing_cls,
        insertion_slot_type,
        enclosing_root,
        root_after_insertion)


@dataclass(slots=True)
class _PrerenderTreeTraversalFrame[ET: PrerenderTreeNode, IT: PrerenderTreeNode]:
    """
    """
    next_subtree_index: int
    existing_subtree: ET
    insertion_subtree: IT
    first_encounters: dict[TemplateClass | None, ET]

    _insertion_subtree_len: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._insertion_subtree_len = len(self.insertion_subtree)

    @property
    def exhausted(self) -> bool:
        """Returns True if the to_merge_subtree has been exhausted, and
        there are no more subtrees to merge.
        """
        return self.next_subtree_index >= self._insertion_subtree_len


def gather_dynamic_class_slots(
        enclosing_template_cls: TemplateClass,
        enclosing_dynamic_slot_names: set[str],
        enclosing_prerender_tree_lookup: dict[TemplateClass, ConcretePrerenderTreeNode]
        ) -> DynamicClassPrerenderTreeNode:
    """This is responsible for combining all of the dynamic slots on
    the enclosing template class with all of the dynamic slots on any
    nested templates (appropriately offset from the enclosing class,
    of course).

    This will actually create the prerender tree node; if you need to merge
    into an existing one (because you've resolved a forward ref), see
    ``merge_dynamic_slots``.

    Note that this can only be called **after** the enclosing prerender tree
    lookup has been fully populated.
    """
    tree_root = DynamicClassPrerenderTreeNode(
        dynamic_class_slot_names=enclosing_dynamic_slot_names)
    # This may or may not be present in the lookup. If it is, we need to
    # handle it like any other slot, because there might be recursion loops.
    # If not... well, then we'll just skip it in the for loop.
    # (This seeming redundancy was the most practical way to do this).
    if (
        enclosing_template_cls in enclosing_prerender_tree_lookup
        and enclosing_dynamic_slot_names
    ):
        merge_dynamic_class_slots(
            enclosing_template_cls,
            enclosing_prerender_tree_lookup,
            tree_root,
            enclosing_template_cls,
            tree_root)

    for nested_template_cls in enclosing_prerender_tree_lookup:
        # Note: more than just deduplication; it literally won't be available
        # at this point, because the signature won't be defined yet
        if nested_template_cls is not enclosing_template_cls:
            nested_template_signature = cast(
                type[TemplateIntersectable],
                nested_template_cls)._templatey_signature
            # Only merge if it actually HAS dynamic class slots! Otherwise,
            # we'd be wasting a bunch of tree traversals during render time!
            if nested_template_signature.dynamic_class_slot_names:
                merge_dynamic_class_slots(
                    enclosing_template_cls,
                    enclosing_prerender_tree_lookup,
                    tree_root,
                    nested_template_cls,
                    nested_template_signature._dynamic_class_prerender_tree)

    return tree_root


def merge_dynamic_class_slots(
        enclosing_template_cls: TemplateClass,
        enclosing_prerender_tree_lookup: dict[TemplateClass, ConcretePrerenderTreeNode],
        enclosing_dynamic_class_prerender_tree: DynamicClassPrerenderTreeNode,
        nested_template_cls: TemplateClass,
        # This needs to be explicit, because it's not always available on the
        # nested_template_cls (yet -- if enclosing_template_cls is
        # nested_template_cls, we might still be in initial signature creation)
        nested_dynamic_class_prerender_tree: DynamicClassPrerenderTreeNode
        ) -> None:
    """After the prerender tree of a newly-resolved nested template class has
    been fully merged, call this to also update the enclosing dynamic
    slots prerender tree for that particular nested_template_cls.

    Note that this modifies the tree in-place!

    This is very similar to _extend_pending_prerender_tree. The general idea
    is that we search for every terminus of the existing concrete tree
    for the nested template class, and then at that terminus, we set
    the terminus to False (since it isn't used by dynamic class slots)
    and insert a copy of the nested template class' dynamic class slot
    tree.
    """
    src_concrete_tree = enclosing_prerender_tree_lookup[nested_template_cls]
    dest_copied_tree = _copy_prerender_tree(
        src_concrete_tree,
        with_node_type=DynamicClassPrerenderTreeNode)
    # Note that it's not an issue that this doesn't track the stack, because
    # we aren't changing the structure of the existing tree at all.
    # (We can't reuse first_encounters because it's meant for a TemplateClass)
    encountered_node_ids: set[int] = set()

    stack: \
        list[
            _PrerenderTreeTraversalFrame[
                DynamicClassPrerenderTreeNode, ConcretePrerenderTreeNode]] = [
        _PrerenderTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=dest_copied_tree,
            insertion_subtree=src_concrete_tree,
            first_encounters={})]

    while stack:
        current_stack_frame = stack[-1]
        if current_stack_frame.exhausted:
            stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Note that we can't rely upon the indices being the same, because as
        # soon as we make an insertion, they'll drift out of sync
        target_subtree = current_stack_frame.existing_subtree.get_route_for(
            next_slot_name, next_slot_type).subtree

        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        # There's never anything to do here; it means we've already done all
        # of our transforms and insertions, because it's always recursive.
        # Doesn't matter where on the tree we are. We've already fixed this
        # node, period, end of story.
        # (Don't forget that the weird doubling-back we do only applies to the
        # next EXISTING subtree, but still advances the next (insertion)
        # subtree. So we're safe in that regard.)
        if next_subtree.id_ in encountered_node_ids:
            continue
        encountered_node_ids.add(next_subtree.id_)

        if next_subtree.is_terminus:
            # As per docstring, the idea here is that we're converting every
            # terminus into an insertion point for a different (pending) slot.
            # Therefore it's no longer a terminus, since the slot is different.
            target_subtree.is_terminus = False
            merge_into_prerender_tree(
                # Note: this is intentionally NOT the enclosing_cls! We're
                # working on an offset tree here, and we need to make sure that
                # we correctly report the class of the OFFSET root!
                next_slot_type,
                None,
                target_subtree,
                nested_dynamic_class_prerender_tree)

        # Whether or not we found a terminus, we need to check all of the
        # possibilities on the next subtree. (As a reminder, you can have a
        # terminus that still has nested nodes!)
        stack.append(_PrerenderTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=target_subtree,
            insertion_subtree=next_subtree,
            # Allow the final merging to handle any culling; we don't need to
            # do it here.
            first_encounters={}))

    merge_into_prerender_tree(
        enclosing_template_cls,
        None,
        enclosing_dynamic_class_prerender_tree,
        dest_copied_tree)


@dataclass(slots=True)
class _DynaClsExtractorFrame:
    """
    """
    active_instance: TemplateParamsInstance
    active_subtree: DynamicClassPrerenderTreeNode
    target_subtree_index: int
    target_instance_index: int
    target_instances_count: int = field(kw_only=True, default=0)
    target_instances: Sequence[TemplateParamsInstance] = field(
        kw_only=True, init=False)

    # See note in extract_dynamic_class_slot_types for why this is necessary
    # in addition to the encountered instance IDs
    direct_recursion_guard: set[int]
    target_slot_name_index: int = 0

    _active_subtree_len: int = field(init=False, repr=False, compare=False)
    _ordered_slot_names: list[str] = field(
        init=False, repr=False, compare=False)
    _ordered_slot_names_len: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._active_subtree_len = len(self.active_subtree)
        self._ordered_slot_names = ordered_slot_names = cast(
            TemplateIntersectable, self.active_instance
        )._templatey_signature._ordered_dynamic_class_slot_names
        self._ordered_slot_names_len = len(ordered_slot_names)

    @property
    def subtrees_exhausted(self) -> bool:
        return self.target_subtree_index >= self._active_subtree_len

    @property
    def dynacls_slots_exhausted(self) -> bool:
        return self.target_slot_name_index >= self._ordered_slot_names_len


# Yes, this is really complicated and really long. But unfortunately, the
# combinatorics are really bad, function calls in python are expensive, and
# this is on the critical hot path for rendering. So it's a calculated smell.
def extract_dynamic_class_slot_types(  # noqa: C901, PLR0912, PLR0915
        root_template_instance: TemplateParamsInstance,
        dynamic_class_prerender_tree: DynamicClassPrerenderTreeNode
        ) -> set[TemplateClass]:
    """Given a root template instance and its associated dynamic class
    prerender tree, walks the tree and extracts all template classes for
    slots defined as having a dynamic class.

    This does no culling based on whether or not the class was already
    loaded; it simply constructs a set of all encountered dynamic
    template classes.
    """
    encountered_instance_ids: set[int] = set()
    dynamic_template_classes: set[TemplateClass] = set()
    stack: list[_DynaClsExtractorFrame] = [
        _DynaClsExtractorFrame(
            active_instance=root_template_instance,
            active_subtree=dynamic_class_prerender_tree,
            target_subtree_index=0,
            target_instance_index=0,
            direct_recursion_guard={id(root_template_instance)})]

    while stack:
        frame = stack[-1]
        active_instance = frame.active_instance
        active_instance_id = id(active_instance)
        active_subtree = frame.active_subtree

        if active_instance_id in encountered_instance_ids:
            stack.pop()
            continue

        # Note: this is also the branch we use for "we only want to do this
        # N times per frame, not N times per subtree"
        if frame.subtrees_exhausted:
            if frame.dynacls_slots_exhausted:
                stack.pop()
                encountered_instance_ids.add(active_instance_id)

            else:
                slot_name = frame._ordered_slot_names[
                    frame.target_slot_name_index]
                target_instance_index = frame.target_instance_index

                # As with subtree checking, this is effectively a nested stack,
                # but we're maintaining state within the current frame to
                # decrease resource usage. See below for more explanation; it's
                # basically the same logic.
                if target_instance_index == 0:
                    target_instances = getattr(active_instance, slot_name)
                    target_instances_count = len(target_instances)

                    if target_instances_count <= 0:
                        frame.target_slot_name_index += 1
                        continue

                    else:
                        frame.target_instances_count = target_instances_count
                        frame.target_instances = target_instances

                else:
                    target_instances_count = frame.target_instances_count
                    if frame.target_instance_index >= target_instances_count:
                        frame.target_instances_count = 0
                        frame.target_instance_index = 0
                        frame.target_slot_name_index += 1
                        continue

                    target_instances = frame.target_instances

                instance_to_check = target_instances[target_instance_index]
                # Note: we need to special-case this separately from indirect
                # recursion, because we can't add the current instance ID to
                # encountered_instance_ids until we're done processing the
                # slots (otherwise we'd skip all of them!).
                # Additionally, this needs to be a stacked-set, and not a
                # simple comparison between instance_to_check and
                # active_instance_id, because otherwise, indirect recursion
                # (A -> B -> A -> ...) will infinitely descend without ever
                # marking anything as encountered. Doing it this way ensures
                # that only the outermost level is responsible for checking A.
                direct_recursion_guard = frame.direct_recursion_guard
                instance_to_check_id = id(instance_to_check)
                if instance_to_check_id not in direct_recursion_guard:
                    xable_to_check = cast(
                        TemplateIntersectable, instance_to_check)
                    dynamic_template_classes.add(type(instance_to_check))
                    # Here we also recurse to check the dynamic-class-slot
                    # instance to check if it, too, has dynamic classes. This
                    # is where the combinatorics really explode, but we're
                    # writing the code this way to keep the size of the stack
                    # minimal.
                    stack.append(_DynaClsExtractorFrame(
                        active_instance=instance_to_check,
                        active_subtree=
                            xable_to_check._templatey_signature
                            ._dynamic_class_prerender_tree,
                        target_subtree_index=0,
                        target_instance_index=0,
                        direct_recursion_guard=
                            direct_recursion_guard | {instance_to_check_id}))

                frame.target_instance_index += 1

            continue

        slot_route = active_subtree[frame.target_subtree_index]
        slot_name, slot_type, subtree = slot_route
        target_instance_index = frame.target_instance_index

        # This is, in a way, a nested stack, but we're maintaining
        # the stack state within the _DynaClsExtractorFrame.
        # At any rate, we use the zero-index iteration of the loop to
        # memoize some values on the stack frame.
        if target_instance_index == 0:
            target_instances = getattr(active_instance, slot_name)
            target_instances_count = len(target_instances)

            # Check in advance if there are no target instances at all,
            # and if so, skip the whole thing. This isn't just for
            # performance; the processing logic depends on it.
            if target_instances_count <= 0:
                # Note: this is critical! Otherwise we'll infinitely loop.
                frame.target_subtree_index += 1
                continue
            else:
                frame.target_instances_count = target_instances_count
                frame.target_instances = target_instances

        else:
            target_instances_count = frame.target_instances_count
            # We've exhausted the target instances; reset the state for
            # the next prerender tree route and then continue.
            if frame.target_instance_index >= target_instances_count:
                # Note: we're deliberately skipping the target instances
                # themselves, because it'll just get overwritten the next
                # time around, so we can save ourselves an operation.
                frame.target_instances_count = 0
                frame.target_instance_index = 0
                # Note: this is critical! Otherwise we'll infinitely loop.
                frame.target_subtree_index += 1
                continue

            # We still have some instances to target; normalize the state
            # so that we can operate on them.
            target_instances = frame.target_instances

        # Okay, status check: we have our stack frame state configured
        # correctly, and we have target instances to check.
        instance_to_check = target_instances[target_instance_index]
        # Note: exact match here; not subclassing! Subclassing breaks too
        # many things, so we don't support it.
        if type(instance_to_check) is slot_type:
            stack.append(_DynaClsExtractorFrame(
                active_instance=instance_to_check,
                active_subtree=subtree,
                target_subtree_index=0,
                target_instance_index=0,
                direct_recursion_guard=frame.direct_recursion_guard))

        frame.target_instance_index += 1

    return dynamic_template_classes
