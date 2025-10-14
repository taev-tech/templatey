from __future__ import annotations

import itertools
import operator
from collections.abc import Iterable
from collections.abc import Sequence
from copy import copy
from dataclasses import KW_ONLY
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any
from typing import Self
from typing import cast

from templatey._error_collector import ErrorCollector
from templatey._provenance import Provenance
from templatey._provenance import ProvenanceNode
from templatey._renderer import FuncExecutionRequest
from templatey._renderer import TemplateInjection
from templatey._renderer import get_precall_cache_key
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import create_templatey_id
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import ParsedTemplateResource

type SlotPath = tuple[str, TemplateClass]


class _ProxyDescriptor:
    """This is a bit of a hack. The goal here is to allow dataclasses
    that also inherit from something else (eg list) to include their
    contents in their default-generated repr. The general strategy is
    to use a non-init field as a proxy to the super() repr of the
    object.

    Instead of using a [[descriptor-typed
    field](https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields)]
    -- which cannot be assigned init=False -- we simply set the field
    as normal, allow the dataclass to be processed, **and then**
    overwrite the field with the descriptor.
    """

    def __get__(self, obj: Any | None, objtype: type | None = None):
        if obj is None:
            return '...'
        elif objtype is None:
            return '...'
        else:
            # Create a disposable shallow copy of the object using the
            # superclass. This makes everything look as expected.
            return objtype.__mro__[1](obj)


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
        """This seems weird and redundant, but it lets us support kwargs
        for creation, as well as the usual tuple signature.
        """
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

    abstract_calls: tuple[InterpolatedFunctionCall, ...]
    dynamic_slot_names: tuple[str, ...]

    id_: int = field(default_factory=create_templatey_id)

    # We use this to make the logic cleaner when navigating trees by paths, but
    # we want the faster performance of the tuple when rendering
    _index_by_slot_path: dict[SlotPath, int] = field(init=False, repr=False)

    # Proxy object for repr; see docnote for _ProxyDescriptor
    _children: list[PrerenderTreeRoute] = field(init=False)

    def __post_init__(
            self,
            routes: Iterable[PrerenderTreeRoute] | None):
        # Explicit instead of super because dataclass on slots breaks super()
        if routes is None:
            list.__init__(self)
        else:
            list.__init__(self, routes)

        self._index_by_slot_path = {
            route.slot_path: index for index, route in enumerate(self)}

    def extract(
            self,
            from_instance: TemplateParamsInstance,
            from_injection: Provenance | None,
            into_injection_backlog: list[TemplateInjection],
            into_precall_backlog: list[FuncExecutionRequest],
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            error_collector: ErrorCollector,
            ) -> None:
        """Extracts all dynamic template injections and function
        executions from the root instance, using ourselves as the tree.
        """
        stack: list[_ExtractionFrame] = [
            _ExtractionFrame(
                active_instance=from_instance,
                active_subtree=self,
                target_subtree_index=0,
                target_instance_index=0,
                wip_provenance=Provenance(
                    (
                        ProvenanceNode(
                            encloser_slot_key='',
                            encloser_slot_index=-1,
                            instance_id=id(from_instance),
                            instance=from_instance),),
                    from_injection=from_injection))]

        while stack:
            frame = stack[-1]
            frame_subtree = frame.active_subtree

            # If the frame is exhausted, we still need to process any dyanamic
            # slots or env func calls on the current instance.
            if frame.exhausted:
                stack.pop()
                frame_instance = frame.active_instance
                frame_provenance = frame.wip_provenance

                for abstract_call in frame_subtree.abstract_calls:
                    args, kwargs = frame_provenance.bind_call_signature(
                        abstract_call,
                        template_preload,
                        error_collector)
                    into_precall_backlog.append(
                        FuncExecutionRequest(
                            abstract_call.name,
                            args=args,
                            kwargs=kwargs,
                            result_key=get_precall_cache_key(
                                frame_provenance, abstract_call),
                            provenance=frame_provenance))

                for dynamic_slot_name in frame_subtree.dynamic_slot_names:
                    into_injection_backlog.extend(
                        (
                            Provenance((
                                ProvenanceNode(
                                    encloser_slot_key='',
                                    encloser_slot_index=-1,
                                    instance_id=id(dynamic_instance),
                                    instance=dynamic_instance),),
                                from_injection=frame_provenance),
                            dynamic_instance)

                        for dynamic_instance
                        in getattr(frame_instance, dynamic_slot_name))

                continue

            slot_route = frame_subtree[frame.target_subtree_index]
            slot_name, slot_type, slot_subtree = slot_route
            target_instance_index = frame.target_instance_index

            # We use the zero-index iteration of the loop
            # to memoize some values on the stack frame.
            # This is, in a way, a nested stack, but we're maintaining
            # the stack state within the _ExtractionFrame.
            if target_instance_index == 0:
                target_instances = getattr(frame.active_instance, slot_name)
                target_instances_count = len(target_instances)

                # Check in advance if there are no target instances at all,
                # and if so, skip the whole thing. This isn't just for
                # performance; the processing logic depends on it.
                if target_instances_count > 0:
                    frame.target_instances_count = target_instances_count
                    frame.target_instances = target_instances
                else:
                    # Note: this is critical! Otherwise we'll infinitely loop.
                    frame.target_subtree_index += 1
                    continue

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

            instance_to_check = target_instances[target_instance_index]
            frame.target_instance_index += 1
            # Okay, status check: we have our stack frame state configured
            # correctly for the next iteration, and we have target instances
            # to check.
            # We still need to verify that the instances we find actually match
            # the slot type (in case of unions), but if we find a match, we'll
            # need to add a new frame to the stack.
            # Note: exact match here; not subclassing! Subclassing breaks too
            # many things, so we don't support it.
            if type(instance_to_check) is slot_type:
                stack.append(_ExtractionFrame(
                    active_instance=instance_to_check,
                    active_subtree=slot_subtree,
                    target_subtree_index=0,
                    target_instance_index=0,
                    wip_provenance=Provenance(
                        (
                            *frame.wip_provenance.slotpath,
                            ProvenanceNode(
                                encloser_slot_key=slot_name,
                                encloser_slot_index=target_instance_index,
                                instance_id=id(instance_to_check),
                                instance=instance_to_check)),
                        from_injection=from_injection)))

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

    def __truediv__(self, other: SlotPath) -> PrerenderTreeNode:
        """A utility method for tree traversal. Only intended for use
        in debugging and testing; not optimized for production use.
        """
        return self.get_route_for(*other)[2]

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
PrerenderTreeNode._children = _ProxyDescriptor()  # type: ignore


@dataclass(slots=True)
class _ExtractionFrame:
    """
    """
    active_instance: TemplateParamsInstance
    active_subtree: PrerenderTreeNode
    target_subtree_index: int
    target_instance_index: int
    target_instances_count: int = field(kw_only=True, default=0)
    target_instances: Sequence[TemplateParamsInstance] = field(
        kw_only=True, init=False)
    wip_provenance: Provenance

    _active_subtree_len: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._active_subtree_len = len(self.active_subtree)

    @property
    def exhausted(self) -> bool:
        return self.target_subtree_index >= self._active_subtree_len


type _SlotTreeRoute = tuple[str, TemplateClass, SlotTreeNode]


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
    helpers, ``make_root`` and ``add_child``.
    """
    # Note: these are (deliberately) redundant with the nodepath that led us
    # here, and included purely for convenience
    slot_name: str
    slot_cls: TemplateClass

    dynamic_slot_names: set[str]
    id_: int = field(default_factory=create_templatey_id)

    # Proxy object for repr; see docnote for _ProxyDescriptor
    _children: list[_SlotTreeRoute] = field(init=False)

    def __truediv__(self, other: SlotPath) -> SlotTreeNode:
        """A utility method for tree traversal. Only intended for use
        in debugging and testing; not optimized for production use.
        """
        for slot_name, slot_cls, slot_node in self:
            if (slot_name, slot_cls) == other:
                return slot_node

        raise LookupError('No such slot path!', other)

    def add_child(
            self,
            slot_name: str,
            slot_cls: TemplateClass
            ) -> SlotTreeNode:
        """Creates a child node and appends it to self, then returns it.
        """
        new_child = SlotTreeNode(
            slot_name=slot_name,
            slot_cls=slot_cls,
            dynamic_slot_names=set(),)
        self.append((slot_name, slot_cls, new_child))
        return new_child

    @classmethod
    def make_root(
            cls,
            template_cls: TemplateClass,
            ) -> SlotTreeNode:
        return cls(
            # Hacky, but... easier than needing to always check for Nones.
            slot_name='',
            slot_cls=template_cls,
            dynamic_slot_names=set(),)

    def distill_prerender_tree(
            self,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            ) -> PrerenderTreeNode | None:
        """Call this on the root of the slot tree to create a prerender
        tree based on the template preload. This will correctly cull all
        non-relevant slot paths (ie, any slot paths with neither dynamic
        slot classes nor env func calls) while keeping recursion loops
        intact.

        This is relatively simple. We first calculate a set of all
        prerender-relevant template classes by checking the preload
        for all function invocations and the template fieldsets for
        dynamic classes. Then, we simply rely upon the already-existing
        recursive totality for each node, and cull any nodes whose
        total inclusions don't overlap with the prerender-relevant
        template classes.

        The only additional logic is just to preserve recursion loops.
        """
        inclusions_with_precall: set[TemplateClass] = {
            template_cls
            for template_cls, parsed_template in template_preload.items()
            if parsed_template.function_calls}
        inclusions_with_dynacls: set[TemplateClass] = {
            template_cls
            for template_cls in template_preload
            if cast(
                type[TemplateIntersectable], template_cls
            # Note: this is a little fragile; if you start abusing the preload
            # to do things it isn't meant to do, you might break things (if
            # the other templates haven't been loaded yet)
            )._templatey_signature.fieldset.dynamic_class_slot_names}
        target_slot_classes = inclusions_with_precall | inclusions_with_dynacls

        # We use this to detect and resolve recursion loops
        provisioned_node_by_id: dict[int, PrerenderTreeNode] = {}

        # Setting the root node to None in advance makes it trivial to detect
        # an empty result. Or more accurately, it means we don't have to detect
        # anything; we can just use the value directly
        root_frame = _PrerenderTreeBuilderFrame.provision(
            slot_tree_node=self,
            template_preload=template_preload,
            target_slot_classes=target_slot_classes,
            provisioned_node_by_id=provisioned_node_by_id)
        # Early return: if we don't need a frame for the root, then it means
        # that the entire tree ~~is boring~~ has no dynamic classes and no
        # precalls and therefore no prerender tree
        if root_frame is None:
            return

        root_node = root_frame.pending_prerender_tree_node
        stack: list[_PrerenderTreeBuilderFrame] = [root_frame]
        while stack:
            frame = stack[-1]

            if frame.exhausted:
                stack.pop()
                continue

            slot_name, slot_cls, nested_slot_tree_node = frame.advance()
            recursive_node = provisioned_node_by_id.get(
                nested_slot_tree_node.id_)

            if recursive_node is None:
                deeper_frame = _PrerenderTreeBuilderFrame.provision(
                    slot_tree_node=nested_slot_tree_node,
                    template_preload=template_preload,
                    target_slot_classes=target_slot_classes,
                    provisioned_node_by_id=provisioned_node_by_id)
                if deeper_frame is not None:
                    frame.pending_prerender_tree_node.append(
                        PrerenderTreeRoute.new(
                            slot_name,
                            slot_cls,
                            deeper_frame.pending_prerender_tree_node))
                    stack.append(deeper_frame)

            # This could be either a direct/trivial recursion or a recursion
            # chain; either way it'll be resolved.
            else:
                frame.pending_prerender_tree_node.append(
                    PrerenderTreeRoute.new(
                        slot_name, slot_cls, recursive_node))

        return root_node

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
SlotTreeNode._children = _ProxyDescriptor()  # type: ignore


@dataclass(slots=True)
class _PrerenderTreeBuilderFrame:
    """``_PrerenderTreeBuilderFrame`` instances are responsible for
    building up a prerender tree from a slot tree.
    """
    slot_tree_node: SlotTreeNode
    pending_prerender_tree_node: PrerenderTreeNode
    child_index: int = 0

    @property
    def exhausted(self) -> bool:
        return self.child_index >= len(self.slot_tree_node)

    def advance(self) -> _SlotTreeRoute:
        next_route = self.slot_tree_node[self.child_index]
        self.child_index += 1
        return next_route

    @classmethod
    def provision(
            cls,
            slot_tree_node: SlotTreeNode,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            target_slot_classes: set[TemplateClass],
            provisioned_node_by_id: dict[int, PrerenderTreeNode],
            ) -> _PrerenderTreeBuilderFrame | None:
        """Call this to create a new frame if, and only if, one is
        required. Returns the created frame if one is needed; otherwise
        (ie, if the branch should be culled) returns None.
        """
        slot_cls = slot_tree_node.slot_cls
        slot_signature = cast(
            type[TemplateIntersectable], slot_cls)._templatey_signature

        # This is pretty simple. We've already gone to the effort of
        # calculating the total inclusions for everything. Now we can just
        # check to see if there's any overlap with the target classes; if so,
        # we need to keep it, but if not, well, we can discard.
        # This fully encapsulates all of the complicated logic checks we'd
        # otherwise be doing into a simple set intersection.
        if target_slot_classes & slot_signature.total_inclusions:
            parsed_template = template_preload[slot_cls]

            new_node = PrerenderTreeNode(
                # This is partly for convenience (by keeping the
                # IDs the same it's easier to see what's going on)
                # but is also critical for resolving recursion
                # (see below)
                id_=slot_tree_node.id_,
                abstract_calls=tuple(itertools.chain.from_iterable(
                    parsed_template.function_calls.values())),
                dynamic_slot_names=tuple(
                    # Sorting here to maintain consistent ordering; can be
                    # helpful with tests
                    sorted(slot_tree_node.dynamic_slot_names)))
            provisioned_node_by_id[new_node.id_] = new_node

            return cls(
                slot_tree_node=slot_tree_node,
                pending_prerender_tree_node=new_node)

        # Implicit second case: if there's no overlap between the total
        # inclusions and the target classes, then we never update the new_node,
        # and this returns None (relevant for the root node) and doesn't append
        # the route to the parent (relevant for other nodes) -- therefore
        # culling the node and its children.
        return None


@dataclass(slots=True)
class _SlotTreeBuilderFrame:
    """``_SlotTreeBuilderFrame`` instances are used to build up a
    slot tree, which is then used to construct the prerender tree.
    """
    slot_cls: TemplateClass
    pending_slot_tree_node: SlotTreeNode
    dynamic_slot_names: frozenset[str]

    slotpaths: list[SlotPath]
    slotpath_index: int = field(default=0, init=False)

    first_encounters: dict[TemplateClass, SlotTreeNode] = field(
        repr=False, kw_only=True)

    def advance(self) -> SlotPath:
        next_slot_path = self.slotpaths[self.slotpath_index]
        self.slotpath_index += 1
        return next_slot_path

    @property
    def exhausted(self) -> bool:
        return self.slotpath_index >= len(self.slotpaths)

    def __post_init__(self):
        # Maybe a bit redundant, but also very defensive
        if self.slot_cls not in self.first_encounters:
            self.first_encounters[self.slot_cls] = self.pending_slot_tree_node

    @classmethod
    def from_slot_cls(
            cls,
            slot_cls: TemplateClass,
            pending_slot_tree_node: SlotTreeNode,
            *,
            first_encounters: dict[TemplateClass, SlotTreeNode] | None = None
            ) -> _SlotTreeBuilderFrame:
        """Constructs a new slot tree builder frame for the passed
        template class. **Note that the pending node is for the passed
        slot_cls!**
        """
        if first_encounters is None:
            first_encounters = {slot_cls: pending_slot_tree_node}

        fieldset = cast(
            type[TemplateIntersectable], slot_cls
        )._templatey_signature.fieldset
        return cls(
            slot_cls=slot_cls,
            pending_slot_tree_node=pending_slot_tree_node,
            slotpaths=list(fieldset.slotpaths),
            dynamic_slot_names=fieldset.dynamic_class_slot_names,
            first_encounters=first_encounters)

    def make_child(
            self,
            slot_cls: TemplateClass,
            pending_slot_tree_node: SlotTreeNode
            ) -> _SlotTreeBuilderFrame:
        """Creates a child frame, correctly handling the first
        encounters for you.
        """
        new_first_encounters: dict[TemplateClass, SlotTreeNode] = {
            **self.first_encounters}
        if slot_cls not in new_first_encounters:
            new_first_encounters[slot_cls] = pending_slot_tree_node

        return self.from_slot_cls(
            slot_cls,
            pending_slot_tree_node,
            first_encounters=new_first_encounters)


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
            # Note: dynamic template classes neither add a frame to the stack,
            # nor are they stored like explicit classes; instead, they simply
            # become string values stored on the pending node.
            # Do this during exhaustion so it only happens once per frame
            # instead of once per iteration.
            frame.pending_slot_tree_node.dynamic_slot_names.update(
                frame.dynamic_slot_names)
            continue

        nested_slot_name, nested_slot_cls = frame.advance()

        # In the simple recursion case -- a template defines a slot of its
        # own class -- we can immediately create a reference loop without
        # any pomp nor circumstance.
        if nested_slot_cls is frame.slot_cls:
            trivially_recursive_slot_route = (
                nested_slot_name,
                nested_slot_cls,
                frame.pending_slot_tree_node)
            frame.pending_slot_tree_node.append(trivially_recursive_slot_route)
            # Note that we don't need to add this to the recursion sources,
            # because trivial recursion can't influence the culling of the tree

        # In the slightly more complicated recursion case -- a template defines
        # a slot of an already-encountered class -- we just need to retrieve
        # the previous node.
        elif (
            first_encounter := frame.first_encounters.get(nested_slot_cls)
        ) is not None:
            mutually_recursive_slot_route = (
                nested_slot_name,
                nested_slot_cls,
                first_encounter)
            frame.pending_slot_tree_node.append(mutually_recursive_slot_route)

        # In the non-recursive case, we need to descend deeper into the
        # dependency graph.
        else:
            next_node = frame.pending_slot_tree_node.add_child(
                nested_slot_name, nested_slot_cls)
            next_frame = frame.make_child(nested_slot_cls, next_node)
            stack.append(next_frame)

    return root_node
