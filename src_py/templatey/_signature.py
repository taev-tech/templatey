from __future__ import annotations

import typing
from collections import defaultdict
from collections import namedtuple
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from textwrap import dedent
from types import UnionType
from typing import Any
from typing import TypeAliasType
from typing import cast
from typing import get_args as get_type_args
from typing import get_origin as get_type_origin
from typing import get_type_hints
from weakref import ref

from templatey._fields import NormalizedFieldset
from templatey._fields import SlotFieldAnnotation
from templatey._fields import ensure_normalized_fieldset
from templatey._provenance import Provenance
from templatey._slot_tree import ConcretePrerenderTreeNode
from templatey._slot_tree import DynamicClassPrerenderTreeNode
from templatey._slot_tree import SlotTreeNode
from templatey._slot_tree import PrerenderTreeNode
from templatey._slot_tree import PrerenderTreeRoute
from templatey._slot_tree import gather_dynamic_class_slots
from templatey._slot_tree import merge_into_prerender_tree
from templatey._slot_tree import update_encloser_with_trees_from_slot
from templatey._types import Content
from templatey._types import DynamicClassSlot
from templatey._types import InterfaceAnnotationFlavor
from templatey._types import Slot
from templatey._types import TemplateClass
from templatey._types import TemplateInstanceID
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import Var

if typing.TYPE_CHECKING:
    from templatey.environments import AsyncTemplateLoader
    from templatey.environments import SyncTemplateLoader
    from templatey.templates import FieldConfig
    from templatey.templates import SegmentModifier
    from templatey.templates import TemplateConfig

type GroupedTemplateInvocations = dict[TemplateClass, list[Provenance]]
type TemplateLookupByID = dict[TemplateInstanceID, TemplateParamsInstance]


def ensure_signature(template_cls: TemplateClass) -> TemplateSignature:
    """This verifies that a template class has a signature calculated,
    or -- if missing -- calculates it and sets it on the template class
    (and all its slots, recursively).

    This must be done after all forward refs have been resolved.
    """
    template_xable = cast(type[TemplateIntersectable], template_cls)
    if hasattr(template_xable, '_templatey_signature'):
        return template_xable._templatey_signature

    fieldset = ensure_normalized_fieldset(template_cls)

    template_xable._templatey_signature = signature = TemplateSignature.new(
        template_cls=template_xable,
        data_names=frozenset(field.name for field in fieldset.data),
        slots={field.name: field.type_ for field in fieldset.slots},
        dynamic_class_slot_names=frozenset(
            field.name for field in fieldset.dynamic_slots),
        var_names=frozenset(field.name for field in fieldset.vars_),
        content_names=frozenset(field.name for field in fieldset.content))

    return signature


@dataclass(slots=True, kw_only=True)
class TemplateSignature2:
    """Signature objects are created immediately upon template
    definition time, and are populated with information on the template
    as it becomes available. Any object decorated with ``@template``
    will have a signature, but not all items in the signature will be
    available at all points of the template's lifecycle.
    """
    # These are all available at template definition time and set then
    config: TemplateConfig
    # Note: whatever kind of object this is, it needs to be understood by the
    # template loader defined in the template environment.
    # In theory we could make this a typevar, but in practice the overarching
    # ``TemplateIntersectable`` would need to have a typevar within a classvar,
    # which python doesn't currently support.
    resource_locator: object
    segment_modifiers: tuple[SegmentModifier, ...]
    # Used primarily for libraries shipping redistributable templates
    explicit_loader: AsyncTemplateLoader | SyncTemplateLoader | None

    # These are all set during the template loading process, in stages, as
    # increasingly more information is available.
    fieldset: NormalizedFieldset = field(init=False, repr=False)
    total_inclusions: frozenset[TemplateClass] = field(
        init=False, repr=False)
    slot_tree: SlotTreeNode = field(init=False, repr=False)
    prerender_tree: PrerenderTreeNode = field(init=False, repr=False)

    def repr_2(self):
        """This wraps the default repr, including any other non-init
        vars. We chose this as the least-bad way to get reprs to work
        while still having a way to debug the signature manually.
        """
        bare_repr = repr(self)
        noninit_fields: list[str] = []
        for dc_field in fields(self):
            if not dc_field.init:
                noninit_fields.append(dc_field.name)

        to_join = [bare_repr[:-1]]
        for noninit_fieldname in noninit_fields:
            try:
                value = getattr(self, noninit_fieldname)
            except AttributeError:
                value = '<unset>'

            to_join.append(f'{noninit_fieldname}={value}')

        joined = ', '.join(to_join)
        return f'{joined})'


@dataclass(slots=True)
class TemplateSignature:
    """This class stores the processed interface based on the params.
    It gets used to compare with the TemplateParse to make sure that
    the two can be used together.

    Not meant to be created directly; instead, you should use the
    TemplateSignature.new() convenience method.

    TODO: we need to get way more consistent about public/private attr
    conventions; this is a holdover from when this was within a public
    module.
    """
    # It's nice to have this available, especially when resolving forward refs,
    # but unlike eg the prerender tree, it's trivially easy for us to avoid GC
    # loops within the signature
    template_cls_ref: ref[TemplateClass]

    data_names: frozenset[str]
    var_names: frozenset[str]
    content_names: frozenset[str]
    # Note that these are all ONLY the direct nesteds; these do not include
    # anything from deeper in the prerender tree.
    slot_names: frozenset[str]
    dynamic_class_slot_names: frozenset[str]

    _dynamic_class_prerender_tree: DynamicClassPrerenderTreeNode
    _ordered_dynamic_class_slot_names: list[str] = field(init=False)
    # Note that these contain all included types, not just the ones on the
    # outermost layer that are associated with the signature. In other words,
    # they include the flattened recursion of all included slots, all the way
    # down the tree
    _prerender_tree_lookup: dict[TemplateClass, ConcretePrerenderTreeNode]

    # I really don't like that we need to remember to recalculate this every
    # time we update the prerender tree lookup, but for rendering performance
    # reasons we want this to be precalculated before every call to render.
    included_template_classes: frozenset[TemplateClass] = field(init=False)

    def __post_init__(self):
        self._ordered_dynamic_class_slot_names = sorted(
            self.dynamic_class_slot_names)

    @classmethod
    def new(
            cls,
            template_cls: type,
            slots: dict[str, SlotFieldAnnotation],
            dynamic_class_slot_names: frozenset[str],
            data_names: frozenset[str],
            var_names: frozenset[str],
            content_names: frozenset[str],
            ) -> TemplateSignature:
        """Create a new TemplateSignature based on the gathered slots,
        vars, and content. This does all of the convenience calculations
        needed to populate the semi-redundant fields.
        """
        slot_names = frozenset(slots)

        # Quick refresher: our goal here is to construct a lookup that gets
        # us a route to every instance of a particular template type. In other
        # words, we want to be able to check a template type, and then see all
        # possible getattr() sequences that arrive at an instance of that
        # template type.
        tree_wip: dict[TemplateClass, ConcretePrerenderTreeNode]
        tree_wip = defaultdict(PrerenderTreeNode)

        concrete_slot_defs = cls._normalize_slot_defs(slots.items())

        # Note that order between concrete_slot_defs and pending_ref_defs
        # DOES matter here. Concrete has to come first, because we need to
        # discover all reference loops back to the template class before
        # we can fully define the pending trees.
        for slot_name, slot_type in concrete_slot_defs:
            # In the simple recursion case -- a template defines a slot of its
            # own class -- we can immediately create a reference loop without
            # any pomp nor circumstance.
            # Also: yes, this is resolved at annotation time, and not a forward
            # ref!
            if slot_type is template_cls:
                recursive_prerender_tree = tree_wip[slot_type]
                recursive_prerender_tree.is_terminus = True
                recursive_prerender_tree.is_recursive = True
                recursive_slot_route = PrerenderTreeRoute.new(
                    slot_name,
                    slot_type,
                    recursive_prerender_tree)
                recursive_prerender_tree.append(recursive_slot_route)

            # Remember that we expanded the union already, so this is
            # guaranteed to be a single concrete ``slot_type``.
            else:
                update_encloser_with_trees_from_slot(
                    template_cls,
                    tree_wip,
                    slot_type,
                    slot_name,)

        # Okay, so here's the deal. Every time we add a slot,
        # might create a recursive reference loop (if it had a forward ref to
        # the current enclosing class). So far so good. The problem
        # is, this means that existing prerender trees will be incomplete.
        # Therefore, at the end of the process, we always circle back and check
        # for a self-referential prerender tree, and if it exists, merge it into
        # all the rest.
        self_referential_recursive_tree = tree_wip.get(template_cls)
        if self_referential_recursive_tree is not None:
            for slot_type, prerender_tree in tree_wip.items():
                if slot_type is not template_cls:
                    merge_into_prerender_tree(
                        template_cls,
                        slot_type,
                        prerender_tree,
                        self_referential_recursive_tree)

        # Oh thank god.
        tree_wip.default_factory = None
        return cls(
            template_cls_ref=ref(template_cls),
            slot_names=slot_names,
            dynamic_class_slot_names=dynamic_class_slot_names,
            _dynamic_class_prerender_tree=gather_dynamic_class_slots(
                template_cls,
                set(dynamic_class_slot_names),
                tree_wip),
            data_names=data_names,
            var_names=var_names,
            content_names=content_names,
            _prerender_tree_lookup=tree_wip)

    @classmethod
    def _normalize_slot_defs(
            cls,
            slot_defs: Iterable[tuple[str, SlotFieldAnnotation]]
            ) -> list[tuple[str, TemplateClass]]:
        """The annotations we get "straight off the tap" (so to speak)
        of the template class can be:
        ++  unions
        ++  type aliases, though we don't quite support these yet
            (though this function should make it relatively
            straightforward to do so)
        ++  concrete backrefs to existing templates
        ++  pending forward refs to not-yet-defined templates

        This function is responsible for giving us a clean, uniform way
        of representing them. First, it expands all unions etc into
        multiple slot paths. Second, it splits the concrete from the
        pending nodes, returning them separately.
        """
        concrete_slots = []

        for slot_name, slot_annotation in slot_defs:
            for flattened_slot_annotation in (
                _recursively_flatten_slot_annotations(slot_annotation)
            ):
                concrete_slots.append((slot_name, flattened_slot_annotation))

        return concrete_slots

    def stringify_all(self) -> str:
        """This is a debug method that creates a prettified string
        version of the entire prerender tree lookup (pending and concrete).
        """
        to_join = []
        to_join.append('prerender tree slots:')
        for template_class, root_node in self._prerender_tree_lookup.items():
            to_join.append(f'  {template_class}')
            to_join.append(root_node.stringify(depth=1))

        return '\n'.join(to_join)
