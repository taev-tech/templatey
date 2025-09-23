from __future__ import annotations

from collections import defaultdict
from collections import namedtuple
from collections.abc import Iterable
from collections.abc import Iterator
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

from templatey._provenance import Provenance
from templatey._slot_tree import ConcreteSlotTreeNode
from templatey._slot_tree import DynamicClassSlotTreeNode
from templatey._slot_tree import SlotTreeNode
from templatey._slot_tree import SlotTreeRoute
from templatey._slot_tree import gather_dynamic_class_slots
from templatey._slot_tree import merge_into_slot_tree
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
from templatey.templates import FieldConfig

type SlotFieldAnnotation = (
    TemplateClass
    | UnionType
    | TypeAliasType)


@dataclass(slots=True, frozen=True, kw_only=True)
class NormalizedField:
    name: str
    config: FieldConfig


@dataclass(slots=True, frozen=True, kw_only=True)
class NormalizedSlotField(NormalizedField):
    type_: SlotFieldAnnotation


@dataclass(slots=True, frozen=True)
class NormalizedFieldset:
    slots: tuple[NormalizedSlotField, ...]
    dynamic_slots: tuple[NormalizedField, ...]
    vars_: tuple[NormalizedField, ...]
    content: tuple[NormalizedField, ...]
    data: tuple[NormalizedField, ...]


def ensure_normalized_fieldset(
        template_cls: TemplateClass
        ) -> NormalizedFieldset:
    template_xable = cast(type[TemplateIntersectable], template_cls)
    if hasattr(template_xable, '_templatey_fieldset'):
        return template_xable._templatey_fieldset

    try:
        template_type_hints = get_type_hints(template_cls)
    except NameError as exc:
        exc.add_note(dedent('''\
            Failed to resolve type hints on template. This is either the
            result of an unresolved forward ref (which should have been
            caught by your type checker), or circular imports hidden behind
            ``if typing.TYPE_CHECKING`` blocks. These must be resolved before
            calculating a template signature.'''))
        raise exc

    slots = []
    dynamic_slots = []
    vars_ = []
    content = []
    data = []
    prerenderers = {}
    # Note that this ignores initvars, which is what we want
    for template_field in fields(template_cls):
        field_classification = _classify_interface_field_flavor(
            template_type_hints, template_field)

        # Note: it's not entirely clear to me that this restriction makes
        # sense; I could potentially see MAYBE there being some kind of
        # environment function that could access other attributes from the
        # dataclass? But also, maybe those should be vars? Again, unclear.
        if field_classification is None:
            raise TypeError(
                'Template parameter definitions may only contain variables, '
                + 'slots, and content!')

        else:
            field_flavors, wrapped_type = field_classification

            # In case it isn't obvious, the .get() here is because we don't
            # require all fields to use the ``template_field`` specifier.
            field_config: FieldConfig | None = template_field.metadata.get(
                'templatey.field_config')
            if field_config is None:
                field_config = FieldConfig()

            # Just for maintainability, we're doing this first here and then
            # replacing it for slots; that way, we don't have a bunch of these
            normtype: NormalizedField | NormalizedSlotField = NormalizedField(
                name=template_field.name,
                config=field_config)

            # A little awkward to effectively just repeat the comparison we did
            # when classifying, but that makes testing easier and control flow
            # clearer
            if InterfaceAnnotationFlavor.VARIABLE in field_flavors:
                vars_.append(normtype)

            elif InterfaceAnnotationFlavor.SLOT in field_flavors:
                if InterfaceAnnotationFlavor.DYNAMIC in field_flavors:
                    dynamic_slots.append(normtype)

                else:
                    normtype = NormalizedSlotField(
                        name=normtype.name,
                        config=normtype.config,
                        type_=cast(SlotFieldAnnotation, wrapped_type))
                    slots.append(normtype)

            elif InterfaceAnnotationFlavor.CONTENT in field_flavors:
                content.append(normtype)

            else:
                data.append(normtype)

            prerenderers[template_field.name] = normtype.config.prerenderer

    prerenderer_cls = namedtuple('TemplateyPrerenderers', tuple(prerenderers))
    template_xable._templatey_prerenderers = prerenderer_cls(**prerenderers)

    fieldset = NormalizedFieldset(
        slots=tuple(slots),
        dynamic_slots=tuple(dynamic_slots),
        vars_=tuple(vars_),
        content=tuple(content),
        data=tuple(data))
    template_xable._templatey_fieldset = fieldset
    return fieldset


# Yes, this is awkward with a bazillion return statements, but in this case,
# clarity is better than elegance
def _classify_interface_field_flavor(  # noqa: PLR0911
        parent_class_type_hints: dict[str, Any],
        template_field: Field
        ) -> tuple[set[InterfaceAnnotationFlavor], type | None]:
    """For a dataclass field, determines whether it was declared as a
    var, slot, or content.

    If none of the above, returns None.
    """
    # Note that dataclasses don't include the actual type (just a string)
    # when in __future__ mode, so we need to get them from the parent class
    # by calling get_type_hints() on it
    resolved_field_type = parent_class_type_hints[template_field.name]
    anno_origin = get_type_origin(resolved_field_type)
    if anno_origin is Var:
        nested_type, = get_type_args(resolved_field_type)
        return {InterfaceAnnotationFlavor.VARIABLE}, nested_type
    elif anno_origin is Slot:
        nested_type, = get_type_args(resolved_field_type)
        return {InterfaceAnnotationFlavor.SLOT}, nested_type
    elif anno_origin is Content:
        nested_type, = get_type_args(resolved_field_type)
        return {InterfaceAnnotationFlavor.CONTENT}, nested_type
    elif anno_origin is DynamicClassSlot:
        return {
            InterfaceAnnotationFlavor.SLOT,
            InterfaceAnnotationFlavor.DYNAMIC
        }, None

    # This is all if there's a generic annotation with no parameter passed,
    # ex ``foo: Var`` or ``bar: Slot`` or (presumably) ``baz: list``
    elif anno_origin is None:
        if resolved_field_type is Var:
            return {InterfaceAnnotationFlavor.VARIABLE}, None

        elif resolved_field_type is Content:
            return {InterfaceAnnotationFlavor.CONTENT}, None

        elif resolved_field_type is DynamicClassSlot:
            return {
                InterfaceAnnotationFlavor.SLOT,
                InterfaceAnnotationFlavor.DYNAMIC
            }, None

        elif resolved_field_type is Slot:
            raise TypeError(
                '``Slot`` annotations require a concrete slot class as a '
                + 'parameter!')

        else:
            return {InterfaceAnnotationFlavor.DATA}, None

    # There was a non-generic parameter passed. Therefore, it can only be
    # one thing: template data.
    else:
        return {InterfaceAnnotationFlavor.DATA}, None


def recursively_flatten_slot_annotations(
        slot_annotation: SlotFieldAnnotation
        ) -> Iterator[TemplateClass]:
    # Note that this still might contain a heterogeneous mix of
    # template classes and forward refs! Hence flattening first.
    if isinstance(slot_annotation, UnionType):
        for union_member in slot_annotation.__args__:
            yield from recursively_flatten_slot_annotations(union_member)

    elif isinstance(slot_annotation, TypeAliasType):
        yield from recursively_flatten_slot_annotations(
            slot_annotation.__value__)

    else:
        yield slot_annotation
