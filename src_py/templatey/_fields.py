from __future__ import annotations

import typing
from collections import namedtuple
from collections.abc import Iterator
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from textwrap import dedent
from types import UnionType
from typing import Any
from typing import NamedTuple
from typing import TypeAliasType
from typing import cast
from typing import get_args as get_type_args
from typing import get_origin as get_type_origin
from typing import get_type_hints

from templatey._types import Content
from templatey._types import DynamicClassSlot
from templatey._types import InterfaceAnnotationFlavor
from templatey._types import Slot
from templatey._types import TemplateClass
from templatey._types import Var
from templatey.templates import TemplateFieldConfig
from templatey.templates import get_closure_locals

if typing.TYPE_CHECKING:
    from templatey._slot_tree import SlotPath

type SlotFieldAnnotation = (
    TemplateClass
    | UnionType
    | TypeAliasType)


@dataclass(slots=True, frozen=True, kw_only=True)
class NormalizedField:
    name: str
    config: TemplateFieldConfig


@dataclass(slots=True, frozen=True, kw_only=True)
class NormalizedSlotField(NormalizedField):
    type_: SlotFieldAnnotation


@dataclass(slots=True)
class NormalizedFieldset:
    slots: tuple[NormalizedSlotField, ...]
    dynamic_slots: tuple[NormalizedField, ...]
    vars_: tuple[NormalizedField, ...]
    content: tuple[NormalizedField, ...]
    data: tuple[NormalizedField, ...]

    data_names: frozenset[str] = field(init=False)
    var_names: frozenset[str] = field(init=False)
    content_names: frozenset[str] = field(init=False)
    # Note that these are all ONLY the direct nesteds; these do not include
    # anything from deeper in the slot tree.
    slot_names: frozenset[str] = field(init=False)
    dynamic_class_slot_names: frozenset[str] = field(init=False)
    slotpaths: frozenset[SlotPath]

    # Oldschool here for performance reasons; otherwise this would be a dict.
    # Field names match the field names from the params; the value is gathered
    # from the metadata value on the field.
    transformers: NamedTuple

    def __post_init__(self):
        self.data_names = frozenset(field.name for field in self.data)
        self.var_names = frozenset(field.name for field in self.vars_)
        self.content_names = frozenset(field.name for field in self.content)
        self.slot_names = frozenset(field.name for field in self.slots)
        self.dynamic_class_slot_names = frozenset(
            field.name for field in self.dynamic_slots)

    # The noqa here is because of too many branches (because we're normalizing
    # stuff, basically)
    @classmethod
    def from_template_cls(  # noqa: PLR0912
            cls,
            template_cls: TemplateClass
            ) -> NormalizedFieldset:
        closure_locals = get_closure_locals(template_cls)
        try:
            if closure_locals is None:
                template_type_hints = get_type_hints(template_cls)
            else:
                template_type_hints = get_type_hints(
                    template_cls, localns=closure_locals)
        except NameError as exc:
            exc.add_note(dedent('''\
                Failed to resolve type hints on template. This is either the
                result of an unresolved forward ref (which should have been
                caught by your type checker), or circular imports hidden behind
                ``if typing.TYPE_CHECKING`` blocks. These must be resolved
                before calculating a template signature.'''))
            raise exc

        slotpaths: set[SlotPath] = set()
        slots = []
        dynamic_slots = []
        vars_ = []
        content = []
        data = []
        transformers = {}
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
                    '``template_field`` definitions may only contain '
                    + 'variables, slots, and content!')

            else:
                field_flavors, wrapped_type = field_classification

                # In case it isn't obvious, the .get() here is because we don't
                # require all fields to use the ``template_field`` specifier.
                field_config: TemplateFieldConfig | None = \
                    template_field.metadata.get(TemplateFieldConfig)
                if field_config is None:
                    field_config = TemplateFieldConfig()

                # Just for maintainability, we're doing this first here and
                # then replacing it for slots; that way, we don't have a bunch
                # of these
                normfield: NormalizedField | NormalizedSlotField = \
                    NormalizedField(
                        name=template_field.name,
                        config=field_config)

                # A little awkward to effectively just repeat the comparison we
                # did when classifying, but that makes testing easier and
                # control flow clearer
                if InterfaceAnnotationFlavor.VARIABLE in field_flavors:
                    vars_.append(normfield)

                elif InterfaceAnnotationFlavor.SLOT in field_flavors:
                    if InterfaceAnnotationFlavor.DYNAMIC in field_flavors:
                        dynamic_slots.append(normfield)

                    else:
                        slot_annotation = cast(
                            SlotFieldAnnotation, wrapped_type)
                        normfield = NormalizedSlotField(
                            name=normfield.name,
                            config=normfield.config,
                            type_=slot_annotation)
                        slots.append(normfield)
                        # Just to be clear: we're flattening unions etc here.
                        slotpaths.update(
                            (template_field.name, normtype)
                            for normtype in normalize_slot_annotations(
                                slot_annotation))

                elif InterfaceAnnotationFlavor.CONTENT in field_flavors:
                    content.append(normfield)

                else:
                    data.append(normfield)

                transformers[template_field.name] = \
                    normfield.config.transformer

        transformer_cls = namedtuple(
            'Templateytransformers', tuple(transformers))

        return NormalizedFieldset(
            slots=tuple(slots),
            dynamic_slots=tuple(dynamic_slots),
            vars_=tuple(vars_),
            content=tuple(content),
            data=tuple(data),
            transformers=transformer_cls(**transformers),
            slotpaths=frozenset(slotpaths))


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


def normalize_slot_annotations(
        slot_annotation: SlotFieldAnnotation
        ) -> Iterator[TemplateClass]:
    # Note that this still might contain a heterogeneous mix of
    # template classes and forward refs! Hence flattening first.
    if isinstance(slot_annotation, UnionType):
        for union_member in slot_annotation.__args__:
            yield from normalize_slot_annotations(union_member)

    elif isinstance(slot_annotation, TypeAliasType):
        yield from normalize_slot_annotations(
            slot_annotation.__value__)

    else:
        yield slot_annotation
