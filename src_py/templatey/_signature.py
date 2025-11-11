from __future__ import annotations

import typing
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields

from templatey._provenance import Provenance
from templatey._types import TemplateClass
from templatey._types import TemplateClassInstance
from templatey._types import TemplateClassInstanceID

if typing.TYPE_CHECKING:
    from templatey._fields import NormalizedFieldset
    from templatey._slot_tree import PrerenderTreeNode
    from templatey._slot_tree import SlotTreeNode
    from templatey.environments import AsyncTemplateLoader
    from templatey.environments import SyncTemplateLoader
    from templatey.templates import ParseConfig
    from templatey.templates import RenderConfig

type GroupedTemplateInvocations = dict[TemplateClass, list[Provenance]]
type TemplateLookupByID = dict[TemplateClassInstanceID, TemplateClassInstance]


@dataclass(slots=True, kw_only=True)
class TemplateSignature:
    """Signature objects are created immediately upon template
    definition time, and are populated with information on the template
    as it becomes available. Any object decorated with ``@template``
    will have a signature, but not all items in the signature will be
    available at all points of the template's lifecycle.
    """
    # These are all available at template definition time and set then
    parse_config: ParseConfig
    render_config: RenderConfig
    # Note: whatever kind of object this is, it needs to be understood by the
    # template loader defined in the template environment.
    # In theory we could make this a typevar, but in practice the overarching
    # ``TemplateIntersectable`` would need to have a typevar within a classvar,
    # which python doesn't currently support.
    resource_locator: object
    # Used primarily for libraries shipping redistributable templates
    # TODO: this should be moved into an EnvConfig object here instead of
    # extracted from it.
    explicit_loader: AsyncTemplateLoader | SyncTemplateLoader | None

    # These are all set during the template loading process, in stages, as
    # increasingly more information is available.
    fieldset: NormalizedFieldset = field(init=False, repr=False)
    total_inclusions: frozenset[TemplateClass] = field(
        init=False, repr=False)
    slot_tree: SlotTreeNode = field(init=False, repr=False)
    # NOTE: a value of None here means that the template HAS NOTHING TO
    # RENDER PREP -- and **not** that the tree hasn't been created yet!
    prerender_tree: PrerenderTreeNode | None = field(init=False, repr=False)

    def expanded_repr(self):
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
