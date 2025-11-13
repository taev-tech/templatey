from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import KW_ONLY
from dataclasses import dataclass
from dataclasses import field
from functools import partialmethod
from functools import singledispatch
from typing import Any
from typing import cast
from typing import overload

from templatey._error_collector import ErrorCollector
from templatey._error_collector import capture_traceback
from templatey._types import TemplateClass
from templatey._types import TemplateClassInstance
from templatey._types import TemplateClassInstanceID
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import ParsedTemplateResource
from templatey.parser import TemplateInstanceContentRef
from templatey.parser import TemplateInstanceDataRef
from templatey.parser import TemplateInstanceVariableRef


@dataclass(slots=True, frozen=True)
class ProvenanceNode:
    """ProvenanceNode instances are unique to the exact location
    on the exact render tree of a particular template instance. If the
    template instance gets reused within the same render tree, it will
    have multiple provenance nodes. And each different slot in an
    enclosing template will have a separate provenance node, potentially
    with different namespace overrides.

    This is used both during function execution (to calculate the
    concrete value of any parameters), and also while flattening the
    render tree.

    Also note that the root template, **along with any templates
    injected into the render by environment functions,** will have an
    empty list of provenance nodes.

    Note that, since overrides from the enclosing template can come
    exclusively from the template body -- and are therefore shared
    across all nested children of the same slot -- they don't get stored
    within the provenance, since we'd require access to the template
    bodies, which we don't yet have.
    """
    encloser_slot_key: str
    encloser_slot_index: int
    # Note: this is a little awkward. On the one hand, we need to store this
    # during rendering so that we can recover the part index to do the correct
    # param lookup during param binding. On the other hand, this can't be known
    # ahead of time, so the slot tree cannot know it.
    # Our compromise is to make it non-comparable, and set it to -1 in the
    # slot tree.
    encloser_part_index: int = field(compare=False)
    # The reason to have both the instance and the instance ID is so that we
    # can have hashability of the ID while not imposing an API on the instances
    instance_id: TemplateClassInstanceID
    instance: TemplateClassInstance = field(compare=False, repr=False)


@dataclass(slots=True, frozen=True)
class Provenance:
    """
    """
    slotpath: tuple[ProvenanceNode, ...] = field(default=())
    _: KW_ONLY
    from_injection: Provenance | None = None

    # Used for memoization
    _hash: int | None = field(
        default=None, compare=False, init=False, repr=False)

    def with_appended(
            self,
            node_to_append: ProvenanceNode,
            *,
            dynamic: bool = False
            ) -> Provenance:
        # This is a little awkward. I think ideally we'd refactor things so
        # that the provenance module is always responsible for constructing
        # provenance instances, and then we wouldn't be discarding the passed
        # node. But that cleanup can be saved for another time.
        if dynamic:
            return Provenance(
                slotpath=(ProvenanceNode(
                    encloser_slot_key='',
                    encloser_slot_index=-1,
                    encloser_part_index=-1,
                    instance_id=node_to_append.instance_id,
                    instance=node_to_append.instance),),
                from_injection=self)
        else:
            return Provenance(
                (*self.slotpath, node_to_append),
                from_injection=self.from_injection)

    def bind_call_signature(
            self,
            abstract_call: InterpolatedFunctionCall,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            error_collector: ErrorCollector,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Given a passed abstract call signature, extracts and
        finalizes the correct *args and **kwargs for its execution
        request.
        """
        # Note that the full call signature is **defined**
        # within the parsed template body, but it may
        # **reference** vars and/or content within the template
        # instance.
        try:
            args = _recursively_coerce_func_execution_params(
                abstract_call.call_args,
                provenance=self,
                template_preload=template_preload,
                error_collector=error_collector)
            kwargs = _recursively_coerce_func_execution_params(
                abstract_call.call_kwargs,
                provenance=self,
                template_preload=template_preload,
                error_collector=error_collector)

            if abstract_call.call_args_exp is None:
                # hmm, I actually don't think this is possible, but we'd need
                # to fix some typing on
                # _recursively_coerce_func_execution_params to know for sure
                if not isinstance(args, tuple):
                    args = tuple(args)

            else:
                args = (*args, *cast(
                    Iterable[Any],
                    _recursively_coerce_func_execution_params(
                        abstract_call.call_args_exp,
                        provenance=self,
                        template_preload=template_preload,
                        error_collector=error_collector)))

            if abstract_call.call_kwargs_exp is not None:
                kwargs.update(cast(
                    Mapping[str, Any],
                    _recursively_coerce_func_execution_params(
                        abstract_call.call_kwargs_exp,
                        provenance=self,
                        template_preload=template_preload,
                        error_collector=error_collector)))

        except Exception as exc:
            error_collector.append(exc)
            # Doesn't matter what these are; they just need to be defined.
            # The rest will get sorted out when we check for errors down the
            # road.
            args = ()
            kwargs = {}

        return args, kwargs

    def _bind_param(
            self,
            name: str,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            error_collector: ErrorCollector,
            *,
            _passthrough_reftype:
                type[TemplateInstanceContentRef]
                | type[TemplateInstanceVariableRef]
            ) -> object:
        """Use this to calculate a concrete value for use in rendering.
        We start from the template params instance, and then -- if the
        value is missing -- we walk up the provenance stack until we
        either find a value or reach the injection point.
        """
        slotpath = self.slotpath
        current_provenance_node = slotpath[-1]
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        value = getattr(current_provenance_node.instance, name, ...)

        # Template params instances take precedence over values passed in via
        # template text. In that case, we can simple short-circuit.
        if value is not ...:
            return value

        # Otherwise, things get a bit weird. Each level in the provenance can
        # reference a value from the parent, but under a different name. So
        # we need to do some remapping as we walk things back.
        param_name = name
        slot_key = current_provenance_node.encloser_slot_key
        part_index = current_provenance_node.encloser_part_index
        for encloser in reversed(slotpath[0:-1]):
            enclosing_template_cls = type(encloser.instance)
            # We do this so that env funcs that inject templates don't try
            # to continue looking up the provenance tree for slots that
            # don't actually exist.
            if hasattr(enclosing_template_cls, '_TEMPLATEY_EMPTY_INSTANCE'):
                break

            params_from_encloser = (
                template_preload[enclosing_template_cls]
                .slots[(slot_key, part_index)]
                .params)
            value_from_encloser = params_from_encloser.get(param_name, ...)

            # If the value in the enclosure was itself a reference to a
            # parameter passed to it, we then need to check one level up.
            if isinstance(value_from_encloser, _passthrough_reftype):
                # First, prep the next iteration, in case we need one.
                slot_key = encloser.encloser_slot_key
                part_index = encloser.encloser_part_index
                param_name = value_from_encloser.name
                # Now, update the value from the enclosure, in case we don't
                # need a next iteration. Note that we use the NEW param name
                # here!
                value = getattr(encloser.instance, param_name, ...)

            # This might be a missing value still (the ellipsis), or an
            # explicit literal value from the template body. Either way, we'll
            # check in a second.
            else:
                value = value_from_encloser

            # Case 1: ``value is not ...`` -- this means we found a value, and
            # we need to stop searching.
            # Case 2: ``value is ... and value_from_encloser is ...`` -- this
            # means there was no explicit literal value, and also no indirect
            # value to check in the next enclosure. We can't continue the
            # search; no value was found.
            if value is not ... or value_from_encloser is ...:
                break

        if value is ...:
            error_collector.append(capture_traceback(
                KeyError(
                    'No param value found with matching type (content vs '
                    + 'variable) on template instance or text!',
                    slotpath[-1].instance, name)))
            value = ''

        return value

    bind_content = partialmethod(
        _bind_param, _passthrough_reftype=TemplateInstanceContentRef)
    bind_variable = partialmethod(
        _bind_param, _passthrough_reftype=TemplateInstanceVariableRef)

    def __hash__(self) -> int:
        memoized = self._hash
        if memoized is None:
            retval = hash(self.slotpath) ^ hash(self.from_injection)
            object.__setattr__(self, '_hash', retval)
            return retval

        else:
            return memoized


@overload
def _recursively_coerce_func_execution_params(
        param_value: str,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> str: ...
@overload
def _recursively_coerce_func_execution_params[K: object, V: object](
        param_value: Mapping[K, V],
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> dict[K, V]: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: list[T] | tuple[T],
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> tuple[T]: ...
@overload
def _recursively_coerce_func_execution_params(
        param_value: TemplateInstanceContentRef | TemplateInstanceVariableRef,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> object: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: T,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> T: ...
@singledispatch
def _recursively_coerce_func_execution_params(
        # Note: singledispatch doesn't support type vars
        param_value: object,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> object:
    """Templates support references to ``Content[...]``, ``Var[..]``,
    and data attributes (normal dataclass fields) on their corresponding
    template params instances as call args/kwargs for env functions.
    This function is responsible for converting abstract references to
    concrete values, using the template preload and the provenance.

    Note that, since env function params can also be iterables (lists)
    and mappings (dicts), we also need to recursively convert the
    contents of containers.
    """
    # Trivial case: plain object. Return it directly.
    return param_value


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        # Note: singledispatch doesn't support type vars
        param_value: list | tuple | dict,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> tuple | dict:
    """Again, in the container case, we want to create a new copy of
    the container, replacing its values with the recursive call.
    Note that the keys in nested dictionaries cannot be references,
    only the values.
    """
    if isinstance(param_value, dict):
        return {
            contained_key: _recursively_coerce_func_execution_params(
                contained_value,
                provenance=provenance,
                template_preload=template_preload,
                error_collector=error_collector)
            for contained_key, contained_value in param_value.items()}

    else:
        return tuple(
            _recursively_coerce_func_execution_params(
                contained_value,
                provenance=provenance,
                template_preload=template_preload,
                error_collector=error_collector)
            for contained_value in param_value)


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        # Note: singledispatch doesn't support type vars
        param_value: str,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> str:
    """We need to be careful here to supply a MORE SPECIFIC dispatch
    type than container for strings, since they are technically also
    containers. Bleh.
    """
    return param_value


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        param_value: TemplateInstanceContentRef,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> object:
    """Template content references need to be bound via the provenance.
    """
    return provenance.bind_content(
        param_value.name,
        template_preload,
        error_collector)


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        param_value: TemplateInstanceVariableRef,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> object:
    """Template variable references need to be bound via the provenance.
    """
    return provenance.bind_variable(
        param_value.name,
        template_preload,
        error_collector)


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        param_value: TemplateInstanceDataRef,
        *,
        provenance: Provenance,
        template_preload: dict[TemplateClass, ParsedTemplateResource],
        error_collector: ErrorCollector,
        ) -> object:
    """Data refs need to be retrieved from the template instance, which
    is always the last member of the provenance slotpath.
    """
    return getattr(provenance.slotpath[-1].instance, param_value.name)
