from __future__ import annotations

import logging
from collections.abc import Collection
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import KW_ONLY
from dataclasses import dataclass
from dataclasses import field
from pprint import pprint
from typing import Annotated
from typing import NamedTuple
from typing import cast

from docnote import Note

from templatey._bootstrapping import EMPTY_TEMPLATE_INSTANCE
from templatey._bootstrapping import EMPTY_TEMPLATE_XABLE
from templatey._error_collector import ErrorCollector
from templatey._error_collector import capture_traceback
from templatey._provenance import Provenance
from templatey._provenance import ProvenanceNode
from templatey._signature import TemplateSignature
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import is_template_instance_xable
from templatey.exceptions import TemplateFunctionFailure
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import ParsedTemplateResource
from templatey.templates import ComplexContent
from templatey.templates import InjectedValue
from templatey.templates import TemplateConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FuncExecutionRequest:
    """
    """
    name: str
    args: Iterable[object]
    kwargs: Mapping[str, object]
    result_key: PrecallCacheKey
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class FuncExecutionResult:
    """
    """
    # Note: must match signature from TemplateFunction!
    name: str
    retval: Sequence[str | TemplateParamsInstance | InjectedValue] | None
    exc: Exception | None

    def filter_injectables(self) -> Iterable[TemplateParamsInstance]:
        if self.retval is not None:
            for item in self.retval:
                if is_template_instance_xable(item):
                    # This doesn't fully work; because of the missing
                    # intersection type, there's nothing linking the xable that
                    # this checks for with the params instance that the type
                    # expects us to yield back
                    yield item  # type: ignore


@dataclass(slots=True)
class RenderEnvRequest:
    """
    """
    to_load: Annotated[
            Collection[TemplateClass],
            Note('''These are any root template **classes** that need loading.
                The render env is responsible for loading them (and any
                dependants) during the completion of the request.''')]
    to_execute: Collection[FuncExecutionRequest]
    error_collector: ErrorCollector

    # These get modified inplace
    injections: list[TemplateInjection]
    template_preload: dict[TemplateClass, ParsedTemplateResource]
    function_precall: dict[PrecallCacheKey, FuncExecutionResult]


# Yes, this is a way, way, way larger function than it should be.
# But function calls in python are slow, and we actually kinda care about
# performance here.
# TODO: look into solutions that would allow for inlining functions, so you
# could carve this up into separate functions. Can this be done without import
# hooks?
def render_driver(  # noqa: C901, PLR0912, PLR0915
        template_instance: TemplateParamsInstance,
        render_ctx: RenderContext,
        ) -> list[str]:
    """This is a shared method for driving rendering, used by both async
    and sync renderers. Note that the prerender step must already be
    complete.
    """
    error_collector = render_ctx.error_collector
    to_join: list[str] = []

    template_signature = cast(
        TemplateIntersectable, template_instance)._templatey_signature
    pprint(template_signature.slot_tree)
    pprint(template_signature.prerender_tree)
    function_precall = render_ctx.function_precall
    template_preload = render_ctx.template_preload
    root_template_preload = template_preload[type(template_instance)]
    render_stack: list[_RenderStackFrame] = [
        _RenderStackFrame(
            parts=root_template_preload.parts,
            part_count=root_template_preload.part_count,
            parse_config=template_signature.parse_config,
            signature=template_signature,
            provenance=Provenance((
                ProvenanceNode(
                    encloser_slot_key='',
                    encloser_slot_index=-1,
                    instance_id=id(template_instance),
                    instance=template_instance),)),
            instance=template_instance,
            transformers=template_signature.fieldset.transformers)]

    while render_stack:
        render_frame = render_stack[-1]
        if render_frame.exhausted:
            render_stack.pop()
            continue

        next_part = render_frame.parts[render_frame.part_index]

        # Quick note on the following code: you might think these isinstance
        # calls are slow, and that you could speed things up by, say, using
        # some kind of sentinel value in a slot. This is incorrect! (At least
        # with 3.13). Isinstance is actually the fastest (and clearest) way
        # to do it.

        # Strings are hardest to deal with because they're also containers,
        # so just get that out of the way first
        if isinstance(next_part, str):
            render_frame.part_index += 1
            to_join.append(next_part)

        elif isinstance(next_part, InterpolatedVariable):
            render_frame.part_index += 1

            raw_val = render_frame.provenance.bind_variable(
                next_part.name,
                template_preload=template_preload,
                error_collector=error_collector)
            transformer = getattr(
                render_frame.transformers, next_part.name, None)
            if transformer is None:
                transformed = raw_val
            else:
                transformed = cast(str | None, transformer(raw_val))

            if transformed is not None:
                unescaped_val = _apply_format(
                    transformed,
                    next_part.config)
                escaped_val = render_frame.parse_config.variable_escaper(
                    unescaped_val)
                # Note that variable interpolations don't support affixes!
                to_join.append(escaped_val)


        elif isinstance(next_part, InterpolatedContent):
            render_frame.part_index += 1

            val_from_params = render_frame.provenance.bind_content(
                next_part.name,
                template_preload=template_preload,
                error_collector=error_collector)

            if isinstance(val_from_params, ComplexContent):
                # Note that _render_complex_content has its own try/catch!
                to_join.extend(_render_complex_content(
                    val_from_params,
                    render_ctx,
                    render_frame.provenance,
                    render_frame.parse_config,
                    next_part.config,
                    render_frame.transformers))

            else:
                transformer = getattr(
                    render_frame.transformers, next_part.name, None)
                if transformer is None:
                    transformed = val_from_params
                else:
                    transformed = cast(
                        str | None, transformer(val_from_params))

                # As usual, values of None are omitted
                if transformed is not None:
                    formatted_val = _apply_format(
                        transformed,
                        next_part.config)
                    render_frame.parse_config.content_verifier(formatted_val)
                    to_join.extend(
                        next_part.config.apply_affix(formatted_val))

        # Slots get a little bit more complicated, but the general idea is
        # to append a stack frame for each depth level. We maintain the
        # state of which instance we're on within the stack frame.
        elif isinstance(next_part, InterpolatedSlot):
            slot_instance_index = render_frame.slot_instance_index
            if slot_instance_index == 0:
                # Note that this needs a dedicated try/catch so that the
                # part_index handling logic remains outside of the failure case
                try:
                    slot_instances = getattr(
                        render_frame.instance, next_part.name)
                    slot_instance_count = len(slot_instances)
                # Note: this would happen if the getattr fails, for example
                # because the wrong type was passed for the slot instance.
                except Exception as exc:
                    error_collector.append(exc)
                    render_frame.part_index += 1
                    continue

                if slot_instance_count > 0:
                    render_frame.slot_instance_count = slot_instance_count
                    render_frame.slot_instances = slot_instances
                # Note that this is also important to skip prefix/suffix.
                else:
                    # Critical: this must stay outside exception handling!
                    render_frame.part_index += 1
                    continue

            else:
                # Note that we skip this entirely if the slot instance
                # count is zero, so by definition, if we hit this branch,
                # we need a suffix.
                to_join.extend(next_part.config.apply_suffix_iter())

                slot_instance_count = render_frame.slot_instance_count
                # We've exhausted the instances; reset the state for the
                # next slot.
                if render_frame.slot_instance_index >= slot_instance_count:
                    render_frame.slot_instance_index = 0
                    render_frame.slot_instance_count = 0
                    # Note: we're deliberately skipping the slot instances,
                    # because it'll just get overwritten the next time
                    # around, so this is faster (though it doesn't free
                    # memory as quickly)
                    render_frame.part_index += 1
                    continue

                slot_instances = render_frame.slot_instances

            # Remember: we skip this entirely if the slot instance count
            # is zero.
            to_join.extend(next_part.config.apply_prefix_iter())
            slot_instance = slot_instances[slot_instance_index]
            # Note that this needs to support both union slot
            # types, and (eventually) dynamic slot types, hence
            # doing this on every iteration instead of
            # precalculating it for the whole interpolated slot
            slot_instance_preload = template_preload[type(slot_instance)]
            slot_instance_signature = cast(
                TemplateIntersectable, slot_instance)._templatey_signature
            render_stack.append(
                _RenderStackFrame(
                    instance=slot_instance,
                    parts=slot_instance_preload.parts,
                    part_count=slot_instance_preload.part_count,
                    parse_config=slot_instance_signature.parse_config,
                    signature=slot_instance_signature,
                    provenance=render_frame.provenance.with_appended(
                        ProvenanceNode(
                            encloser_slot_key=next_part.name,
                            encloser_slot_index=slot_instance_index,
                            instance_id=id(slot_instance),
                            instance=slot_instance)),
                    transformers=
                        slot_instance_signature.fieldset.transformers))

            # It's critical that this is outside the exception catching, so
            # that the current frame doesn't get stuck in an infinite loop on
            # this slot instance.
            render_frame.slot_instance_index += 1

        elif isinstance(next_part, InterpolatedFunctionCall):
            render_frame.part_index += 1

            pprint(function_precall)
            execution_result = function_precall[
                get_precall_cache_key(render_frame.provenance, next_part)]
            nested_render_node = _build_render_frame_for_func_result(
                render_frame.instance,
                render_frame.provenance,
                next_part,
                execution_result,
                render_frame.parse_config,
                error_collector)
            if nested_render_node is not None:
                render_stack.append(nested_render_node)

        # Similar to slots, but different enough that it warrants a
        # separate approach. Trust me, I tried to unify them, and 1. I
        # never got it fully working, 2. it was a pretty big hack (a
        # virtual slot mechanism with some __getattr__ shenanigans), and
        # 3. it created way more problems than it was worth.
        # More info in the docstring for _InjectedInstanceContainer.
        elif isinstance(next_part, _InjectedInstanceContainer):
            render_frame.part_index += 1
            injected_instance = next_part.instance

            # Note that this needs to support both union slot
            # types, and dynamic slot types, hence
            # doing this on every iteration instead of
            # precalculating it for the whole interpolated slot
            injected_instance_preload = render_ctx.template_preload[
                type(injected_instance)]
            injected_instance_signature = cast(
                TemplateIntersectable, injected_instance
            )._templatey_signature
            render_stack.append(
                _RenderStackFrame(
                    instance=injected_instance,
                    parts=injected_instance_preload.parts,
                    part_count=injected_instance_preload.part_count,
                    parse_config=injected_instance_signature.parse_config,
                    signature=injected_instance_signature,
                    # Note that the correct ``from_injection`` value is
                    # added when creating the current stack frame.
                    provenance=render_frame.provenance.with_appended(
                        ProvenanceNode(
                            encloser_slot_key='',
                            encloser_slot_index=-1,
                            instance_id=id(injected_instance),
                            instance=injected_instance),),
                    transformers=
                        injected_instance_signature.fieldset.transformers))

        else:
            raise TypeError(
                'Templatey internal error: unknown template part!',
                next_part)

    return to_join


@dataclass(slots=True)
class _InjectedInstanceContainer:
    """When env functions inject instances of templates, the render
    stack gets into a bit of a weird state. We want to avoid needing to
    extend the stack by one for each and every injected template and/or
    stringified value that the function returned. But we can't modify
    the existing parts of a frame; first, it's a sometimes (always?) a
    tuple, but more importantly, it would screw up our entire indexing
    mechanism.

    In theory we could add a "pending injections" sub-stack or something
    to the _RenderStackFrame, but then we'd be stuck checking for that
    during every single iteration of the render driver loop.

    So instead, we do this: wrap the instance into a container, and
    then (as the LAST, and therefore SLOWEST, check in the render driver
    loop) we can check for the container, and have special logic for
    handling them there.
    """
    instance: TemplateParamsInstance


@dataclass(slots=True)
class _RenderStackFrame:
    """
    """
    instance: TemplateParamsInstance
    parse_config: TemplateConfig
    signature: TemplateSignature
    provenance: Provenance
    transformers: NamedTuple
    parts: Sequence[
        str
        | InterpolatedSlot
        | InterpolatedContent
        | InterpolatedVariable
        | InterpolatedFunctionCall
        | _InjectedInstanceContainer]

    _: KW_ONLY
    part_count: int
    part_index: int = field(default=0, init=False)
    slot_instance_count: int = field(default=0, init=False)
    slot_instance_index: int = field(default=0, init=False)
    slot_instances: Sequence[TemplateParamsInstance] = field(init=False)

    @property
    def exhausted(self) -> bool:
        # Note that the slot index exhaustion is handled separately, by
        # controlling exactly when the part index is incremented
        return self.part_index >= self.part_count


type TemplateInjection = tuple[Provenance | None, TemplateParamsInstance]


@dataclass(slots=True)
class RenderContext:
    """
    """
    template_preload: dict[TemplateClass, ParsedTemplateResource]
    function_precall: dict[PrecallCacheKey, FuncExecutionResult]
    error_collector: ErrorCollector

    def prep_render(
            self,
            root_template_instance: TemplateParamsInstance,
            error_collector: ErrorCollector,
            ) -> Iterable[RenderEnvRequest]:
        """For the passed root template, populates the template_preload
        and function_precall until either all resources have been
        prepared, or it needs help from the render environment.
        """
        template_preload = self.template_preload
        function_precall = self.function_precall

        local_root_instances: list[TemplateInjection] = [
            (None, root_template_instance)]
        to_execute: list[FuncExecutionRequest] = []

        while local_root_instances or to_execute:
            # We need to make a copy of this here so that we can mutate the
            # local_root_instances later (to determine the next batch) without
            # affecting the processing of the current one
            previous_local_roots = tuple(local_root_instances)
            to_load: set[TemplateClass] = {
                type(injection[1]) for injection in local_root_instances}
            new_injections: list[TemplateInjection] = []

            # This strips anything we've already loaded -- including any
            # nested classes -- to avoid extra effort from the render env.
            to_load.difference_update(template_preload)

            # This might seem redundant with the toplevel ``while ...:``,
            # but we might find ourselves in an edge case where the local root
            # is a dynamic template class that was already part of the preload,
            # and it had no executions, but we still need to explore its
            # preload tree for any downstream dependencies.
            if to_load or to_execute:
                yield RenderEnvRequest(
                    to_load=to_load,
                    to_execute=to_execute,
                    error_collector=self.error_collector,
                    injections=new_injections,
                    template_preload=template_preload,
                    function_precall=function_precall)
            # We need to reset these in advance of prepping the next round,
            # otherwise we'll mix up the current round and the next
            local_root_instances.clear()
            to_execute.clear()

            # Status check: we can now say for sure that the preload is
            # complete for all included classes so far, and that the prerender
            # tree has been populated as well. So now we need to check the
            # prerender tree on each root instance for any downstream dynamic
            # templates or function calls.
            for local_provenance, local_instance in previous_local_roots:
                prerender_tree = cast(
                    TemplateIntersectable, local_instance
                )._templatey_signature.prerender_tree

                # Note: a value of None is not an error! We use it to indicate
                # that the template class has no dynamic slots nor function
                # invocations.
                if prerender_tree is not None:
                    # Note that this operates in-place, extending the two
                    # control lists as needed.
                    prerender_tree.extract(
                        from_instance=local_instance,
                        from_injection=local_provenance,
                        into_injection_backlog=local_root_instances,
                        into_precall_backlog=to_execute,
                        template_preload=template_preload,
                        error_collector=error_collector)

            # Status check: we can now say for sure that all of the precalls
            # from the previous batches are complete, and we have a complete
            # record of all injected instances in ``injections``.
            local_root_instances.extend(new_injections)


type _PrecallExecutionRequest = tuple[
    Provenance, InterpolatedFunctionCall]
type PrecallCacheKey = Hashable


def get_precall_cache_key(
        provenance: Provenance,
        interpolated_call: InterpolatedFunctionCall
        ) -> PrecallCacheKey:
    """For a particular template instance and interpolated function
    call, creates the hashable cache key to be used for the render
    context.
    """
    # Note that template provenance includes the actual current template
    # instance, so by definition, this encodes exactly which function call
    # is being referenced
    return (provenance, interpolated_call)


def _render_complex_content(
        complex_content: ComplexContent,
        render_ctx: RenderContext,
        provenance: Provenance,
        parse_config: TemplateConfig,
        interpolation_config: InterpolationConfig,
        transformers: NamedTuple,
        ) -> Iterable[str]:
    try:
        extracted_vars = {
            key: provenance.bind_variable(
                key,
                template_preload=render_ctx.template_preload,
                error_collector=render_ctx.error_collector)
            for key in complex_content.dependencies}
        extracted_transformers = {
            key: getattr(transformers, key, None)
            for key in complex_content.dependencies}

        for content_segment in complex_content.flatten(
            extracted_vars, interpolation_config, extracted_transformers
        ):
            if isinstance(content_segment, InjectedValue):
                raw_val = content_segment.value
                if raw_val is None:
                    continue

                unescaped_val = _apply_format(raw_val, content_segment.config)

                if content_segment.use_variable_escaper:
                    escaped_val = parse_config.variable_escaper(
                        unescaped_val)
                else:
                    escaped_val = unescaped_val

                if content_segment.use_content_verifier:
                    parse_config.content_verifier(escaped_val)

                yield escaped_val

            # Note: as usual, None values get omitted!
            elif content_segment is not None:
                formatted_val = _apply_format(
                    content_segment, interpolation_config)
                parse_config.content_verifier(formatted_val)
                yield formatted_val

    except Exception as exc:
        exc.add_note('Failed to render complex content!')
        render_ctx.error_collector.append(exc)


def _build_render_frame_for_func_result(  # noqa: C901
        enclosing_instance: TemplateParamsInstance,
        enclosing_provenance: Provenance,
        abstract_call: InterpolatedFunctionCall,
        execution_result: FuncExecutionResult,
        template_config: TemplateConfig,
        error_collector: ErrorCollector
        ) -> _RenderStackFrame | None:
    """This constructs a _RenderNode for the given execution result and
    returns it (or None, if there was an error).
    """
    injected_templates: list[tuple[int, TemplateParamsInstance]] = []
    resulting_parts: list[str | _InjectedInstanceContainer] = []
    if execution_result.exc is None:
        if execution_result.retval is None:
            raise TypeError(
                'Impossible branch! Malformed func exe result',
                execution_result)

        for index, result_part in enumerate(execution_result.retval):
            if isinstance(result_part, str):
                resulting_parts.append(
                    template_config.variable_escaper(result_part))
            elif isinstance(result_part, InjectedValue):
                resulting_parts.append(
                    _coerce_injected_value(result_part, template_config))
            elif is_template_instance_xable(result_part):
                injected_templates.append((index, result_part))
                # This is just a placeholder; it gets overwritten in
                # _build_render_stack_extension
                resulting_parts.append('')
            else:
                error_collector.append(capture_traceback(
                    TypeError(
                        'Invalid return from env function!',
                        execution_result, result_part)))

    else:
        if execution_result.retval is not None:
            raise TypeError(
                'Impossible branch! Malformed func exe result',
                execution_result)

        error_collector.append(capture_traceback(
            TemplateFunctionFailure('Env function raised!'),
            from_exc=execution_result.exc))

    empty_template_signature = EMPTY_TEMPLATE_XABLE._templatey_signature
    if injected_templates:
        for index, template_instance in injected_templates:
            resulting_parts[index] = _InjectedInstanceContainer(
                template_instance)

        return _RenderStackFrame(
            parts=resulting_parts,
            part_count=len(resulting_parts),
            parse_config=empty_template_signature.parse_config,
            signature=empty_template_signature,
            # Note: keep this empty here, because we need the instance info
            # to match the injected template, and the whole idea here is to
            # avoid a bunch of extraneous stack frames. We'll add in the
            # correct initial node in the render driver code, where we deal
            # with _InjectedInstanceContainers
            provenance=Provenance(from_injection=enclosing_provenance),
            instance=EMPTY_TEMPLATE_INSTANCE,
            transformers=empty_template_signature.fieldset.transformers)

    elif resulting_parts:
        return _RenderStackFrame(
            parts=resulting_parts,
            part_count=len(resulting_parts),
            parse_config=empty_template_signature.parse_config,
            signature=empty_template_signature,
            provenance=Provenance(),
            instance=EMPTY_TEMPLATE_INSTANCE,
            transformers=empty_template_signature.fieldset.transformers)


def _apply_format(raw_value, config: InterpolationConfig) -> str:
    """For both interpolated variables and injected values, we allow
    format specs and conversions to be supplied. We need to actually
    apply these, but the stdlib doesn't really give us a good way of
    doing that. So this is how we do that instead.
    """
    # hot path go fast
    if config is None or config.fmt is None:
        # Note: yes, strings can be formatted with eg padding, but we literally
        # just checked to make sure that there was no format spec, so format
        # would have nothing to do here!
        if isinstance(raw_value, str):
            formatted_value = raw_value
        else:
            formatted_value = format(raw_value)

    else:
        formatted_value = format(raw_value, config.fmt)

    return formatted_value


def _coerce_injected_value(
        injected_value: InjectedValue,
        template_config: TemplateConfig
        ) -> str:
    """InjectedValue instances are used within the return value of
    environment functions and complex content to indicate that the
    result should be sourced from the variables and/or the content of
    the current render call. This function is responsible for converting
    the ``InjectedValue`` instance into the final resulting string to
    render.
    """
    unescaped_value = _apply_format(
        injected_value.value,
        injected_value.config)

    if injected_value.use_variable_escaper:
        escapish_value = template_config.variable_escaper(unescaped_value)
    else:
        escapish_value = unescaped_value

    if injected_value.use_content_verifier:
        template_config.content_verifier(escapish_value)

    return escapish_value
