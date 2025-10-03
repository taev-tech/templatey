import inspect
import typing
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import cast
from typing import runtime_checkable

try:
    import anyio
    from anyio import create_task_group
except ImportError:
    if typing.TYPE_CHECKING:
        import anyio
        from anyio import create_task_group
from docnote import Note

from templatey._bootstrapping import PARSED_EMPTY_TEMPLATE
from templatey._bootstrapping import EmptyTemplate
from templatey._error_collector import ErrorCollector
from templatey._finalizers import ensure_prerender_tree
from templatey._finalizers import ensure_recursive_totality
from templatey._finalizers import ensure_slot_tree
from templatey._renderer import FuncExecutionRequest
from templatey._renderer import FuncExecutionResult
from templatey._renderer import PrecallCacheKey
from templatey._renderer import RenderContext
from templatey._renderer import TemplateInjection
from templatey._renderer import render_driver
from templatey._signature import TemplateSignature
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import is_template_instance
from templatey.exceptions import MismatchedRenderColor
from templatey.exceptions import MismatchedTemplateEnvironment
from templatey.exceptions import MismatchedTemplateSignature
from templatey.exceptions import TemplateyException
from templatey.parser import ParsedTemplateResource
from templatey.parser import parse
from templatey.templates import InjectedValue

# Note: strings here will be escaped. InjectedValues may decide whether or not
# escaping should be applied. Nested templates will not be escaped.
EnvFunction = Callable[
    ..., Sequence[str | TemplateParamsInstance | InjectedValue]]
EnvFunctionAsync = Callable[
    ..., Awaitable[Sequence[str | TemplateParamsInstance | InjectedValue]]]


@dataclass(frozen=True, slots=True)
class _TemplateFunctionContainer[F: EnvFunction | EnvFunctionAsync]:
    """
    """
    name: str
    function: F
    signature: inspect.Signature
    is_async: bool


@runtime_checkable
class SyncTemplateLoader[L: object](Protocol):

    def load_sync(
            self,
            template: type[TemplateParamsInstance],
            template_resource_locator: L
            ) -> str:
        """This is responsible for loading the actual template text,
        based on the passed resource locator.
        """
        ...


@runtime_checkable
class AsyncTemplateLoader[L: object](Protocol):

    async def load_async(
            self,
            template: type[TemplateParamsInstance],
            template_resource_locator: L
            ) -> str:
        """This is responsible for loading the actual template text,
        based on the passed resource locator.
        """
        ...


class RenderEnvironment:
    _parsed_template_cache: dict[
        type[TemplateParamsInstance], ParsedTemplateResource]
    _env_functions: dict[str, _TemplateFunctionContainer]
    # We use this to prevent registering template functions after any calls
    # to load() have been made, because it would result in different template
    # functions per template
    _has_loaded_any_template: bool
    _template_loader: SyncTemplateLoader | AsyncTemplateLoader
    strict_interpolation_validation: bool

    def __init__(
            self,
            template_loader: SyncTemplateLoader | AsyncTemplateLoader,
            env_functions:
                Optional[Iterable[EnvFunction | EnvFunctionAsync]] = None,
            # If True, this will make sure that the template interface exactly
            # matches the template text. If False, this will just make sure
            # that the template interface is at least sufficient for the
            # template text.
            strict_interpolation_validation: bool = True):
        self.strict_interpolation_validation = strict_interpolation_validation
        self._has_loaded_any_template = False

        self._env_functions = {}
        if env_functions is not None:
            for function in env_functions:
                self.register_env_function(function)

        self._can_load_sync = isinstance(template_loader, SyncTemplateLoader)
        self._can_load_async = isinstance(template_loader, AsyncTemplateLoader)
        self._template_loader = template_loader
        self._parsed_template_cache = {}

    def register_env_function(
            self,
            env_function: EnvFunction | EnvFunctionAsync,
            *,
            force_async: bool = False,
            with_name: str | None = None):
        """Manually register an environment function with the render
        environment, instead of passing it in to the environment
        constructor.

        This can be used to force a function to be registerd as async,
        in case it was not inferred as such, by passing
        ``force_async=True``.

        Normally, registered functions are assigned their __name__ as
        the function name; manual registration can also be used to
        override this behavior via the ``with_name`` parameter.
        """
        if self._has_loaded_any_template:
            raise TemplateyException(
                'To prevent having different template functions per template, '
                + 'you cannot register new template functions after loading '
                + 'any templates in an environment.')

        if with_name is None:
            function_name = env_function.__name__
        else:
            function_name = with_name

        self._env_functions[function_name] = _TemplateFunctionContainer(
            name=function_name,
            function=env_function,
            signature=inspect.signature(env_function),
            is_async=force_async or _infer_asyncness(env_function))

    async def load_async(
            self,
            template: type[TemplateParamsInstance],
            *,
            override_validation_strictness: None | bool = None,
            force_reload: bool = False,
            preload: Annotated[
                    dict[TemplateClass, ParsedTemplateResource] | None,
                    Note('''If desired, pass ``preload`` to recover all of the
                        parsed template resources, including dependencies.

                        If omitted, the dependent resources will still be
                        loaded and cached, but there won't be any guarantees
                        against them subsequently being evicted before the
                        call to load returns.''')
                ] = None
            ) -> ParsedTemplateResource:
        """Loads a template resource from a TemplateParamsInstance
        class. Caches it within the environment and returns the
        resulting ParsedTemplateResource.

        If force_reload is True, bypasses the cache.

        Note that this will also load any and all other templates that
        might be required (via nested slots) to render the passed
        template. These will be cached, but not directly returned.
        """
        # Note: doing this first for thread safety in the sync case, preventing
        # any new template functions from being registered.
        self._has_loaded_any_template = True
        # Note: in this case, we can bypass iterally everything.
        if template is EmptyTemplate:
            await anyio.sleep(0)
            return PARSED_EMPTY_TEMPLATE

        template_xable = cast(type[TemplateIntersectable], template)
        signature = template_xable._templatey_signature
        template_loader = self._get_loader(
            signature,
            self._can_load_async,
            AsyncTemplateLoader)

        # As the name suggests, this also recursively finalizes all inclusions.
        ensure_recursive_totality(signature, template)

        # Note that, because we need to potentially populate a preload, and
        # not just the cache, we have to do this every time, even if the
        # desired root template is already in the cache (because its
        # requirements might not be).
        # Three things:
        # 1. we're prioritizing the case where preload is set, because that
        #    happens during render calls, which need to be fast.
        # 2. we're assuming that the total inclusions are significantly
        #    smaller than the total template cache.
        # 3. this isn't, strictly speaking, threadsafe, so if and when we add
        #    support for finite caches, we'll need to wrap this into a lock.
        #    Or something. It's not clear how this would work if someone wants
        #    both sync and async loading at the same time.
        required_loads: set[TemplateClass] = set()
        target_resource: ParsedTemplateResource | None = None
        if preload is None:
            preload = {}

        # This is cleaner than a difference, since we need to handle both the
        # excluded and included cases if we have a preload
        for included_template_cls in signature.total_inclusions:
            if force_reload:
                required_loads.add(included_template_cls)
            elif (
                from_cache := self._parsed_template_cache.get(
                    included_template_cls)
            ) is not None:
                preload[included_template_cls] = from_cache
                if included_template_cls is template:
                    target_resource = from_cache
            else:
                required_loads.add(included_template_cls)

        # We're doing this just in case everything was cached, and we never
        # await a single loader.
        await anyio.sleep(0)
        for required_template_cls in required_loads:
            requirement_signature = cast(
                type[TemplateIntersectable], required_template_cls
            )._templatey_signature

            requirement_text = await template_loader.load_async(
                required_template_cls,
                requirement_signature.resource_locator)
            parsed_requirement_template = self._parse_and_cache(
                required_template_cls,
                requirement_signature,
                requirement_text,
                override_validation_strictness)

            preload[required_template_cls] = parsed_requirement_template
            if required_template_cls is template:
                target_resource = parsed_requirement_template

            # Note that we have to do this on all of the requirements;
            # otherwise, they'll be stored in the cache with an incomplete
            # signature.
            ensure_slot_tree(requirement_signature, required_template_cls)
            ensure_prerender_tree(requirement_signature, preload)

        if target_resource is None:
            raise RuntimeError(
                'Impossible branch: target template missing from load results',
                template)

        return target_resource

    def load_sync(
            self,
            template: type[TemplateParamsInstance],
            *,
            override_validation_strictness: None | bool = None,
            force_reload: bool = False,
            preload: Annotated[
                    dict[TemplateClass, ParsedTemplateResource] | None,
                    Note('''If desired, pass ``preload`` to recover all of the
                        parsed template resources, including dependencies.

                        If omitted, the dependent resources will still be
                        loaded and cached, but there won't be any guarantees
                        against them subsequently being evicted before the
                        call to load returns.''')
                ] = None
            ) -> ParsedTemplateResource:
        """Loads a template resource from a TemplateParamsInstance
        class. Caches it within the environment and returns the
        resulting ParsedTemplateResource.

        If force_reload is True, bypasses the cache.

        Note that this will also load any and all other templates that
        might be required (via nested slots) to render the passed
        template. These will be cached, but not directly returned.

        Note that cache operations here aren't threadsafe, but as long
        as the underlying resource doesn't change, the worst case is
        that we load and parse the same template resource multiple times
        without calling force_reload. This is probably better than
        wrapping the whole thing in a lock.

        Also note that we're prioritizing speed here in the cached happy
        case over technical correctness w.r.t. always raising if we
        don't support this loading flavor.
        """
        # Note: doing this first for thread safety in the sync case, preventing
        # any new template functions from being registered.
        self._has_loaded_any_template = True
        # Note: in this case, we can bypass iterally everything.
        if template is EmptyTemplate:
            return PARSED_EMPTY_TEMPLATE

        template_xable = cast(type[TemplateIntersectable], template)
        signature = template_xable._templatey_signature
        template_loader = self._get_loader(
            signature,
            self._can_load_sync,
            SyncTemplateLoader)

        # As the name suggests, this also recursively finalizes all inclusions.
        ensure_recursive_totality(signature, template)

        # Note that, because we need to potentially populate a preload, and
        # not just the cache, we have to do this every time, even if the
        # desired root template is already in the cache (because its
        # requirements might not be).
        # Three things:
        # 1. we're prioritizing the case where preload is set, because that
        #    happens during render calls, which need to be fast.
        # 2. we're assuming that the total inclusions are significantly
        #    smaller than the total template cache.
        # 3. this isn't, strictly speaking, threadsafe, so if and when we add
        #    support for finite caches, we'll need to wrap this into a lock.
        #    Or something. It's not clear how this would work if someone wants
        #    both sync and async loading at the same time.
        required_loads: set[TemplateClass] = set()
        target_resource: ParsedTemplateResource | None = None
        # Note that we need a preload to construct prerender trees regardless
        # of whether or not the caller cares about it.
        if preload is None:
            preload = {}

        # This is cleaner than a difference, since we need to handle both the
        # excluded and included cases if we have a preload
        for included_template_cls in signature.total_inclusions:
            if force_reload:
                required_loads.add(included_template_cls)
            elif (
                from_cache := self._parsed_template_cache.get(
                    included_template_cls)
            ) is not None:
                preload[included_template_cls] = from_cache
                if included_template_cls is template:
                    target_resource = from_cache
            else:
                required_loads.add(included_template_cls)

        for required_template_cls in required_loads:
            requirement_signature = cast(
                type[TemplateIntersectable], required_template_cls
            )._templatey_signature
            requirement_text = template_loader.load_sync(
                required_template_cls,
                requirement_signature.resource_locator)
            parsed_requirement_template = self._parse_and_cache(
                required_template_cls,
                requirement_signature,
                requirement_text,
                override_validation_strictness)

            preload[required_template_cls] = parsed_requirement_template
            if required_template_cls is template:
                target_resource = parsed_requirement_template

            # Note that we have to do this on all of the requirements;
            # otherwise, they'll be stored in the cache with an incomplete
            # signature.
            ensure_slot_tree(requirement_signature, required_template_cls)
            ensure_prerender_tree(requirement_signature, preload)

        if target_resource is None:
            raise RuntimeError(
                'Impossible branch: target template missing from load results',
                template)
        return target_resource

    def _get_loader[T: AsyncTemplateLoader | SyncTemplateLoader](
            self,
            signature: TemplateSignature,
            can_load_flavor: bool,
            flavor_cls: type[T]
            ) -> T:
        explicit_loader = signature.explicit_loader
        if explicit_loader is None:
            if not can_load_flavor:
                raise TypeError(
                    'Environment template loader does not support current '
                    + 'loading flavor (sync/async)', self._template_loader)

            # The cast here is because it could be a sync and/or async loader,
            # and the type system doesn't know we already verified that via
            # _can_load_<sync|async>.
            template_loader = cast(T, self._template_loader)
        else:
            if not isinstance(explicit_loader, flavor_cls):
                raise TypeError(
                    'Explicit template loader does not support current '
                    + 'loading flavor (sync/async)', explicit_loader)
            template_loader = explicit_loader

        return template_loader

    def _parse_and_cache(
            self,
            template_cls: TemplateClass,
            signature: TemplateSignature,
            template_text: str,
            override_validation_strictness: None | bool
            ) -> ParsedTemplateResource:
        parsed_template_resource = parse(
            template_text,
            signature.parse_config.interpolator,
            signature.segment_modifiers)

        if override_validation_strictness is None:
            strict_mode = self.strict_interpolation_validation
        else:
            strict_mode = override_validation_strictness

        self._validate_env_functions(
            template_cls, parsed_template_resource)
        self._validate_template_signature(
            template_cls, signature, parsed_template_resource,
            strict_mode=strict_mode)

        self._parsed_template_cache[template_cls] = parsed_template_resource
        return parsed_template_resource

    def _validate_env_functions(
            self,
            template_class: TemplateClass,
            parsed_template_resource: ParsedTemplateResource
            ) -> Literal[True]:
        """Makes sure that the template environment contains all of the
        template functions referenced in the template text. Returns
        True or raises MismatchedTemplateEnvironment.

        Note that we never use strict mode here, because it would be
        silly to require every single template to call every single
        template function. That's simply not what they're meant to be
        used for!
        """
        # Interestingly, .difference() works here, but plain ``-`` doesn't
        function_mismatch = parsed_template_resource.function_names.difference(
            self._env_functions)
        if function_mismatch:
            raise MismatchedTemplateEnvironment(
                'Template environment functions did not match the template '
                + 'text!', template_class, function_mismatch)

        for (
            function_name, function_calls
        ) in parsed_template_resource.function_calls.items():
            function_container = self._env_functions[function_name]
            for function_call in function_calls:
                try:
                    function_container.signature.bind(
                        *function_call.call_args,
                        **function_call.call_kwargs)
                except TypeError as exc:
                    raise MismatchedTemplateEnvironment(
                        'Template environment function had invalid call '
                        + 'signature', template_class, function_call
                    ) from exc

        return True

    def _validate_template_signature(
            self,
            template_class: TemplateClass,
            signature: TemplateSignature,
            parsed_template_resource: ParsedTemplateResource,
            *,
            strict_mode: bool
            ) -> Literal[True]:
        """Makes sure that the template signature includes all of the
        names referenced in the template text. Returns True or
        raises MismatchedTemplateSignature.
        """
        fieldset = signature.fieldset
        variable_names = fieldset.var_names
        slot_names = fieldset.slot_names
        dynamic_class_slot_names = fieldset.dynamic_class_slot_names
        content_names = fieldset.content_names
        data_names = fieldset.data_names

        if strict_mode:
            variables_mismatch = (
                parsed_template_resource.variable_names ^ variable_names)
            slot_mismatch = (
                parsed_template_resource.slot_names
                ^ (slot_names | dynamic_class_slot_names))
            content_mismatch = (
                parsed_template_resource.content_names ^ content_names)
            data_mismatch = (
                parsed_template_resource.data_names ^ data_names)

        else:
            variables_mismatch = (
                parsed_template_resource.variable_names - variable_names)
            slot_mismatch = (
                parsed_template_resource.slot_names
                - (slot_names | dynamic_class_slot_names))
            content_mismatch = (
                parsed_template_resource.content_names - content_names)
            data_mismatch = (
                parsed_template_resource.data_names - data_names)

        if (
            variables_mismatch
            or slot_mismatch
            or content_mismatch
            or data_mismatch
        ):
            raise MismatchedTemplateSignature(
                'Template interface variables, content, or slots did not '
                + 'match the template text!', template_class,
                variables_mismatch, slot_mismatch, content_mismatch,
                data_mismatch)

        return True

    def render_sync(
            self,
            template_instance: TemplateParamsInstance
            ) -> str:
        error_collector = ErrorCollector()
        template_preload: dict[TemplateClass, ParsedTemplateResource] = {}
        function_precall: dict[PrecallCacheKey, FuncExecutionResult] = {}

        render_ctx = RenderContext(
            template_preload=template_preload,
            function_precall=function_precall,
            error_collector=error_collector)
        for env_request in render_ctx.prep_render(
            template_instance,
            error_collector
        ):
            for root_to_load in env_request.to_load:
                # Note that this will already set the root_to_load within the
                # preload dict.
                self.load_sync(root_to_load, preload=template_preload)

            for to_execute in env_request.to_execute:
                self._execute_env_function_sync(
                    to_execute,
                    env_request.injections,
                    function_precall)

        to_join = render_driver(template_instance, render_ctx)
        if error_collector:
            raise ExceptionGroup('Failed to render template', error_collector)

        return ''.join(to_join)

    def _execute_env_function_sync(
            self,
            request: FuncExecutionRequest,
            injections: list[TemplateInjection],
            precall: dict[PrecallCacheKey, FuncExecutionResult]
            ) -> None:
        try:
            container = self._env_functions[request.name]
            if container.is_async:
                raise MismatchedRenderColor(
                    'Async env funcs cannot be used within render_sync!')

            exe_result = container.function(*request.args, **request.kwargs)
            precall[request.result_key] = FuncExecutionResult(
                name=request.name,
                retval=exe_result,
                exc=None)

            for result_segment in exe_result:
                if is_template_instance(result_segment):
                    injections.append((request.provenance, result_segment))

        except Exception as exc:
            precall[request.result_key] = FuncExecutionResult(
                name=request.name,
                retval=None,
                exc=exc)

    async def render_async(
            self,
            template_instance: TemplateParamsInstance
            ) -> str:
        error_collector = ErrorCollector()
        template_preload: dict[TemplateClass, ParsedTemplateResource] = {}
        function_precall: dict[PrecallCacheKey, FuncExecutionResult] = {}

        render_ctx = RenderContext(
            template_preload=template_preload,
            function_precall=function_precall,
            error_collector=error_collector)

        for env_request in render_ctx.prep_render(
            template_instance,
            error_collector
        ):
            async with create_task_group() as task_group:
                for root_to_load in env_request.to_load:
                    # Note that this will already set the root_to_load within
                    # the preload dict.
                    task_group.start_soon(partial(
                        self.load_async,
                        root_to_load,
                        preload=template_preload))

                for to_execute in env_request.to_execute:
                    task_group.start_soon(
                        self._execute_env_function_async,
                        to_execute,
                        env_request.injections,
                        function_precall)

        to_join = render_driver(template_instance, render_ctx)
        if error_collector:
            raise ExceptionGroup('Failed to render template', error_collector)

        return ''.join(to_join)

    async def _execute_env_function_async(
            self,
            request: FuncExecutionRequest,
            injections: list[TemplateInjection],
            precall: dict[PrecallCacheKey, FuncExecutionResult]
            ) -> None:
        try:
            container = self._env_functions[request.name]
            if container.is_async:
                exe_result = await container.function(
                    *request.args, **request.kwargs)
            else:
                exe_result = container.function(
                    *request.args, **request.kwargs)

            for result_segment in exe_result:
                if is_template_instance(result_segment):
                    injections.append((request.provenance, result_segment))

            precall[request.result_key] = FuncExecutionResult(
                name=request.name,
                retval=exe_result,
                exc=None)

        except Exception as exc:
            precall[request.result_key] = FuncExecutionResult(
                name=request.name,
                retval=None,
                exc=exc)


def _infer_asyncness(env_function: EnvFunction | EnvFunctionAsync) -> bool:
    """Infers, based on the type of the function, whether it should be
    considered sync or async.
    """
    return (
        inspect.iscoroutinefunction(env_function)
        or inspect.isawaitable(env_function))
