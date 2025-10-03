from __future__ import annotations

import functools
import inspect
import logging
import sys
import typing
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import _MISSING_TYPE
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from types import FrameType
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TypedDict
from typing import Unpack
from typing import dataclass_transform
from typing import overload

from docnote import DocnoteConfig
from docnote import Note
from docnote import docnote

from templatey._signature import TemplateSignature
from templatey._types import TemplateClass
from templatey.interpolators import NamedInterpolator
from templatey.modifiers import SegmentModifier
from templatey.parser import InterpolationConfig

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from templatey.environments import AsyncTemplateLoader
    from templatey.environments import SyncTemplateLoader

_CLOSURE_ANCHORS: ContextVar[dict[TemplateClass, FrameType]] = ContextVar(
    '_CLOSURE_ANCHORS')


class VariableEscaper(Protocol):

    def __call__(self, value: str) -> str:
        """Variable escaper functions accept a single positional
        argument: the value of the variable to escape. It then does any
        required escaping and returns the final string.
        """
        ...


class ContentVerifier(Protocol):

    def __call__(self, value: str) -> Literal[True]:
        """Content verifier functions accept a single positional
        argument: the value of the content to verify. It does any
        verification, and then returns True if the content was okay,
        or raises BlockedContentValue if the content was not acceptable.

        Note that we raise instead of trying to escape for two reasons:
        1.. We don't really know what to replace it with. This is also
            true with variables, but:
        2.. We expect that content is coming from -- if not trusted,
            then at least authoritative -- sources, and therefore, we
            should fail loudly, because it gives the author a chance to
            correct the problem before it becomes user-facing.
        """
        ...


class InterpolationTransformer[T](Protocol):

    def __call__(
            self,
            value: Annotated[
                T,
                Note(
                    '''The value of the variable or content. A value of
                    ``None`` indicates that the value is intended to be
                    omitted, but you may still provide a fallback
                    instead.
                    ''')]
            ) -> str | None:
        """Interpolation transformers give you a chance to modify the
        rendered result of a particular content or variable value, omit
        it entirely, or provide a fallback for missing values.

        transformers are applied before formatting, escaping, and
        verification, and the result of the transformer is used to
        determine whether or not the value should be included in the
        result. If your transformer returns ``None``, the parameter will
        be completely omitted (including any prefix or suffix).
        """
        ...


@dataclass(slots=True)
class FieldConfig[T]:
    transformer: InterpolationTransformer[T] | None = None


class _DataclassFieldKwargs(TypedDict, total=False):
    init: bool
    repr: bool
    hash: bool | None
    compare: bool
    metadata: Mapping[Any, Any] | None
    kw_only: bool | Literal[_MISSING_TYPE.MISSING]


class _DataclassFieldKwargs14(_DataclassFieldKwargs, TypedDict, total=False):
    doc: str | None


# The following is adapted directly from typeshed. We did some formatting
# updates, and inserted our transformer param.
if sys.version_info >= (3, 14):
    @overload
    def template_field[_T](
            field_config: FieldConfig[_T] = ...,
            /, *,
            default: _T,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            **field_kwargs: Unpack[_DataclassFieldKwargs14]
            ) -> _T: ...
    @overload
    def template_field[_T](
            field_config: FieldConfig[_T] = ...,
            /, *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Callable[[], _T],
            **field_kwargs: Unpack[_DataclassFieldKwargs14]
            ) -> _T: ...
    @overload
    def template_field[_T](
            field_config: FieldConfig[_T] = ...,
            /, *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            **field_kwargs: Unpack[_DataclassFieldKwargs14]
            ) -> Any: ...

# This is technically only valid for >=3.10, but we require that anyways
else:
    @overload
    def template_field[_T](
            field_config: FieldConfig[_T] = ...,
            /, *,
            default: _T,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            **field_kwargs: Unpack[_DataclassFieldKwargs]
            ) -> _T: ...
    @overload
    def template_field[_T](
            field_config: FieldConfig[_T] = ...,
            /, *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Callable[[], _T],
            **field_kwargs: Unpack[_DataclassFieldKwargs]
            ) -> _T: ...
    @overload
    def template_field[_T](
            field_config: FieldConfig[_T] = ...,
            /, *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            **field_kwargs: Unpack[_DataclassFieldKwargs]
            ) -> Any: ...


def template_field(
        field_config: FieldConfig | None = None,
        /, *,
        metadata: Mapping[Any, Any] | None = None,
        **field_kwargs):
    if field_config is None:
        field_config = FieldConfig()

    if metadata is None:
        metadata = {'templatey.field_config': field_config}

    else:
        metadata = {
            **metadata,
            'templatey.field_config': field_config}

    return field(**field_kwargs, metadata=metadata)


# The following is adapted directly from typeshed. We did some formatting
# updates, and inserted our transformer param.
if sys.version_info >= (3, 14):
    @overload
    def param[_T](
            *,
            default: _T,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            # Deprecated name, but param itself is deprecated
            prerenderer: InterpolationTransformer[_T] | None = None,
            **field_kwargs: Unpack[_DataclassFieldKwargs14],
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Callable[[], _T],
            # Deprecated name, but param itself is deprecated
            prerenderer: InterpolationTransformer[_T] | None = None,
            **field_kwargs: Unpack[_DataclassFieldKwargs14],
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            # Deprecated name, but param itself is deprecated
            prerenderer: InterpolationTransformer[_T] | None = None,
            **field_kwargs: Unpack[_DataclassFieldKwargs14],
            ) -> Any: ...

# This is technically only valid for >=3.10, but we require that anyways
else:
    @overload
    def param[_T](
            *,
            default: _T,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            # Deprecated name, but param itself is deprecated
            prerenderer: InterpolationTransformer[_T] | None = None,
            **field_kwargs: Unpack[_DataclassFieldKwargs]
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Callable[[], _T],
            # Deprecated name, but param itself is deprecated
            prerenderer: InterpolationTransformer[_T] | None = None,
            **field_kwargs: Unpack[_DataclassFieldKwargs]
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            # Deprecated name, but param itself is deprecated
            prerenderer: InterpolationTransformer[_T] | None = None,
            **field_kwargs: Unpack[_DataclassFieldKwargs]
            ) -> Any: ...


# DEPRECATED. Use ``template_field`` instead.
@docnote(DocnoteConfig(include_in_docs=False))
def param(
        *,
        # Deprecated name, but param itself is deprecated
        prerenderer: InterpolationTransformer | None = None,
        metadata: Mapping[Any, Any] | None = None,
        **field_kwargs):
    field_config = FieldConfig(transformer=prerenderer)

    if metadata is None:
        metadata = {'templatey.field_config': field_config}

    else:
        metadata = {
            **metadata,
            'templatey.field_config': field_config}

    return field(**field_kwargs, metadata=metadata)


class _DataclassKwargs(TypedDict, total=False):
    init: bool
    repr: bool
    eq: bool
    order: bool
    unsafe_hash: bool
    frozen: bool
    match_args: bool
    kw_only: bool
    slots: bool
    weakref_slot: bool


@dataclass_transform(field_specifiers=(template_field, param, field, Field))
def template[T: type](
        parse_config: TemplateConfig,
        template_resource_locator: object,
        /, *,
        loader: Annotated[
                AsyncTemplateLoader | SyncTemplateLoader | None,
                Note('''Explicit template loaders can be passed to a template
                    instance to bypass the loader declared in the template
                    environment. This is primarily intended as a mechanism for
                    library developers to define redistributable templates as
                    part of their codebases, while not depending on the
                    template loader of the end user's codebase.''')
            ] = None,
        segment_modifiers: Annotated[
                Sequence[SegmentModifier] | None,
                Note('''An ordered sequence of ``SegmentModifier`` instances
                    to apply to every literal string segment of the loaded
                    template text.

                    The modifiers will be applied in the order that they are
                    declared. The first modifier to match the segment will
                    short-circuit the remaining modifiers, regardless of
                    whether or not it applies any changes.''')
            ] = None,
        **dataclass_kwargs: Unpack[_DataclassKwargs]
        ) -> Callable[[T], T]:
    """This both transforms the decorated class into a stdlib dataclass
    and declares it as a templatey template.

    **Note that unlike the stdlib dataclass decorator, this defaults to
    ``slots=True``.** If you find yourself having problems with
    metaclasses and/or subclassing, you can disable this by passing
    ``slots=False``. Generally speaking, though, this provides a
    free performance benefit. **If weakref support is required, be sure
    to pass ``weakref_slot=True``.
    """
    if segment_modifiers is None:
        segment_modifiers = []

    # As per docs, we default to using slots, since it makes everything
    # faster
    if 'slots' not in dataclass_kwargs:
        dataclass_kwargs['slots'] = True

    return functools.partial(
        make_template_definition,
        dataclass_kwargs=dataclass_kwargs,
        template_resource_locator=template_resource_locator,
        template_config=parse_config,
        segment_modifiers=segment_modifiers,
        explicit_loader=loader)


@dataclass(frozen=True)
class TemplateConfig[T: type, L: object]:
    """Template configs specify how the template and its interpolated
    content should be processed.
    """
    interpolator: Annotated[
        NamedInterpolator,
        Note('''The interpolator determines what characters are used for
            performing interpolations within the template. They can be
            escaped by repeating them, for example ``{{}}`` would be
            a literal ``{}`` with a curly braces interpolator.''')]
    variable_escaper: Annotated[
        VariableEscaper,
        Note('''Variables are always escaped. The variable escaper is
            the callable responsible for performing that escaping. If you
            don't need escaping, there are noop escapers within the prebaked
            template configs that you can use for convenience.''')]
    content_verifier: Annotated[
        ContentVerifier,
        Note('''Content isn't escaped, but it ^^is^^ verified. Content
            verification is a simple process that either succeeds or fails;
            it allows, for example, to allowlist certain HTML tags.''')]


@dataclass_transform(field_specifiers=(template_field, param, field, Field))
def make_template_definition[T: type](
        cls: T,
        *,
        dataclass_kwargs: _DataclassKwargs,
        # Note: needs to be understandable by template loader
        template_resource_locator: object,
        template_config: TemplateConfig,
        segment_modifiers: Sequence[SegmentModifier],
        explicit_loader: AsyncTemplateLoader | SyncTemplateLoader | None
        ) -> T:
    """Programmatically creates a template definition. Converts the
    requested class into a dataclass, passing along ``dataclass_kwargs``
    to the dataclass constructor. Then performs some templatey-specific
    bookkeeping. Returns the resulting dataclass.
    """
    cls = dataclass(**dataclass_kwargs)(cls)
    cls._templatey_signature = TemplateSignature(
        parse_config=template_config,
        resource_locator=template_resource_locator,
        explicit_loader=explicit_loader,
        segment_modifiers=tuple(segment_modifiers))

    closure_anchor = _CLOSURE_ANCHORS.get(None)
    if closure_anchor is not None:
        upstack_frame = _get_first_frame_from_other_module()
        if upstack_frame is not None:
            closure_anchor[cls] = upstack_frame

    return cls


@dataclass(frozen=True, slots=True)
class InjectedValue:
    """This is used by environment functions and complex content to
    indicate that a value is being injected into the template. Use it
    instead of a bare string to preserve an existing interpolation
    config, or to indicate whether verification and/or escaping should
    be applied to the value after conversion to a string.

    Note that, if both are defined, the variable escaper will be called
    first, before the content verifier.
    """
    value: object

    config: InterpolationConfig = field(default_factory=InterpolationConfig)
    use_content_verifier: bool = False
    use_variable_escaper: bool = True

    def __post_init__(self):
        if self.config.prefix is not None or self.config.suffix is not None:
            raise ValueError(
                'Injected values cannot have prefixes nor suffixes. If you '
                + 'need similar behavior, simply add the affix(es) to the '
                + 'iterable returned by the complex content or env function.')


class _ComplexContentBase(Protocol):

    def flatten(
            self,
            dependencies: Annotated[
                Mapping[str, object],
                Note(
                    '''The values of the variables declared as dependencies
                    in the constructor are passed to the call to ``flatten``
                    during rendering.
                    ''')],
            config: Annotated[
                InterpolationConfig,
                Note(
                    '''The interpolation configuration of the content
                    interpolation that the complex content is a member
                    of. Note that neither prefix nor suffix can be passed
                    on to an ``InjectedValue``; they must be manually included
                    in the return value if desired.
                    ''')],
            transformers: Annotated[
                Mapping[str, InterpolationTransformer | None],
                Note(
                    '''If a transformer is defined on a dependency variable,
                    it will be included here; otherwise, the value will be
                    set to ``None``.
                    ''')],
            ) -> Iterable[object | InjectedValue]:
        """Implement this for any instance of complex content.

        First, do whatever content modification you need to, based on
        the dependency variables declared in the constructor. Then,
        if needed, merge in the variable itself using an
        ``InjectedValue``, configuring it as appropriate.

        **Note that the parent interpolation config will be ignored by
        all strings returned by flattening individually.** So if, for
        example, you included a prefix in the content interpolation
        within the template itself, and then passed a ``ComplexContent``
        instance to the template instance, the prefix would be ignored
        completely (unless you do something with it in ``flatten``).

        **Also note that you are responsible for calling the dependency
        variable's ``InterpolationTransformer``. directly,** within your
        implementation of ``flatten``. This affords you the option to
        skip it if desired.

        > Example: noun quantity
        __embed__: 'code/python'
            class NaivePluralContent(ComplexContent):

                def flatten(
                        self,
                        dependencies: Mapping[str, object],
                        config: InterpolationConfig,
                        transformers:
                            Mapping[str, InterpolationTransformer | None],
                        ) -> Iterable[str | InjectedValue]:
                    \"""Pluralizes the name of the provided dependency.
                    For example, ``{'widget': 1}`` will be rendered as
                    "1 widget", but ``{'widget': 2}`` will be rendered as
                    "2 widgets".
                    \"""

                    # Assume only 1 dependency
                    name, value = next(iter(dependencies.items()))

                    if 0 <= value <= 1:
                        return (
                            InjectedValue(
                                value,
                                # This assumes no prefix/suffix
                                config=config,
                                use_content_verifier=False,
                                use_variable_escaper=True),
                            ' ',
                            name)

                    else:
                        return (
                            InjectedValue(
                                value,
                                # This assumes no prefix/suffix
                                config=config,
                                use_content_verifier=False,
                                use_variable_escaper=True),
                            ' ',
                            name,
                            's')
        """
        ...


@dataclass(slots=True, kw_only=True)
class ComplexContent(_ComplexContentBase):
    """Sometimes content isn't as simple as a ``string``. For example,
    content might include variable interpolations. Or you might need
    to modify the content slightly based on the variables -- for
    example, to get subject/verb alignment based on a number, gender
    alignment based on a pronoun, or whatever. ComplexContent gives
    you an escape hatch to do this: simply pass a ComplexContent
    instance as a value instead of a string.
    """

    dependencies: Annotated[
        Collection[str],
        Note(
            '''Complex content dependencies are the **variable** names
            that a piece of complex content depends on. These will be
            passed to the implemented ``flatten`` function during
            rendering.
            ''')]


@contextmanager
def anchor_closure_scope():
    """We strongly recommend against defining templates within a
    closure, as it can cause a number of fragility issues, and just
    generally makes less sense than defining templates at the module
    level. However, if you absolutely must create a new template within
    a closure, you must use ``anchor_closure_scope`` to give the
    templates a known closure scope. Can be used either as a decorator
    or a context:

    > Decorator usage
    __embed__: 'code/python'
        @anchor_closure_scope()
        def my_func():
            # template definition goes here
            ...

    > Context manager usage
    __embed__: 'code/python'
        def my_other_func():
            with anchor_closure_scope():
                # template definition goes here
                ...
    """
    ctx_token = _CLOSURE_ANCHORS.set({})
    try:
        yield
    finally:
        _CLOSURE_ANCHORS.reset(ctx_token)


def _get_first_frame_from_other_module() -> FrameType | None:
    """When templates are created from inside a closure (ex, during
    testing, where this is extremely common), we need access to the
    locals from the closure to resolve type hints. This method relies
    upon ``inspect`` to extract them.

    Note that this can be very sensitive to where, exactly, you put it
    within the templatey code. Always put it as close as possible to
    the public API method, so that the first frame from another module
    coincides with the call to decorate a template class.

    **Note that this is pretty fragile.** Or, put a different way: it
    does exactly what the function name suggest it does: it finds the
    FIRST frame from another module. That doesn't mean we won't return
    to this module; it doesn't mean it's from the actual client library,
    etc. It just means it's the first frame that isn't from this
    module.
    """
    upstack_frame = inspect.currentframe()
    if upstack_frame is None:
        return None
    else:
        # Technically not 100% correct since we might have reloads, but...
        # I mean at that point, good fucking luck with closures.
        this_module = upstack_module = sys.modules[__name__]
        while upstack_module is this_module:
            if upstack_frame is None:
                return None

            upstack_frame = upstack_frame.f_back
            upstack_module = inspect.getmodule(upstack_frame)

    return upstack_frame


@docnote(DocnoteConfig(include_in_docs=False))
def get_closure_locals(template_cls: TemplateClass) -> dict[str, Any] | None:
    """This function -- which is not intended to be part of the public
    API for templatey -- checks first for an anchored closure scope,
    and then (if one is found), checks for a registered frame for the
    passed template class. If everything checks out, we extract the
    locals from that frame and return them; otherwise, we return None.
    """
    closure_anchor = _CLOSURE_ANCHORS.get(None)
    if closure_anchor is not None:
        upstack_frame = closure_anchor.get(template_cls)
        if upstack_frame is not None:
            return upstack_frame.f_locals
