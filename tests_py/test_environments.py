from __future__ import annotations

from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import anyio
import pytest

from templatey._bootstrapping import EMPTY_TEMPLATE_XABLE
from templatey._finalizers import ensure_recursive_totality
from templatey._provenance import Provenance
from templatey._renderer import FuncExecutionRequest
from templatey._renderer import FuncExecutionResult
from templatey._types import Slot
from templatey._types import TemplateIntersectable
from templatey._types import Var
from templatey.environments import RenderEnvironment
from templatey.environments import _TemplateFunctionContainer
from templatey.exceptions import MismatchedTemplateEnvironment
from templatey.exceptions import MismatchedTemplateSignature
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import LiteralTemplateString
from templatey.parser import ParsedTemplateResource
from templatey.parser import parse
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.loaders import InlineStringTemplateLoader
from templatey.prebaked.template_configs import html
from templatey.templates import template

from templatey_testutils import fake_template_config

_inline_loader = InlineStringTemplateLoader()


def href(val: str) -> tuple[str, ...]:
    """A fake template function."""
    return (val,)


def func_with_starrings(
        *args: str,
        **kwargs: str
        ) -> tuple[str, ...]:
    return (
        *(str(arg) for arg in args),
        *kwargs,
        *kwargs.values())


async def func_with_starrings_async(
        *args: str,
        **kwargs: str
        ) -> tuple[str, ...]:
    await anyio.sleep(0)
    return (
        *(str(arg) for arg in args),
        *kwargs,
        *kwargs.values())


@template(fake_template_config, 'fake_global')
class FakeGlobalTemplate:
    """There's some strangeness in the way that get_type_hints is
    implemented when inside closures, which causes locally defined
    references to break. So instead, we just moved the template out to
    a global scope, so that get_type_hints can find it.

    That being said, I've since added a workaround for that, so... I
    guess this could be moved back into a closure if desired.
    """
    foo: Var[str]


@template(
    html,
    '<em>my super special inline template {var.foo}</em>',
    loader=_inline_loader)
class InlineTemplate:
    foo: Var[str]


class TestRenderEnvironment:

    def test_with_init_func(self):
        """Creating a render environment with functions included in the
        call to __init__ must add them to the render env's function
        registry.
        """
        render_env = RenderEnvironment(
            env_functions=(href,),
            template_loader=DictTemplateLoader())
        assert isinstance(
            render_env._env_functions['href'], _TemplateFunctionContainer)
        assert render_env._env_functions['href'].function is href

    def test_with_register_func(self):
        """Calling register_env_function must add a function to the
        render env's function registry.
        """
        render_env = RenderEnvironment(
            template_loader=DictTemplateLoader())
        assert not render_env._env_functions

        render_env.register_env_function(href)
        assert isinstance(
            render_env._env_functions['href'], _TemplateFunctionContainer)
        assert render_env._env_functions['href'].function is href

    def test_load_sync_success(self):
        """The load_sync wrapper around must successfully invoke
        _parse_and_validate on the happy case and return the result of
        template loading.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_sync, wraps=loader.load_sync)
        loader.load_sync = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(render_env, '_parse_and_validate', pa_va_mock):
            result = render_env.load_sync(FakeTemplate)
        assert loader_mock.call_count == 1
        assert pa_va_mock.call_count == 1
        assert result is pa_va_mock.return_value

    def test_load_sync_with_explicit_loader(self):
        """Load sync with an explicit loader must correctly bypass the
        environment loader and load the template directly from the
        explicit loader.
        """
        loader = DictTemplateLoader(templates={})
        env_loader_mock = Mock(spec=loader.load_sync, wraps=loader.load_sync)
        loader.load_sync = env_loader_mock
        expl_loader_mock = Mock(
            spec=_inline_loader.load_sync, wraps=_inline_loader.load_sync)

        render_env = RenderEnvironment(template_loader=loader)
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(
            render_env, '_parse_and_validate', pa_va_mock
        ), patch.object(_inline_loader, 'load_sync', expl_loader_mock):
            result = render_env.load_sync(InlineTemplate)

        assert env_loader_mock.call_count == 0
        assert expl_loader_mock.call_count == 1
        assert pa_va_mock.call_count == 1
        assert result is pa_va_mock.return_value

    def test_load_sync_cache_hit(self):
        """The load_sync wrapper must successfully retrieve an
        already-loaded template from its cache rather than reloading it.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_sync, wraps=loader.load_sync)
        loader.load_sync = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        render_env._parsed_template_cache[FakeTemplate] = Mock()
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(render_env, '_parse_and_validate', pa_va_mock):
            result = render_env.load_sync(FakeTemplate)
        assert loader_mock.call_count == 0
        assert pa_va_mock.call_count == 0
        assert result is render_env._parsed_template_cache[FakeTemplate]

    def test_load_sync_force_reload(self):
        """The load_sync wrapper must bypass the loader cache if it's
        passed force_reload=True.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_sync, wraps=loader.load_sync)
        loader.load_sync = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        render_env._parsed_template_cache[FakeTemplate] = Mock()
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(render_env, '_parse_and_validate', pa_va_mock):
            result = render_env.load_sync(FakeTemplate, force_reload=True)
        assert loader_mock.call_count == 1
        assert pa_va_mock.call_count == 1
        assert result is pa_va_mock.return_value

    @pytest.mark.anyio
    async def test_load_async_success(self):
        """The load_async wrapper around must successfully invoke
        _parse_and_validate on the happy case and return the result of
        template loading.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(render_env, '_parse_and_validate', pa_va_mock):
            result = await render_env.load_async(FakeTemplate)
        assert loader_mock.call_count == 1
        assert pa_va_mock.call_count == 1
        assert result is pa_va_mock.return_value

    @pytest.mark.anyio
    async def test_load_async_with_explicit_loader(self):
        """Load async with an explicit loader must correctly bypass the
        environment loader and load the template directly from the
        explicit loader.
        """
        loader = DictTemplateLoader(templates={})
        env_loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = env_loader_mock
        expl_loader_mock = Mock(
            spec=_inline_loader.load_async, wraps=_inline_loader.load_async)

        render_env = RenderEnvironment(template_loader=loader)
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(
            render_env, '_parse_and_validate', pa_va_mock
        ), patch.object(_inline_loader, 'load_async', expl_loader_mock):
            result = await render_env.load_async(InlineTemplate)

        assert env_loader_mock.call_count == 0
        assert expl_loader_mock.call_count == 1
        assert pa_va_mock.call_count == 1
        assert result is pa_va_mock.return_value

    @pytest.mark.anyio
    async def test_load_async_cache_hit(self):
        """The load_async wrapper must successfully retrieve an
        already-loaded template from its cache rather than reloading it.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        render_env._parsed_template_cache[FakeTemplate] = Mock()
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(render_env, '_parse_and_validate', pa_va_mock):
            result = await render_env.load_async(FakeTemplate)
        assert loader_mock.call_count == 0
        assert pa_va_mock.call_count == 0
        assert result is render_env._parsed_template_cache[FakeTemplate]

    @pytest.mark.anyio
    async def test_load_async_force_reload(self):
        """The load_async wrapper must bypass the loader cache if it's
        passed force_reload=True.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        render_env._parsed_template_cache[FakeTemplate] = Mock()
        pa_va_mock = Mock(spec=render_env._parse_and_validate)
        pa_va_mock.return_value.function_calls = {}
        with patch.object(render_env, '_parse_and_validate', pa_va_mock):
            result = await render_env.load_async(
                FakeTemplate, force_reload=True)
        assert loader_mock.call_count == 1
        assert pa_va_mock.call_count == 1
        assert result is pa_va_mock.return_value

    @patch('templatey.environments.parse', spec=parse)
    def test_parse_and_validate_does_validation(self, mock_parse):
        """parse_and_cache must call into both template validation
        functions. It must also, of course, call into parsing.

        It must not update the cache.
        """
        # Doesn't need to be right, just needs to be a dataclass instance
        mock_parse.return_value = ParsedTemplateResource(
            parts=(),
            variable_names=frozenset(),
            content_names=frozenset(),
            slot_names=frozenset(),
            function_names=frozenset(),
            data_names=frozenset(),
            function_calls={},
            slots={})

        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        render_env._validate_env_functions = Mock(
            spec=render_env._validate_env_functions)
        render_env._validate_template_signature = Mock(
            spec=render_env._validate_template_signature)
        result = render_env._parse_and_validate(
            FakeTemplate,
            EMPTY_TEMPLATE_XABLE._templatey_signature,
            template_text='foobar',
            override_validation_strictness=None)

        # Note that it won't be the actual object, because we created a copy
        # while applying modifiers
        assert result == mock_parse.return_value
        assert mock_parse.call_count == 1
        assert render_env._validate_env_functions.call_count == 1
        assert render_env._validate_template_signature.call_count == 1

        assert FakeTemplate not in render_env._parsed_template_cache

    def test_validate_env_functions_matching_trivial(self):
        """_validate_env_functions must succeed if the template
        functions defined in the render environment match those defined
        in the template itself.

        This tests the trivial case, where no functions are defined in
        either the environment nor the template.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        result = render_env._validate_env_functions(
            FakeTemplate,
            ParsedTemplateResource(
                parts=(LiteralTemplateString('foobar', part_index=0),),
                variable_names=frozenset(),
                content_names=frozenset(),
                slot_names=frozenset(),
                slots={},
                data_names=frozenset(),
                function_names=frozenset(),
                function_calls={}))

        assert result

    def test_validate_env_functions_with_matching_function(self):
        """_validate_env_functions must succeed if the template
        functions defined in the render environment match those defined
        in the template itself.

        This tests when the template actually does have function calls
        defined, and those are included in the environment.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(
            env_functions=(href,),
            template_loader=loader)
        result = render_env._validate_env_functions(
            FakeTemplate,
            ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedFunctionCall(
                        call_args_exp=None,
                        call_kwargs_exp=None,
                        part_index=1,
                        name='href',
                        call_args=['foo'],
                        call_kwargs={})),
                variable_names=frozenset(),
                content_names=frozenset(),
                slot_names=frozenset(),
                slots={},
                data_names=frozenset(),
                function_names=frozenset({'href'}),
                function_calls={}))

        assert result

    def test_validate_env_functions_with_missing_function(self):
        """_validate_env_functions must raise
        MismatchedTemplateEnvironment if the template defines function
        calls that aren't present inside the current render environment.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        with pytest.raises(MismatchedTemplateEnvironment):
            render_env._validate_env_functions(
                FakeTemplate,
                ParsedTemplateResource(
                    parts=(LiteralTemplateString('foobar', part_index=0),),
                    variable_names=frozenset(),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    data_names=frozenset(),
                    function_names=frozenset({'href'}),
                    function_calls={'href': (InterpolatedFunctionCall(
                        call_args_exp=None,
                        call_kwargs_exp=None,
                        part_index=1,
                        name='href',
                        call_args=['foo'],
                        call_kwargs={}),)}))

    def test_validate_env_functions_with_mismatched_signature(self):
        """_validate_env_functions must raise
        MismatchedTemplateEnvironment if the template defines function
        calls that don't match the function signature of their actual
        functions.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(
            env_functions=(href,),
            template_loader=loader)
        with pytest.raises(MismatchedTemplateEnvironment):
            render_env._validate_env_functions(
                FakeTemplate,
                ParsedTemplateResource(
                    parts=(LiteralTemplateString('foobar', part_index=0),),
                    variable_names=frozenset(),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    data_names=frozenset(),
                    function_names=frozenset({'href'}),
                    function_calls={'href': (InterpolatedFunctionCall(
                        call_args_exp=None,
                        call_kwargs_exp=None,
                        part_index=1,
                        name='href',
                        call_args=['foo'],
                        call_kwargs={'not_present': True}),)}))

    def test_validate_signature_success(self):
        """_validate_template_signature must return True if the
        in-code template signature matches the as-parsed template
        signature.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]
            bar: Slot[FakeGlobalTemplate]
            baz: str

        template_signature = cast(
            type[TemplateIntersectable], FakeTemplate)._templatey_signature
        ensure_recursive_totality(template_signature, FakeTemplate)

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        result = render_env._validate_template_signature(
            FakeTemplate,
            template_signature,
            ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=1,
                        name='foo',
                        config=InterpolationConfig()),
                    InterpolatedSlot(
                        part_index=2,
                        name='bar',
                        params={},
                        config=InterpolationConfig())),
                variable_names=frozenset({'foo'}),
                content_names=frozenset(),
                slot_names=frozenset({'bar'}),
                slots={},
                data_names=frozenset({'baz'}),
                function_names=frozenset(),
                function_calls={}),
            strict_mode=True)

        assert result

    def test_validate_signature_var_vs_content(self):
        """_validate_template_signature must raise
        MismatchedTemplateSignature if you mix up content vs variables
        between your in-code template vs in-text template.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        template_signature = cast(
            type[TemplateIntersectable], FakeTemplate)._templatey_signature
        ensure_recursive_totality(template_signature, FakeTemplate)

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        with pytest.raises(MismatchedTemplateSignature):
            render_env._validate_template_signature(
                FakeTemplate,
                template_signature,
                ParsedTemplateResource(
                    parts=(
                        LiteralTemplateString('foobar', part_index=0),
                        InterpolatedContent(
                            part_index=1,
                            name='foo',
                            config=InterpolationConfig())),
                    variable_names=frozenset(),
                    content_names=frozenset({'foo'}),
                    slot_names=frozenset(),
                    slots={},
                    data_names=frozenset(),
                    function_names=frozenset(),
                    function_calls={}),
                strict_mode=True)

    def test_validate_signature_too_much_in_strict_mode(self):
        """_validate_template_signature must raise
        MismatchedTemplateSignature when running in strict mode if there
        are vars defined in code that are missing in the template text.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        template_signature = cast(
            type[TemplateIntersectable], FakeTemplate)._templatey_signature
        ensure_recursive_totality(template_signature, FakeTemplate)

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        with pytest.raises(MismatchedTemplateSignature):
            render_env._validate_template_signature(
                FakeTemplate,
                template_signature,
                ParsedTemplateResource(
                    parts=(LiteralTemplateString('foobar', part_index=0),),
                    variable_names=frozenset(),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    data_names=frozenset(),
                    function_names=frozenset(),
                    function_calls={}),
                strict_mode=True)

    def test_validate_signature_extra_data_in_strict_mode(self):
        """_validate_template_signature must not raise if there are
        DATA attributes defined on the template that are missing in the
        template text.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]
            bar: int

        template_signature = cast(
            type[TemplateIntersectable], FakeTemplate)._templatey_signature
        ensure_recursive_totality(template_signature, FakeTemplate)

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        result = render_env._validate_template_signature(
            FakeTemplate,
            template_signature,
            ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=1,
                        name='foo',
                        config=InterpolationConfig()),),
                variable_names=frozenset({'foo'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                slots={},
                data_names=frozenset(),
                function_names=frozenset(),
                function_calls={}),
            strict_mode=True)

        assert result

    def test_validate_signature_too_much_in_lax_mode(self):
        """_validate_template_signature must NOT raise
        MismatchedTemplateSignature when running in strict mode if there
        are vars defined in code that are missing in the template text.
        """
        @template(fake_template_config, 'fake')
        class FakeTemplate:
            foo: Var[str]

        template_signature = cast(
            type[TemplateIntersectable], FakeTemplate)._templatey_signature
        ensure_recursive_totality(template_signature, FakeTemplate)

        loader = DictTemplateLoader(templates={'fake': 'foobar'})
        loader_mock = Mock(spec=loader.load_async, wraps=loader.load_async)
        loader.load_async = loader_mock

        render_env = RenderEnvironment(template_loader=loader)
        result = render_env._validate_template_signature(
            FakeTemplate,
            template_signature,
            ParsedTemplateResource(
                parts=(LiteralTemplateString('foobar', part_index=0),),
                variable_names=frozenset(),
                content_names=frozenset(),
                slot_names=frozenset(),
                slots={},
                data_names=frozenset(),
                function_names=frozenset(),
                function_calls={}),
            strict_mode=False)

        assert result

    def test_execute_env_function_sync(self):
        """An env function execution must successfully execute.
        """
        result_key = object()
        render_env = RenderEnvironment(
            env_functions=(func_with_starrings,),
            template_loader=DictTemplateLoader())
        request = FuncExecutionRequest(
            name='func_with_starrings',
            args=['foo'],
            kwargs={'bar': 'baz'},
            result_key=result_key,
            provenance=Provenance())
        precall = {}

        render_env._execute_env_function_sync(request, [], precall)
        assert result_key in precall
        result = precall[result_key]
        assert isinstance(result, FuncExecutionResult)
        assert result.retval == ('foo', 'bar', 'baz')

    @pytest.mark.anyio
    async def test_execute_env_function_async(self):
        """An env function execution must successfully execute.
        """
        result_key = object()
        render_env = RenderEnvironment(
            env_functions=(func_with_starrings_async,),
            template_loader=DictTemplateLoader())
        request = FuncExecutionRequest(
            name='func_with_starrings_async',
            args=['foo'],
            kwargs={'bar': 'baz'},
            result_key=result_key,
            provenance=Provenance())
        precall = {}

        await render_env._execute_env_function_async(request, [], precall)
        assert result_key in precall
        result = precall[result_key]
        assert isinstance(result, FuncExecutionResult)
        assert result.retval == ('foo', 'bar', 'baz')
