from __future__ import annotations

from typing import cast

from templatey._error_collector import ErrorCollector
from templatey._finalizers import finalize_signature
from templatey._renderer import RenderContext
from templatey._types import DynamicClassSlot
from templatey._types import TemplateIntersectable
from templatey._types import Var
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import LiteralTemplateString
from templatey.parser import ParsedTemplateResource
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestRenderContext:

    def test_prep_render_with_multiply_nested_dynamics(self):
        """Dynamic-class slots, which have instances that themselves
        declare dynamic-class slots, must result in all slot types
        being successfully loaded.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: DynamicClassSlot

        @template(fake_template_config, object())
        class Middle:
            innermost: DynamicClassSlot

        @template(fake_template_config, object())
        class Innermost:
            var: Var[str]

        resources = {
            Outermost: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='middle',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'middle'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={}),
            Middle: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='innermost',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'innermost'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={}),
            Innermost: ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=1,
                        name='var',
                        config=InterpolationConfig()),),
                variable_names=frozenset({'var'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                function_names=frozenset(),
                data_names=frozenset(),
                function_calls={},
                slots={}),}
        render_ctx = RenderContext(
            template_preload={},
            function_precall={},
            error_collector=ErrorCollector())
        root_instance = Outermost(
            middle=[
                Middle(
                    innermost=[Innermost(var='foo')])])

        req_count = 0
        for req in render_ctx.prep_render(root_instance):
            req_count += 1
            for template_cls in req.to_load:
                signature = cast(
                    type[TemplateIntersectable], template_cls
                    )._templatey_signature
                # Note that we can't just call the finalization steps directly;
                # there's more going on behind the scenes than that within the
                # finalizer
                with finalize_signature(
                    signature,
                    template_cls,
                    preload=render_ctx.template_preload,
                    # Note: DO NOT use resources here! The finalizer assumes
                    # that any existing resources are already finalized, and
                    # then we end up skipping all of the finalization steps
                    parse_cache={}
                ) as sig_finalizer:
                    for required_template_cls in sig_finalizer.required_loads:
                        sig_finalizer.preload[required_template_cls] = \
                            resources[required_template_cls]

        assert req_count == 3
        assert Middle in render_ctx.template_preload
        assert Outermost in render_ctx.template_preload
        assert Innermost in render_ctx.template_preload

    def test_similar_recursion_new_instance(self):
        """Dynamic-class slots that find an instance of an
        already-encountered dynamic class must continue to search the
        instance tree for new templates, and not accidentally short
        circuit on the first known class (since there might be new
        classes nested deeper within the instance tree).
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: DynamicClassSlot

        @template(fake_template_config, object())
        class Middle:
            innermost: DynamicClassSlot

        @template(fake_template_config, object())
        class Innermost:
            var: Var[str]

        resources = {
            Outermost: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='middle',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'middle'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={}),
            Middle: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='innermost',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'innermost'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={}),
            Innermost: ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=1,
                        name='var',
                        config=InterpolationConfig()),),
                variable_names=frozenset({'var'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                function_names=frozenset(),
                data_names=frozenset(),
                function_calls={},
                slots={}),}
        render_ctx = RenderContext(
            template_preload={},
            function_precall={},
            error_collector=ErrorCollector())

        # THIS IS THE ONLY DIFFERENCE TO THE ABOVE TEST!!
        # Note that you might consider parameterizing them so that there
        # isn't this huge redundant test function, but... not sure. Don't
        # want to lose the docs
        root_instance = Outermost(
            middle=[
                Middle(
                    innermost=[
                        Middle(
                            innermost=[
                                Middle(
                                    innermost=[Innermost(var='foo')])])])])

        req_count = 0
        for req in render_ctx.prep_render(root_instance):
            req_count += 1
            for template_cls in req.to_load:
                signature = cast(
                    type[TemplateIntersectable], template_cls
                    )._templatey_signature
                # Note that we can't just call the finalization steps directly;
                # there's more going on behind the scenes than that within the
                # finalizer
                with finalize_signature(
                    signature,
                    template_cls,
                    preload=render_ctx.template_preload,
                    # Note: DO NOT use resources here! The finalizer assumes
                    # that any existing resources are already finalized, and
                    # then we end up skipping all of the finalization steps
                    parse_cache={}
                ) as sig_finalizer:
                    for required_template_cls in sig_finalizer.required_loads:
                        sig_finalizer.preload[required_template_cls] = \
                            resources[required_template_cls]

        assert req_count == 3
        assert Middle in render_ctx.template_preload
        assert Outermost in render_ctx.template_preload
        assert Innermost in render_ctx.template_preload
