from __future__ import annotations

from typing import cast

from templatey._finalizers import ensure_recursive_totality
from templatey._slot_tree import PrerenderTreeNode
from templatey._slot_tree import build_slot_tree
from templatey._types import DynamicClassSlot
from templatey._types import Slot
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import Var
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import LiteralTemplateString
from templatey.parser import ParsedTemplateResource
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestBuildSlotTree:

    def test_forward_ref_works(self):
        """Slot trees with simple forward references must be created
        successfully and correctly.
        """
        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], Bar)._templatey_signature,
            Bar)

        slot_tree = build_slot_tree(Bar)

        assert (slot_tree / ('foo', Foo)).slot_cls is Foo

    def test_double_forward_ref(self):
        """Slot trees with forward references must be created
        successfully and correctly.

        This expands on the simple forward ref test case by having
        additional forward and backward refs on the resolved template
        class.
        """
        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Baz:
            zab: Var[str]

        @template(fake_template_config, object())
        class Foo:
            oof: Var[str]
            baz: Slot[Baz]
            runout: Slot[IRanOutOfFooLikeNames]

        @template(fake_template_config, object())
        class IRanOutOfFooLikeNames:
            ranout: Var[str]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], Bar)._templatey_signature,
            Bar)

        slot_tree = build_slot_tree(Bar)

        assert (slot_tree / ('foo', Foo)).slot_cls is Foo
        assert (
            slot_tree / ('foo', Foo) / ('runout', IRanOutOfFooLikeNames)
        ).slot_cls is IRanOutOfFooLikeNames
        assert (slot_tree / ('foo', Foo) / ('baz', Baz)).slot_cls is Baz

    def test_nested_forward_ref_works(self):
        """Slot trees with forward references must be created
        successfully and correctly.

        This expands on the simple forward ref test case by making sure
        that forward references are correctly passed along from nested
        templates to their enclosing templates.
        """
        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Baz:
            bar: Slot[Bar]

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], Baz)._templatey_signature,
            Baz)

        slot_tree = build_slot_tree(Baz)

        assert (slot_tree / ('bar', Bar) / ('foo', Foo)).slot_cls is Foo

    def test_simple_recursion_works(self):
        """Slot trees must support simple recursions back to the
        template class being declared.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Slot[Foo]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], Foo)._templatey_signature,
            Foo)

        slot_tree = build_slot_tree(Foo)

        assert (slot_tree / ('foo', Foo)) is slot_tree

    def test_recursion_loop_works(self):
        """Slot trees must support indirect recursion loops (chains)
        using forward references.
        """
        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Foo:
            bar: Slot[Bar]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], Bar)._templatey_signature,
            Bar)

        slot_tree = build_slot_tree(Bar)

        assert (slot_tree / ('foo', Foo) / ('bar', Bar)) is slot_tree

    def test_slot_multiples_recursion(self):
        """Templates with multiple slots of the same recursive type must
        be correctly defined, with both separate routes in the slot
        tree.
        """
        @template(fake_template_config, object())
        class FakeTemplateFordref:
            slot1: Slot[Baz | Foo | Bar]
            slot2: Slot[Baz | Foo | Bar]

        @template(fake_template_config, object())
        class Foo:
            bar_or_baz: Slot[Bar | Baz]

        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Baz:
            value: Var[str]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], FakeTemplateFordref
                )._templatey_signature,
            FakeTemplateFordref)

        slot_tree = build_slot_tree(FakeTemplateFordref)

        assert (
            slot_tree
            / ('slot1', Foo)
            / ('bar_or_baz', Bar)
            / ('foo', Foo)
        ).slot_cls is Foo
        assert (
            slot_tree
            / ('slot1', Bar)
            / ('foo', Foo)
        ).slot_cls is Foo
        assert (
            slot_tree
            / ('slot2', Bar)
            / ('foo', Foo)
        ).slot_cls is Foo


class TestPrerenderTreeDistillation:

    def test_empty_culling(self):
        """Distilling a prerender tree with nothing interesting -- no
        slots, no dynamic classes -- must result in a null tree.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], Foo
                )._templatey_signature,
            Foo)
        slot_tree = build_slot_tree(Foo)
        preload: dict[TemplateClass, ParsedTemplateResource] = {
            Foo: ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=1,
                        name='foo',
                        config=InterpolationConfig()),),
                variable_names=frozenset({'foo'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                function_names=frozenset(),
                data_names=frozenset(),
                function_calls={},
                slots={}),}

        prerender_tree = slot_tree.distill_prerender_tree(preload)
        assert prerender_tree is None

    def test_simple_extraction_culling(self):
        """Distilling a prerender tree containing a single static slot
        (without any function calls) and single dynamic slot must cull
        the irrelevant slot branch.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        @template(fake_template_config, object())
        class FakeTemplate:
            foo: Slot[Foo]
            bar: DynamicClassSlot

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], FakeTemplate
                )._templatey_signature,
            FakeTemplate)
        slot_tree = build_slot_tree(FakeTemplate)
        preload: dict[TemplateClass, ParsedTemplateResource] = {
            Foo: ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=1,
                        name='foo',
                        config=InterpolationConfig()),),
                variable_names=frozenset({'foo'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                function_names=frozenset(),
                data_names=frozenset(),
                function_calls={},
                slots={}),
            FakeTemplate: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='foo',
                        params={},
                        config=InterpolationConfig()),
                    InterpolatedSlot(
                        part_index=1,
                        name='bar',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'foo', 'bar'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={})}

        prerender_tree = slot_tree.distill_prerender_tree(preload)

        assert isinstance(prerender_tree, PrerenderTreeNode)
        assert not prerender_tree.has_route_for('foo', Foo)
        assert prerender_tree.dynamic_slot_names == ('bar',)

    def test_recursion_loop(self):
        """Recursion loops must be preserved when distilling the
        prerender tree, and templates with function calls must not be
        culled.
        """
        @template(fake_template_config, object())
        class FakeTemplateFordref:
            slot1: Slot[Baz | Foo | Bar]
            slot2: Slot[Baz | Foo | Bar]

        @template(fake_template_config, object())
        class Foo:
            bar_or_baz: Slot[Bar | Baz]

        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]
            dynamico: DynamicClassSlot

        @template(fake_template_config, object())
        class Baz:
            value: Var[str]

        ensure_recursive_totality(
            cast(type[TemplateIntersectable], FakeTemplateFordref
                )._templatey_signature,
            FakeTemplateFordref)
        slot_tree = build_slot_tree(FakeTemplateFordref)
        preload: dict[TemplateClass, ParsedTemplateResource] = {
            Foo: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='bar_or_baz',
                        params={},
                        config=InterpolationConfig()),),
                variable_names=frozenset(),
                content_names=frozenset(),
                slot_names=frozenset({'bar_or_baz'}),
                function_names=frozenset(),
                data_names=frozenset(),
                function_calls={},
                slots={}),
            Bar: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='foo',
                        params={},
                        config=InterpolationConfig()),
                    InterpolatedSlot(
                        part_index=1,
                        name='dynamico',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'foo', 'dynamico'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={}),
            Baz: ParsedTemplateResource(
                parts=(
                    LiteralTemplateString('foobar', part_index=0),
                    InterpolatedVariable(
                        part_index=0,
                        name='value',
                        config=InterpolationConfig()),
                    (interp_call := InterpolatedFunctionCall(
                        call_args_exp=None,
                        call_kwargs_exp=None,
                        part_index=1,
                        name='href',
                        call_args=['foo'],
                        call_kwargs={}))),
                variable_names=frozenset({'value'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                function_names=frozenset({'href'}),
                data_names=frozenset(),
                function_calls={'href': (interp_call,)},
                slots={}),
            FakeTemplateFordref: ParsedTemplateResource(
                parts=(
                    InterpolatedSlot(
                        part_index=0,
                        name='slot1',
                        params={},
                        config=InterpolationConfig()),
                    InterpolatedSlot(
                        part_index=1,
                        name='slot2',
                        params={},
                        config=InterpolationConfig()),
                    ),
                variable_names=frozenset({}),
                content_names=frozenset(),
                slot_names=frozenset({'slot1', 'slot2'}),
                slots={},
                data_names=frozenset({}),
                function_names=frozenset(),
                function_calls={}),}

        prerender_tree = slot_tree.distill_prerender_tree(preload)

        assert isinstance(prerender_tree, PrerenderTreeNode)
        assert prerender_tree.has_route_for('slot1', Foo)
        assert prerender_tree.has_route_for('slot1', Bar)
        assert prerender_tree.has_route_for('slot1', Baz)
        assert (
            prerender_tree
            / ('slot1', Foo)
            / ('bar_or_baz', Bar)
            / ('foo', Foo)
        ) is (prerender_tree / ('slot1', Foo))

        assert len((
            prerender_tree / ('slot1', Foo) / ('bar_or_baz', Baz)
        ).abstract_calls) == 1

        assert (
            prerender_tree / ('slot2', Bar)
        ).dynamic_slot_names == ('dynamico',)
