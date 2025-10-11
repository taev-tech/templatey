from __future__ import annotations

from templatey._fields import NormalizedFieldset
from templatey._types import Content
from templatey._types import DynamicClassSlot
from templatey._types import Slot
from templatey._types import Var
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestNormalizezdFieldset:

    def test_correct_extracted_names(self):
        """NormalizedFieldset.from_template_cls must extract params and
        data into their associated categories.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        @template(fake_template_config, object())
        class FakeTemplate:
            foo: Slot[Foo]
            bar: Var[str]
            baz: Content[str]
            data: int
            dynacls: DynamicClassSlot

        fieldset = NormalizedFieldset.from_template_cls(FakeTemplate)
        assert fieldset.slotpaths == frozenset({
            ('foo', Foo)})
        assert fieldset.slot_names == frozenset({'foo'})
        assert fieldset.data_names == frozenset({'data'})
        assert fieldset.content_names == frozenset({'baz'})
        assert fieldset.var_names == frozenset({'bar'})
        assert fieldset.dynamic_class_slot_names == frozenset({'dynacls'})

    def test_forward_ref_works(self):
        """Fieldsets must be created correctly from template classes
        with slots defined as forward refs.
        """
        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        fieldset = NormalizedFieldset.from_template_cls(Bar)
        assert fieldset.slotpaths == frozenset({
            ('foo', Foo)})

    def test_double_forward_ref(self):
        """Fieldsets must be created correctly from template classes
        with slots defined as forward refs.

        This expands on the plain forward ref test case by having
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

        bar_fieldset = NormalizedFieldset.from_template_cls(Bar)
        foo_fieldset = NormalizedFieldset.from_template_cls(Foo)
        assert bar_fieldset.slotpaths == frozenset({
            ('foo', Foo)})
        assert foo_fieldset.slotpaths == frozenset({
            ('baz', Baz), ('runout', IRanOutOfFooLikeNames)})

    def test_slot_union(self):
        """Slots declared as the union of two templates must be expanded
        to include both of the union members as independent slotpaths.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        @template(fake_template_config, object())
        class Bar:
            bar: Var[str]

        @template(fake_template_config, object())
        class FakeTemplate:
            foo: Slot[Foo | Bar]

        fieldset = NormalizedFieldset.from_template_cls(FakeTemplate)
        assert fieldset.slotpaths == frozenset({
            ('foo', Foo),
            ('foo', Bar)})

    def test_slot_multiples_union(self):
        """Slots declared as the union of two templates must be expanded
        to include both of the union members as independent slotpaths.

        This must alwo work for templates that declare multiple
        identical unions in different slots.
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            foo1: Slot[Foo | Bar | Baz]
            foo2: Slot[Foo | Bar | Baz]

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        @template(fake_template_config, object())
        class Bar:
            foo: Var[str]

        @template(fake_template_config, object())
        class Baz:
            foo: Var[str]

        fieldset = NormalizedFieldset.from_template_cls(FakeTemplate)
        assert fieldset.slotpaths == frozenset({
            ('foo1', Foo),
            ('foo1', Bar),
            ('foo1', Baz),
            ('foo2', Foo),
            ('foo2', Bar),
            ('foo2', Baz),})
