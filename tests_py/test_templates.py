from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import FrozenInstanceError
from dataclasses import is_dataclass
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from templatey._types import Slot
from templatey._types import TemplateIntersectable
from templatey._types import Var
from templatey._types import is_template_class
from templatey._types import is_template_class_xable
from templatey._types import is_template_instance_xable
from templatey.templates import ParseConfig
from templatey.templates import SegmentModifier
from templatey.templates import make_template_definition
from templatey.templates import template

from templatey_testutils import fake_render_config
from templatey_testutils import fake_template_config


class TestIsTemplateClass:

    def test_positive(self):
        class FakeTemplate:
            ...

        # Quick and dirty. This will break if we add anything that isn't a
        # class var to the TemplateIntersectable class.
        for key in get_type_hints(
            TemplateIntersectable,
            localns=defaultdict(MagicMock)
        ):
            setattr(FakeTemplate, key, object())

        assert is_template_class_xable(FakeTemplate)

    def test_negative(self):
        class FakeTemplate:
            ...

        assert not is_template_class_xable(FakeTemplate)


class TestIsTemplateInstance:
    def test_positive(self):
        class FakeTemplate:
            ...

        # Quick and dirty. This will break if we add anything that isn't a
        # class var to the TemplateIntersectable class.
        for key in get_type_hints(
            TemplateIntersectable,
            localns=defaultdict(MagicMock)
        ):
            setattr(FakeTemplate, key, object())

        instance = FakeTemplate()
        assert is_template_instance_xable(instance)

    def test_negative(self):
        class FakeTemplate:
            ...
        instance = FakeTemplate()

        assert not is_template_instance_xable(instance)


class TestTemplate:
    """template() must have the expected API. (Added 2025-10-12: must
    also include backcompat for legacy API; to be removed at some future
    date).
    """

    def test_works(self):
        """The template decorator must complete without error and
        result in a template class.
        """
        locator = object()

        @template(locator, fake_render_config)
        class FakeTemplate:
            ...

        assert isinstance(FakeTemplate, type)
        assert is_template_class_xable(FakeTemplate)
        assert FakeTemplate._templatey_signature.resource_locator is locator

    def test_supports_passthrough(self):
        """Additional params must be passed through to the dataclass
        decorator.
        """
        locator = object()

        @template(locator, fake_render_config, frozen=True)
        class FakeTemplate:
            foo: Var[str]

        instance = FakeTemplate(foo='foo')
        with pytest.raises(FrozenInstanceError):
            instance.foo = 'bar'  # type: ignore

        assert isinstance(FakeTemplate, type)
        assert is_template_class_xable(FakeTemplate)
        assert FakeTemplate._templatey_signature.resource_locator is locator

    def test_legacy_works(self):
        """The template decorator must complete without error and
        result in a template class.
        """
        locator = object()

        @template(fake_template_config, locator)
        class FakeTemplate:
            ...

        assert isinstance(FakeTemplate, type)
        assert is_template_class_xable(FakeTemplate)
        assert FakeTemplate._templatey_signature.resource_locator is locator

    def test_legacy_supports_passthrough(self):
        """Additional params must be passed through to the dataclass
        decorator.
        """
        locator = object()

        @template(fake_template_config, locator, frozen=True)
        class FakeTemplate:
            foo: Var[str]

        instance = FakeTemplate(foo='foo')
        with pytest.raises(FrozenInstanceError):
            instance.foo = 'bar'  # type: ignore

        assert isinstance(FakeTemplate, type)
        assert is_template_class_xable(FakeTemplate)
        assert FakeTemplate._templatey_signature.resource_locator is locator


class TestMakeTemplateDefinition:
    """make_template_definition()
    """

    def test_simplest_case(self):
        """The signature attribute must be added to the class before
        exiting the template definition maker, along with the initial
        values.
        """
        class Foo:
            foo: Var[str]

        retval = make_template_definition(
            Foo,
            dataclass_kwargs={},
            template_resource_locator=object(),
            render_config=fake_render_config,
            parse_config=None,
            env_config=None)
        assert is_template_class(retval)
        assert is_template_class_xable(retval)
        # This is weird, but the point here is to make sure that the attribute
        # is defined. We could do ``hasattr``, but then we'd be missing type
        # support.
        assert retval._templatey_signature.parse_config is not object()
        assert retval._templatey_signature.render_config is not object()
        assert retval._templatey_signature.resource_locator is not object()
        assert retval._templatey_signature.explicit_loader is not object()

    def test_segment_modifiers_assigned(self):
        """Segment modifiers, if defined, must be added to the template
        signature during the initial definition.
        """
        class Foo:
            foo: Var[str]

        modifiers = [
            SegmentModifier(
                pattern=re.compile(''),
                modifier=lambda modifier_match: [])]

        retval = make_template_definition(
            Foo,
            dataclass_kwargs={},
            template_resource_locator=object(),
            render_config=fake_render_config,
            parse_config=ParseConfig(segment_modifiers=tuple(modifiers)),
            env_config=None)

        assert is_template_class_xable(retval)
        assert retval._templatey_signature.parse_config.segment_modifiers == \
            tuple(modifiers)

    def test_closure_resolution_works(self):
        """Another template referenced as a slot must successfully
        return, even if that template was defined in a closure.

        This is specifically targeting our workaround to get_type_hints
        needing access to the function locals within the closure.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        class Bar:
            foo: Slot[Foo]

        retval = make_template_definition(
            Bar,
            dataclass_kwargs={},
            template_resource_locator=object(),
            render_config=fake_render_config,
            parse_config=None,
            env_config=None)
        assert is_template_class_xable(retval)

    def test_forward_ref_works(self):
        """Slots must be definable using forward references.
        """
        class Bar:
            foo: Slot[Foo]

        retval = make_template_definition(
            Bar,
            dataclass_kwargs={},
            template_resource_locator=object(),
            render_config=fake_render_config,
            parse_config=None,
            env_config=None)
        assert is_template_class_xable(retval)

        # Mostly just here to make sure the typechecker doesn't freak out;
        # otherwise this is kinda redundant
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        assert is_template_class_xable(Foo)

    def test_is_dataclass(self):
        """The template maker must also convert the class to a
        dataclass.
        """
        class FakeTemplate:
            foo: Var[str]

        retval = make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            render_config=fake_render_config,
            parse_config=None,
            env_config=None)
        assert is_dataclass(retval)

    def test_supports_passthrough(self):
        """Dataclass kwargs must be forwarded to the dataclass
        constructor.
        """
        class FakeTemplate:
            foo: Var[str]

        template_cls = make_template_definition(
            FakeTemplate,
            dataclass_kwargs={'frozen': True, 'slots': True},
            template_resource_locator=object(),
            render_config=fake_render_config,
            parse_config=None,
            env_config=None)

        instance = template_cls(foo='foo')  # type: ignore
        with pytest.raises(FrozenInstanceError):
            instance.foo = 'bar'  # type: ignore

        assert hasattr(instance, '__slots__')
