from __future__ import annotations

from typing import cast

from templatey._error_collector import ErrorCollector
from templatey._fields import NormalizedFieldset
from templatey._finalizers import ensure_prerender_tree
from templatey._finalizers import ensure_recursive_totality
from templatey._finalizers import ensure_slot_tree
from templatey._finalizers import finalize_signature
from templatey._renderer import RenderContext
from templatey._slot_tree import SlotTreeNode
from templatey._types import Slot
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import Var
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import LiteralTemplateString
from templatey.parser import ParsedTemplateResource
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestFinalizeSignature:

    def test_discovers_all_required_nocache(self):
        """The returned finalizer must include all of the classes in the
        required loads when none are in the cache (and no override
        specified).

        Additionally, the finalizer must be yielded with the target
        template resource set, and the cache must be populated with all
        of the required template resources.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: Slot[Middle]

        @template(fake_template_config, object())
        class Middle:
            innermost: Slot[Innermost]

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

        signature = cast(
            type[TemplateIntersectable], Outermost
            )._templatey_signature
        # Note: DO NOT use resources here! The finalizer assumes
        # that any existing resources are already finalized, and
        # then we end up skipping all of the finalization steps
        parse_cache = {}

        with finalize_signature(
            signature,
            Outermost,
            preload=None,
            parse_cache=parse_cache
        ) as sig_finalizer:
            for required_template_cls in sig_finalizer.required_loads:
                sig_finalizer.preload[required_template_cls] = \
                    resources[required_template_cls]

        assert sig_finalizer.required_loads == {Outermost, Middle, Innermost}
        assert parse_cache == resources
        assert sig_finalizer.target_resource is resources[Outermost]

    def test_discovers_all_required_cache(self):
        """The returned finalizer must include only the uncached classes
        in the required loads when cached resources are available (and
        no override specified).

        Additionally, cached resources must nonetheless be placed into
        the preload.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: Slot[Middle]

        @template(fake_template_config, object())
        class Middle:
            innermost: Slot[Innermost]

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

        signature = cast(
            type[TemplateIntersectable], Outermost
            )._templatey_signature
        # Note: DO NOT use resources here! The finalizer assumes
        # that any existing resources are already finalized, and
        # then we end up skipping all of the finalization steps
        parse_cache: dict[TemplateClass, ParsedTemplateResource] = {
            Innermost: resources[Innermost]}
        innermost_signature = cast(
            type[TemplateIntersectable], Innermost
            )._templatey_signature
        ensure_recursive_totality(innermost_signature, Innermost)
        ensure_slot_tree(innermost_signature, Innermost)
        ensure_prerender_tree(
            # Note: this can't just use resources; see note re: fragility in
            # distill_prerender_tree
            innermost_signature, {Innermost: resources[Innermost]})

        with finalize_signature(
            signature,
            Outermost,
            preload=None,
            parse_cache=parse_cache
        ) as sig_finalizer:
            for required_template_cls in sig_finalizer.required_loads:
                sig_finalizer.preload[required_template_cls] = \
                    resources[required_template_cls]

        assert sig_finalizer.required_loads == {Outermost, Middle}
        assert parse_cache == resources
        assert sig_finalizer.target_resource is resources[Outermost]
        assert sig_finalizer.preload == resources

    def test_all_loads_fully_finalized(self):
        """After the finalizer context is exited, all of the templates
        must be fully finalized, recursively -- the fieldset, slot tree,
        prerender tree, etc must all be present.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: Slot[Middle]

        @template(fake_template_config, object())
        class Middle:
            innermost: Slot[Innermost]

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

        outermost_signature = cast(
            type[TemplateIntersectable], Outermost
            )._templatey_signature
        middle_signature = cast(
            type[TemplateIntersectable], Middle
            )._templatey_signature
        innermost_signature = cast(
            type[TemplateIntersectable], Innermost
            )._templatey_signature
        # Note: DO NOT use resources here! The finalizer assumes
        # that any existing resources are already finalized, and
        # then we end up skipping all of the finalization steps
        parse_cache = {}

        with finalize_signature(
            outermost_signature,
            Outermost,
            preload=None,
            parse_cache=parse_cache
        ) as sig_finalizer:
            for required_template_cls in sig_finalizer.required_loads:
                sig_finalizer.preload[required_template_cls] = \
                    resources[required_template_cls]

        # These are all effectively hasattr's but with type checking
        assert isinstance(outermost_signature.fieldset, NormalizedFieldset)
        assert isinstance(outermost_signature.total_inclusions, frozenset)
        assert isinstance(outermost_signature.slot_tree, SlotTreeNode)
        assert outermost_signature.prerender_tree is None

        assert isinstance(middle_signature.fieldset, NormalizedFieldset)
        assert isinstance(middle_signature.total_inclusions, frozenset)
        assert isinstance(middle_signature.slot_tree, SlotTreeNode)
        assert middle_signature.prerender_tree is None

        assert isinstance(innermost_signature.fieldset, NormalizedFieldset)
        assert isinstance(innermost_signature.total_inclusions, frozenset)
        assert isinstance(innermost_signature.slot_tree, SlotTreeNode)
        assert innermost_signature.prerender_tree is None


class TestEnsureRecursiveTotality:

    def test_includes_fieldset(self):
        """Recursive totality must also set the fieldset on the
        template class.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: Slot[Middle]

        @template(fake_template_config, object())
        class Middle:
            innermost: Slot[Innermost]

        @template(fake_template_config, object())
        class Innermost:
            var: Var[str]

        outermost_signature = cast(
            type[TemplateIntersectable], Outermost
            )._templatey_signature

        assert not hasattr(outermost_signature, 'fieldset')
        ensure_recursive_totality(outermost_signature, Outermost)

        assert isinstance(outermost_signature.fieldset, NormalizedFieldset)

    def test_actually_recursive(self):
        """Recursive totality must, yknow, actually recursively process
        nested template classes, and not just the outermost one.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: Slot[Middle]

        @template(fake_template_config, object())
        class Middle:
            innermost: Slot[Innermost]

        @template(fake_template_config, object())
        class Innermost:
            var: Var[str]

        outermost_signature = cast(
            type[TemplateIntersectable], Outermost
            )._templatey_signature
        middle_signature = cast(
            type[TemplateIntersectable], Middle
            )._templatey_signature
        innermost_signature = cast(
            type[TemplateIntersectable], Innermost
            )._templatey_signature

        assert not hasattr(outermost_signature, 'total_inclusions')
        assert not hasattr(middle_signature, 'total_inclusions')
        assert not hasattr(innermost_signature, 'total_inclusions')

        ensure_recursive_totality(outermost_signature, Outermost)

        assert isinstance(outermost_signature.total_inclusions, frozenset)
        assert isinstance(middle_signature.total_inclusions, frozenset)
        assert isinstance(innermost_signature.total_inclusions, frozenset)
