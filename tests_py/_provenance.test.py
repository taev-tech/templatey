from __future__ import annotations

import pytest

from templatey._error_collector import ErrorCollector
from templatey._provenance import Provenance
from templatey._provenance import ProvenanceNode
from templatey._provenance import _recursively_coerce_func_execution_params
from templatey._types import Content
from templatey._types import Var
from templatey.parser import TemplateInstanceContentRef
from templatey.parser import TemplateInstanceDataRef
from templatey.parser import TemplateInstanceVariableRef
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestRecursivelyCoerceFuncExecutionParams:
    """_recursively_coerce_func_execution_params()"""

    def test_int(self):
        """Integers must not break things, and must be returned
        unchanged.
        """
        ec = ErrorCollector()
        preload = {}
        provenance = Provenance()

        retval = _recursively_coerce_func_execution_params(
            42,
            provenance=provenance,
            template_preload=preload,
            error_collector=ec)
        assert retval == 42

    def test_string(self):
        """Strings must return the string unchanged. In particular, they
        must not be expanded into a list of substrings, each one char
        long!
        """
        ec = ErrorCollector()
        preload = {}
        provenance = Provenance()

        retval = _recursively_coerce_func_execution_params(
            'foo',
            provenance=provenance,
            template_preload=preload,
            error_collector=ec)
        assert retval == 'foo'

    def test_list_of_strings(self):
        """List of strings must also be returned unchanged, other than
        being coerced into a tuple.
        """
        ec = ErrorCollector()
        preload = {}
        provenance = Provenance()

        retval = _recursively_coerce_func_execution_params(
            ['foo', 'bar'],
            provenance=provenance,
            template_preload=preload,
            error_collector=ec)
        assert retval == ('foo', 'bar')

    def test_dict_of_strings(self):
        """Dict of strings must also be returned unchanged
        """
        ec = ErrorCollector()
        preload = {}
        provenance = Provenance()

        retval = _recursively_coerce_func_execution_params(
            {'foo': 'oof', 'bar': 'rab'},
            provenance=provenance,
            template_preload=preload,
            error_collector=ec)
        assert retval == {'foo': 'oof', 'bar': 'rab'}

    @pytest.mark.parametrize(
        'before,expected_after',
        [
            (TemplateInstanceDataRef('data1'), '1data'),
            ([TemplateInstanceDataRef('data1')], ('1data',)),
            ({'foo': TemplateInstanceDataRef('data1')}, {'foo': '1data'}),
            (['beep', TemplateInstanceDataRef('data1')], ('beep', '1data')),
            (TemplateInstanceContentRef('foo'), 'oof'),
            ([TemplateInstanceContentRef('foo')], ('oof',)),
            ({'foo': TemplateInstanceContentRef('foo')}, {'foo': 'oof'}),
            (['beep', TemplateInstanceContentRef('foo')], ('beep', 'oof')),
            (TemplateInstanceVariableRef('bar'), 'rab'),
            ([TemplateInstanceVariableRef('bar')], ('rab',)),
            ({'bar': TemplateInstanceVariableRef('bar')}, {'bar': 'rab'}),
            (['beep', TemplateInstanceVariableRef('bar')], ('beep', 'rab')),
            (
                [
                    TemplateInstanceContentRef('foo'),
                    TemplateInstanceVariableRef('bar')],
                ('oof', 'rab'))])
    def test_recursive_nested_reference(self, before, expected_after):
        """``TemplateInstanceContentRef``s and
        ``TemplateInstanceVariableRef``s,
        including those nested inside collections, must correctly be
        coerced (dereferenced).
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            data1: str
            foo: Content[str]
            bar: Var[str]

        ec = ErrorCollector()
        preload = {}
        fake_instance = FakeTemplate(
            data1='1data',
            foo='oof',
            bar='rab')
        provenance = Provenance(slotpath=(ProvenanceNode(
            encloser_slot_key='',
            encloser_slot_index=-1,
            encloser_part_index=-1,
            instance_id=123,
            instance=fake_instance),))

        retval = _recursively_coerce_func_execution_params(
            before,
            provenance=provenance,
            template_preload=preload,
            error_collector=ec)
        assert retval == expected_after
