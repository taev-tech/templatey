from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from textwrap import dedent

import pytest

from templatey import Content
from templatey import Slot
from templatey import Var
from templatey import param
from templatey import template
from templatey.environments import RenderEnvironment
from templatey.exceptions import OvercomplicatedSlotTree
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.loaders import InlineStringTemplateLoader
from templatey.prebaked.template_configs import html

_loader = InlineStringTemplateLoader()
type HtmlTemplate = (
    HtmlGenericElement
    | PlaintextTemplate
    | ModuleSummaryTemplate
    | VariableSummaryTemplate
    | ClassSummaryTemplate
    | CallableSummaryTemplate
    | TypespecTemplate
    | SignatureSummaryTemplate
    | ParamSummaryTemplate
    | RetvalSummaryTemplate)


@template(
    html,
    '<{content.tag}{slot.attrs: __prefix__=" "}>{slot.body}</{content.tag}>',
    loader=_loader,
    kw_only=True)
class HtmlGenericElement:
    tag: Content[str]
    attrs: Slot[HtmlAttr] = field(default_factory=list)
    # Pyright has an intermittent bug with not recognizing htmltemplate as
    # a dataclass instance. This probably has something to do with our
    # missing intersection type.
    body: Slot[HtmlTemplate]  # type: ignore


@template(html, '{content.key}="{var.value}"', loader=_loader)
class HtmlAttr:
    key: Content[str]
    value: Var[str]


@template(html, '{var.text}', loader=_loader)
class PlaintextTemplate:
    text: Var[str]


@template(
    html,
    dedent('''\
        <article>
        <h2>{var.fullname}</h2>
        <header>
            <section>{slot.docstring}</section>
            <section><ul>{slot.dunder_all}</ul></section>
        </header>
        {slot.members}
        </article>
        '''),
    loader=_loader)
class ModuleSummaryTemplate:
    fullname: Var[str]
    docstring: Slot[HtmlTemplate]  # type: ignore
    dunder_all: Slot[HtmlGenericElement]
    members: Slot[HtmlTemplate]  # type: ignore


@template(
    html,
    dedent('''\
        <details>
            <summary>
                <hgroup>
                    <h3>{var.name}</h3>
                    <p>{slot.typespec}</p>
                </hgroup>
            </summary>
            {slot.notes}
        </details>
        '''),
    loader=_loader)
class VariableSummaryTemplate:
    name: Var[str]
    typespec: Slot[HtmlTemplate]  # type: ignore
    notes: Slot[HtmlTemplate]  # type: ignore


@dataclass(slots=True)
class _CrossrefSummaryTemplateBase:
    qualname: Var[str]
    shortname: Var[str]
    traversals: Var[str | None] = field(default=None, kw_only=True)

    has_traversals: Content[bool] = param(  # noqa: RUF009
        init=False, prerenderer=lambda value: '<...>' if value else None)

    def __post_init__(self):
        self.has_traversals = (self.traversals is not None)


@template(
    html,
    dedent('''\
        <abbr title="{var.qualname}{var.traversals}">
            <a href="{var.target}">{var.shortname}{content.has_traversals}</a>
        </abbr>
        '''),
    loader=_loader)
class LinkableCrossrefSummaryTemplate(_CrossrefSummaryTemplateBase):
    # Note: the error here is because it's not understanding our use of
    # ``param()`` in the base class
    target: Var[str]  # type: ignore


@template(
    html,
    dedent('''\
        <abbr title="{var.qualname}{var.traversals}">
            {var.shortname}{content.has_traversals}
        </abbr>
        '''),
    loader=_loader)
class UnlinkableCrossrefSummaryTemplate(_CrossrefSummaryTemplateBase):
    """This just inherits from the summary template base.
    """


type CrossrefSummaryTemplate = (
    LinkableCrossrefSummaryTemplate | UnlinkableCrossrefSummaryTemplate)


@template(
    html,
    dedent('''\
        <details open>
            <summary>
                <h3>{var.name}</h3>
                <p>{slot.metaclass}</p>
            </summary>
            <header>
                <section><ol>{
                    slot.bases:
                    __prefix__='<li>',
                    __suffix__='</li>'}</ol></section>
                <section>{slot.docstring}</section>
            </header>
            {slot.members}
        </details>
        '''),
    loader=_loader)
class ClassSummaryTemplate:
    name: Var[str]
    metaclass: Slot[NormalizedConcreteTypeTemplate]
    bases: Slot[NormalizedConcreteTypeTemplate]
    docstring: Slot[HtmlTemplate]  # type: ignore
    members: Slot[HtmlTemplate]  # type: ignore


@template(
    html,
    dedent('''\
        <details>
            <summary>
                <h3>{var.name}</h3>
                <span>{content.color}</span>
                <span>{content.method_type}</span>
                <span>{content.is_generator}</span>
            </summary>
            <header>{slot.docstring}</section></header>
            {slot.signatures}
        </details>
        '''),
    loader=_loader)
class CallableSummaryTemplate:
    name: Var[str]
    docstring: Slot[HtmlTemplate]  # type: ignore

    color: Content[str]
    method_type: Content[str | None]
    is_generator: Content[bool]

    signatures: Slot[HtmlTemplate]  # type: ignore


type NormalizedTypeTemplate = (
    NormalizedUnionTypeTemplate
    | NormalizedEmptyGenericTypeTemplate
    | NormalizedConcreteTypeTemplate
    | NormalizedSpecialTypeTemplate
    | NormalizedLiteralTypeTemplate)


@template(
    html,
    '<dt>{content.key}</dt><dd>{content.value}</dd>',
    loader=_loader)
class TypespecTagTemplate:
    key: Content[str]
    value: Content[bool]


@template(
    html,
    dedent('''\
        <ul data-docnote-component="typespec">
            <dl data-docnote-component="typespec.tags">{slot.tags}</dl>
            {slot.normtype: __prefix__='<li>', __suffix__='</li>'}
        </ul>'''),
    loader=_loader)
class TypespecTemplate:
    normtype: Slot[NormalizedTypeTemplate]
    tags: Slot[TypespecTagTemplate]


@template(
    html,
    dedent('''\
        <ul data-docnote-component="typespec.union">{
            slot.normtypes:
            __prefix__='<li>',
            __suffix__='</li>'
        }</ul>'''),
    loader=_loader)
class NormalizedUnionTypeTemplate:
    normtypes: Slot[NormalizedTypeTemplate]


@template(
    html,
    dedent('''\
        <div data-docnote-component="typespec.concrete.primary">{
            slot.primary}</div>
        <ol data-docnote-component="typespec.concrete.params">{
            slot.params:
            __prefix__='<li>',
            __suffix__='</li>'}</ol>'''),
    loader=_loader)
class NormalizedConcreteTypeTemplate:
    primary: Slot[CrossrefSummaryTemplate]
    params: Slot[TypespecTemplate]


@template(
    html,
    dedent('''\
        <ol data-docnote-component="typespec.empty-generic.params">{
            slot.params:
            __prefix__='<li>',
            __suffix__='</li>'}</ol>'''),
    loader=_loader)
class NormalizedEmptyGenericTypeTemplate:
    params: Slot[TypespecTemplate]


@template(
    html,
    dedent('''\
        <div data-docnote-component="typespec.special-form">{
            slot.type_}</div>'''),
    loader=_loader)
class NormalizedSpecialTypeTemplate:
    type_: Slot[CrossrefSummaryTemplate]


@template(
    html,
    dedent('''\
        <ul data-docnote-component="typespec.literal.values">{
            slot.values:
            __prefix__='<li>',
            __suffix__='</li>'}
        </ul>'''),
    loader=_loader)
class NormalizedLiteralTypeTemplate:
    values: Slot[HtmlGenericElement | CrossrefSummaryTemplate]


@template(
    html,
    dedent('''\
        <section data-docnote-component="signature">
            <div data-docnote-component="signature.params">
                <dl data-docnote-component="signature.params.pos-only">{
                    slot.params_pos_only}
                </dl>
                <dl data-docnote-component="signature.params.pos-or-kw">{
                    slot.params_pos_or_kw}
                </dl>
                <dl data-docnote-component="signature.params.pos-starred">{
                    slot.params_pos_starred}
                </dl>
                <dl data-docnote-component="signature.params.kw-only">{
                    slot.params_kw_only}
                </dl>
                <dl data-docnote-component="signature.params.kw-starred">{
                    slot.params_kw_starred}
                </dl>
            </div>
            <div data-docnote-component="signature.retval">{slot.retval}</div>
            <div data-docnote-component="signature.docstring">{
                slot.docstring}</div>
        </section>
        '''),
    loader=_loader)
class SignatureSummaryTemplate:
    # Note: all of the htmltemplates here are because of the way we've done
    # types in the _dispatch_transform
    params_pos_only: Slot[HtmlTemplate | ParamSummaryTemplate]
    params_pos_or_kw: Slot[HtmlTemplate | ParamSummaryTemplate]
    params_pos_starred: Slot[HtmlTemplate | ParamSummaryTemplate]
    params_kw_only: Slot[HtmlTemplate | ParamSummaryTemplate]
    params_kw_starred: Slot[HtmlTemplate | ParamSummaryTemplate]
    retval: Slot[HtmlTemplate | RetvalSummaryTemplate]
    docstring: Slot[HtmlTemplate]  # type: ignore


@template(
    html,
    dedent('''\
        <div data-docnote-component="param">
            <dt>{var.name}</dt>
            <dd>{slot.typespec}</dd>
            <dd>{var.default}</dd>
            <dd>{slot.notes}</dd>
        </div>
        '''),
    loader=_loader)
class ParamSummaryTemplate:
    name: Var[str]
    typespec: Slot[HtmlTemplate]  # type: ignore
    default: Var[str | None]
    notes: Slot[HtmlTemplate]  # type: ignore


@template(
    html,
    dedent('''\
        <div data-docnote-component="retval">
            <div>{slot.typespec}</div>\
            <div>{slot.notes}</div>
        </div>
        '''),
    loader=_loader)
class RetvalSummaryTemplate:
    typespec: Slot[HtmlTemplate]  # type: ignore
    notes: Slot[HtmlTemplate]  # type: ignore


class TestFinalization:
    """This is basically one big stress test for finalization.
    """

    def test_finalization_via_load(self):
        """Loading templates must not cause unreasonable
        delays for overcomplicated template trees, but instead raise.
        """
        render_env = RenderEnvironment(
            env_functions=(),
            template_loader=DictTemplateLoader(
                templates={}),
            slot_tree_complexity_limiter=True)

        before = time.monotonic()
        with pytest.raises(OvercomplicatedSlotTree):
            render_env.load_sync(ModuleSummaryTemplate)
        after = time.monotonic()

        # This is an arbitrary threshold, but... seems reasonable. And by
        # reasonable I mean, extremely slow in production, but not flaky
        # during testing.
        assert (after - before) < .5
