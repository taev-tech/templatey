from __future__ import annotations

# import xml.sax.saxutils
from contextvars import ContextVar
from html import escape as html_escape
from html.parser import HTMLParser
from typing import Annotated
from typing import Literal

from docnote import Note

from templatey.exceptions import BlockedContentValue
from templatey.interpolators import NamedInterpolator
from templatey.templates import ParseConfig
from templatey.templates import RenderConfig

ALLOWABLE_HTML_CONTENT_TAGS: ContextVar[set[str]] = ContextVar(
    'ALLOWABLE_HTML_CONTENT_TAGS', default={  # noqa: B039
        'address', 'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hgroup',
        'section', 'blockquote', 'dd', 'div', 'dl', 'dt', 'figcaption',
        'figure', 'hr', 'li', 'menu', 'ol', 'p', 'pre', 'ul', 'a', 'abbr',
        'b', 'bdi', 'bdo', 'br', 'cite', 'data', 'dfn', 'em', 'i', 'kbd',
        'mark', 'q', 'rp', 'rt', 'ruby', 's', 'samp', 'small', 'span',
        'strong', 'sub', 'sup', 'time', 'u', 'var', 'wbr', 'area', 'audio',
        'img', 'map', 'track', 'video', 'embed', 'object', 'picture', 'source',
        'svg', 'math', 'del', 'ins', 'caption', 'col', 'colgroup', 'table',
        'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'button', 'details',
        'dialog', 'summary'})


def noop_escaper(value: str) -> str:
    return value


def noop_verifier(value: str) -> Literal[True]:
    return True


def html_escaper(value: str) -> str:
    return html_escape(value, quote=True)


def html_verifier(value: str) -> Literal[True]:
    parser = _HtmlVerifierParser()
    parser.feed(value)
    parser.close()
    return True


class _HtmlVerifierParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        allowlist = ALLOWABLE_HTML_CONTENT_TAGS.get()
        if tag not in allowlist:
            raise BlockedContentValue(
                'Tag not allowed in HTML content using the current allowlist',
                tag)

    def handle_endtag(self, tag):
        allowlist = ALLOWABLE_HTML_CONTENT_TAGS.get()
        if tag not in allowlist:
            raise BlockedContentValue(
                'Tag not allowed in HTML content using the current allowlist',
                tag)


# I was gonna write XML verifiers/escapers, but then I read that there are
# a bunch of DoS vulnerabilities in the stdlib XML stuff:
# https://docs.python.org/3/library/xml.html#xml-vulnerabilities
# That implies that prebaked should be a subpackage, and each submodule should
# have optional 3rd-party deps
# def xml_escaper(value: str) -> str:
#     return xml.sax.saxutils.escape(value)


trusted_text: Annotated[
        RenderConfig,
        Note('''This prebaked render config includes **no escaping or
            verification**.

            Use this if, and **only if**, you trust all variables and
            content passed to the template!''')
    ] = RenderConfig(
        variable_escaper=noop_escaper,
        content_verifier=noop_verifier)


html: Annotated[
        RenderConfig,
        Note('''This prebaked render config uses a dedicated HTML escaper and
            verifier, with specific allowlisted HTML tags for context.''')
    ] = RenderConfig(
        variable_escaper=html_escaper,
        content_verifier=html_verifier)


unicon: Annotated[
        ParseConfig,
        Note('''This prebaked parse config uses unicode control characters as
            the interpolator. Use it if you need to use curly braces within the
            template text. One example use case would be using a template as a
            CSS preprocessor.''')
    ] = ParseConfig(interpolator=NamedInterpolator.UNICODE_CONTROL)
