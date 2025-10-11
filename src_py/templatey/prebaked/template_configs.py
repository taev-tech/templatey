"""Deprecated. Use prebaked.configs instead.
"""
from __future__ import annotations

from typing import Annotated

from docnote import ClcNote
from docnote import DocnoteConfig

from templatey.interpolators import NamedInterpolator
from templatey.prebaked.configs import html_escaper
from templatey.prebaked.configs import html_verifier
from templatey.prebaked.configs import noop_escaper
from templatey.prebaked.configs import noop_verifier
from templatey.templates import TemplateConfig

# This whole thing is deprecated; use prebaked.configs instead!
DOCNOTE_CONFIG = DocnoteConfig(include_in_docs=False)


html: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses curly braces as the interpolator
        along with a dedicated HTML escaper and verifier, with specific
        allowlisted HTML tags for context.

        Use this if you need to write HTML templates that don't inline
        javascript, CSS, etc.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.CURLY_BRACES,
    variable_escaper=html_escaper,
    content_verifier=html_verifier)


html_unicon: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses unicode control characters as the
        interpolator along with a dedicated HTML escaper and verifier, with
        specific allowlisted HTML tags for context.

        Use this if you need to write HTML templates that make use of curly
        braces within the literal template definition -- for example, if
        your template text contains inline javascript, CSS, etc.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.UNICODE_CONTROL,
    variable_escaper=html_escaper,
    content_verifier=html_verifier)


trusted: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses curly brackets as the
        interpolator, but includes **no escaping or verification**.

        Use this:
        ++  if, and **only if**, you trust all variables and content passed
            to the template
        ++  if you don't need to use curly braces within the template itself

        One example use case would be using a template for plaintext.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.CURLY_BRACES,
    variable_escaper=noop_escaper,
    content_verifier=noop_verifier)


trusted_unicon: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses unicode control characters as the
        interpolator, but includes **no escaping or verification**.

        Use this:
        ++  if, and **only if**, you trust all variables and content passed
            to the template
        ++  if you need to use curly braces within the template itself

        One example use case would be using a template as a CSS preprocessor.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.UNICODE_CONTROL,
    variable_escaper=noop_escaper,
    content_verifier=noop_verifier)
