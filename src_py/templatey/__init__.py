from docnote import DocnoteConfig
from docnote import MarkupLang

import templatey.prebaked as prebaked  # noqa: PLR0402
from templatey._types import Content
from templatey._types import DynamicClassSlot
from templatey._types import Slot
from templatey._types import TemplateClass
from templatey._types import TemplateClassInstance
from templatey._types import Var
from templatey.environments import RenderEnvironment
from templatey.templates import ComplexContent
from templatey.templates import InjectedValue
from templatey.templates import TemplateConfig  #  type: ignore  # noqa: F401
from templatey.templates import TemplateFieldConfig
from templatey.templates import TemplateParseConfig
from templatey.templates import TemplateRenderConfig
from templatey.templates import TemplateResourceConfig
from templatey.templates import anchor_closure_scope
from templatey.templates import param  #  type: ignore  # noqa: F401
from templatey.templates import template  #  type: ignore  # noqa: F401
from templatey.templates import template_field  #  type: ignore  # noqa: F401

__all__ = [
    'ComplexContent',
    'Content',
    'DynamicClassSlot',
    'InjectedValue',
    'RenderEnvironment',
    'Slot',
    'TemplateClass',
    'TemplateClassInstance',
    'TemplateFieldConfig',
    'TemplateParseConfig',
    'TemplateRenderConfig',
    'TemplateResourceConfig',
    'Var',
    'anchor_closure_scope',
    'prebaked',
]


DOCNOTE_CONFIG = DocnoteConfig(markup_lang=MarkupLang.CLEANCOPY)
