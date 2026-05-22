import os
import sys

# -- Project information -----------------------------------------------------
project = 'MASDiff'
copyright = '2024, MASDiff Authors'
author = 'MASDiff Authors'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []
language = 'zh_CN'

# MyST-Parser settings (enables Markdown)
myst_enable_extensions = [
    'dollarmath',   # $...$ and $$...$$
    'amsmath',
    'colon_fence',
    'deflist',
    'html_image',
]
myst_dmath_double_inline = True

# Mermaid
mermaid_version = "10.6.1"

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

html_css_files = ['custom.css']

# -- Source suffix -----------------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'
