# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'Linux 体系和编程实验指导'
copyright = '2026'
author = ''

extensions = [
    'myst_parser',  # Markdown support
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
