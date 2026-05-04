from fvgp import __version__

project = 'fvGP'
copyright = '2024, Marcus Michael Noack'
author = 'Marcus Michael Noack'
version = __version__
release = __version__

extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

nb_execution_mode = 'off'

myst_enable_extensions = ['colon_fence']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'logo': {'text': 'fvGP'},
    'github_url': 'https://github.com/lbl-camera/fvgp',
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['navbar-icon-links'],
    'secondary_sidebar_items': ['page-toc'],
    'footer_start': ['copyright'],
    'footer_end': [],
}

html_static_path = ['_static']
html_css_files = ['custom.css']

autodoc_member_order = 'bysource'
autoclass_content = 'both'
