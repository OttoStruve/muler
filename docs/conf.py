# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))
# sys.path.append(os.path.join(os.path.dirname(__name__), ".."))

import muler


# -- Project information -----------------------------------------------------

project = "muler"
copyright = "2021, gully"
author = "gully"

# The full version, including alpha/beta/rc tags
release = "0.3.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx_gallery.load_style",
    "numpydoc",
]

autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]
nbsphinx_timeout = 60

# Execute notebooks? Possible values: 'always', 'never', 'auto' (default)
nbsphinx_execute = "always"

autosummary_generate = True
html_show_sourcelink = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "vs"

nbsphinx_thumbnails = {
    "tutorials/refined_sky_subtraction": "_static/hpf_logo_short.png",
    "tutorials/Deblazing_HPF_spectra": "_static/hpf_logo_short.png",
    "tutorials/telluric_correction_with_gollum": "_static/hpf_logo_short.png",
    "tutorials/blank_sky_observations": "_static/hpf_logo_short.png",
    "tutorials/Combine_spectra_misaligned": "_static/hpf_logo_short.png",
    "tutorials/All_spectral_orders": "_static/IGRINS_logo.png",
    "tutorials/Download_IGRINS_data_from_GoogleDrive": "_static/IGRINS_logo.png",
    "tutorials/IGRINS_SpecList_demo": "_static/IGRINS_logo.png",
    "tutorials/Masking_and_flattening": "_static/hpf_logo_short.png",
    "tutorials/Masking_and_flattening_plus": "_static/hpf_logo_short.png",
    "tutorials/Flattening_with_savgol": "_static/hpf_logo_short.png",
    "tutorials/Combining_uncertainties_with_specutils": "_static/error_propagation.png",
    "tutorials/Download_IGRINS_data_from_Box": "_static/IGRINS_logo.png",
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_material"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Set link name generated in the top bar.
html_title = "Home"

nbsphinx_codecell_lexer = "python3"
nbsphinx_allow_errors = True

# Material theme options (see theme.conf for more information)
html_theme_options = {
    "base_url": "https://muler.readthedocs.io/",
    "nav_title": "μler docs",
    "nav_links": [
        {"title": "Quickstart", "href": "quickstart", "internal": True},
        {"title": "Installation", "href": "install", "internal": True},
        {"title": "API", "href": "api", "internal": True},
        {"title": "Tutorials", "href": "tutorials/index", "internal": True},
    ],
    "color_primary": "orange",
    "color_accent": "amber",
    "theme_color": "d35400",
    "repo_url": "https://github.com/OttoStruve/muler/",
    "repo_name": "muler",
    "repo_type": "github",
    "master_doc": True,
    "globaltoc_depth": 2,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
    "logo_icon": "&#xE85C",
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}


html_use_index = True
html_domain_indices = True

nbsphinx_kernel_name = "python3"
