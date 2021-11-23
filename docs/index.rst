.. muler documentation master file, created by
   sphinx-quickstart on Wed Jan  6 09:42:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

μler's documentation
====================

**μl·er**
`muler`
/myo͞olər/
*noun*

1. a portmanteau of *micron* (μm) and *ruler*
2. a Python Package for IGRINS and HPF data analysis

`muler` aims to simplify astronomical echelle spectroscopic data analysis through abstraction of common astronomy tasks.  The project structure allows users to customize these routine tasks for their science purpose, while providing sane defaults that will work for a broad range of use cases.  The architecture is based loosely on the `lightkurve <https://docs.lightkurve.org/>`_ framework, adapted to echelle spectroscopy data analysis.  We currently support the  `IGRINS <https://www.as.utexas.edu/astronomy/research/people/jaffe/igrins.html>`_ and `HPF <https://hpf.psu.edu/>`_ spectrographs.  The project is currently in early stages of development, we hope you will provide community input on our `GitHub Issues page <https://www.github.com/OttoStruve/muler/issues>`_.

`muler` will make it easy to chain routine operations in sequence:

.. code-block:: python

  spectrum = HPFSpectrum(file=file, order=15)
  spectrum.remove_nans().sky_subtract().blaze_divide_spline().normalize().plot()



.. toctree::
   :hidden:
   :caption: Getting Started
   :maxdepth: 1

   Installation <install>
   Quickstart <quickstart.ipynb>
   Python Spectroscopy Ecosystem <broader_ecosystem.rst>


.. toctree::
   :hidden:
   :caption: Tutorials
   :maxdepth: 3

   Tutorials <tutorials/index>


.. toctree::
   :hidden:
   :caption: API
   :maxdepth: 1

   Application Programming Interface <api>
