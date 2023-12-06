.. _broadereco:

*****************************
Python Spectroscopy Ecosystem
*****************************

muler is part of an ever-evolving ecosystem of Python packages for astronomical spectroscopy.  Here's how `muler` fits in.

sibling packages
----------------

`muler` has a closely-related sibling package named `gollum <https://gollum-astro.readthedocs.io>`_.  `muler` is the interface to *reduced data* and `gollum` is the interface to precomputed synthetic spectral models.  The main reason to separate these packages is to keep them smaller and more maintainable.  They share some common methods, like `normalize()` or `flatten()`, but overall the types of operations you perform on a model spectrum are different than those of a data spectrum. For one, a data spectrum has uncertainty whereas a model spectrum does not.  

parent packages
---------------

`muler`'s parent package is `specutils <https://specutils.readthedocs.io/en/stable/>`_.  `specutils` provides lots of the foundation for spectroscopy tasks, such as units, spectral axes, radial velocity, resampling, trimming, basic model fitting, among others.  To some extent, `muler` can be seen as a `fluent interface <https://en.wikipedia.org/wiki/Fluent_interface>`_ wrapper around specutils.  We considered alternatives to specutils, such as pandas DataFrames, but in the end specutils appears to be the right choice.

The parent of specutils is `astropy <https://www.astropy.org>`_, which I suppose makes astropy the grandparent of `muler`.  

The data that muler consumes comes from data reduction pipelines.  We do not inherit any code from these pipelines and simply treat their products as standing alone, with fixed formatting.  We currently support HPF data from `Goldilocks <https://github.com/grzeimann/Goldilocks_Documentation>`_ and the closed-source HPF instrument team.  We support IGRINS data from the `plp <https://github.com/igrins/plp>`_, and we have preliminary support for data from the Keck NIRSPEC `NSDRP <https://www2.keck.hawaii.edu/koa/nsdrp/nsdrp.html>`_.


community packages
------------------

There are dozens, maybe hundreds of actively maintained open source Python packages for astronomical spectroscopy.  Here we provide an incomplete curated list of some packages in this broader ecosystem.  muler has inherited ideas and code from some of these, and muler can serve as a pre-processing step to others, so this list can be thought of as a loosely connected family of tools.  We hope to see those interconnections bloom as existing projects grow and new ones sprout.

Data-model comparison
=====================

* `starfish <https://starfish.readthedocs.io>`_ - Compare data and pre-computed synthetic models with robust likelihood-based inference and PCA-based spectral emulation
* `blasé <https://blase.readthedocs.io>`_ - Clone precomputed synthetic models with PyTorch and Autodiff
* `specmatch-emp <https://github.com/samuelyeewl/specmatch-emp>`_ - Empirical spectral template matching for visible echelle spectra
* `The Payne <https://github.com/pacargile/ThePayne>`_ - Artificial Neural-Net compression and fitting of synthetic spectral grids 


Data-driven methods
===================
* `wobble <https://wobble.readthedocs.io/>`_ - Automatically isolate bulk RV, stellar, and telluric lines from multiepoch spectra with large barycentric variation
* `psoap <https://psoap.readthedocs.io/>`_ - Isolate underlying stellar templates from composite spectra undergoing orbital motion
* `The Cannon <https://github.com/annayqho/TheCannon>`_ - Data driven spectral labels from many examples


Retrievals and multipurpose
===========================
* `exojax <https://github.com/HajimeKawahara/exojax>`_ - Atmospheric retrieval framework built on JAX, aimed at brown dwarfs and exoplanets
* `species <https://species.readthedocs.io/>`_ - Provides access to PHOENIX models among others, data access, retrieval modeling
* `splat <https://splat.physics.ucsd.edu/splat/>`_ - SpeX Prism Library Archive Toolkit, access to data from the popular IRTF instrument
* `pysynphot <https://pysynphot.readthedocs.io/en/latest/>`_ - Synthetic photometry and access to precomputed models
* `radis <https://radis.readthedocs.io/en/latest/>`_ - A fast line-by-line code for infrared molecular opacities


Telluric correction
===================
* `telfit <https://telfit.readthedocs.io/en/latest/>`_ - Wrapper to the line-by-line radiative transfer model for Earth's atmospheric absorption

Radial velocity
===============
* `radvel <https://radvel.readthedocs.io/en/latest/>`_ - Radial Velocity fitting
* `exoplanet <https://docs.exoplanet.codes/en/latest/>`_ - Exoplanet RV radial velocity orbit fitting and more
* `doppler <https://doppler.readthedocs.io/en/latest/>`_ - Derive RV from an input spectrum 
* `igrins_rv <https://github.com/shihyuntang/igrins_rv>`_ - Radial velocity analysis specific to IGRINS
* `eniric <https://github.com/jason-neal/eniric>`_ - Extended Near InfraRed spectra Information Content analysis
* `sapphires <https://github.com/tofflemire/saphires>`_ - Stellar Analysis in Python for HIgh REsolution Spectroscopy

Doppler Imaging
===============
* `starry <https://github.com/rodluger/starry>`_ - Probabilistic Doppler Imaging

Abundances / Correlations
==========================
* `zaspe <https://github.com/rabrahm/zaspe>`_ - A Code to Measure Stellar Atmospheric Parameters and their Covariance from Spectra
* `Chem-I-Calc <https://chem-i-calc.readthedocs.io>`_ - Chemical information calculator
* `smhr <https://github.com/andycasey/smhr>`_ - Spectroscopy Made Harder
* `turbospectrum <https://github.com/bertrandplez/Turbospectrum2019>`_ - Synthetic spectrum calculation companion to MARCS
