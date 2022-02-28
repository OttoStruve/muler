# `muler` (_μler_)

### version 0.3.0

<a href="https://muler.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Read-the%20docs-blue"></a>
<a href="https://pypi.org/project/muler/"><img src="https://img.shields.io/badge/pip_install-muler-9b59b6"></a>
<a href="https://anaconda.org/conda-forge/muler"><img src="https://img.shields.io/badge/conda%20install%20--c%20conda--forge-muler-9b59b6"></a>


<a href="https://sites.google.com/site/igrinsatgemini/"><img src="https://img.shields.io/badge/Works_with-IGRINS-brightgreen"></a>
<a href="https://hpf.psu.edu/"><img src="https://img.shields.io/badge/Works_with-HPF-brightgreen"></a>  
<a href="https://www2.keck.hawaii.edu/inst/nirspec/"><img src="https://img.shields.io/badge/Works_with-Keck_NIRSPEC-brightgreen"></a>
![example workflow](https://github.com/OttoStruve/muler/actions/workflows/muler-tests.yml/badge.svg)

A Python package for analyzing pipeline-processed data from high resolution near-infrared echelle spectrographs.

### The echelle spectrum problem

Imagine you have just received data from the an echelle spectrograph like [IGRINS](https://www.as.utexas.edu/astronomy/research/people/jaffe/igrins.html) or [HPF](https://hpf.psu.edu/) and you want to start science. Oftentimes you will get handed pipeline-reduced data from the observatory facility. When you examine the data you may notice some remaining instrumental signals: [telluric contamination](https://en.wikipedia.org/wiki/Telluric_contamination) artifact and instrument-induced slopes stand out. Or maybe you want to apply a [barycentric correction](https://sites.psu.edu/astrowright/2014/09/16/barycentric-corrections-at-1-mms/) based on telescope pointing information, but you're not sure which [FITS header](https://docs.astropy.org/en/stable/io/fits/usage/headers.html) columns to use when multiple are available. You may want to normalize, deblaze (_a.k.a._ flatten), and plot the spectrum, or populate the spectrum into a [pandas](https://pandas.pydata.org/docs/user_guide/index.html) dataframe, or estimate a coarse radial velocity based on a noise-free stellar [model atmosphere](https://en.wikipedia.org/wiki/Model_photosphere) or a standard star template. Or maybe you want to measure an equivalent width, with or without an error bar. All of these operations are relatively routine, but their application, order, and algorithm choice may depend on the science case, and therefore they cannot be built into a default pipeline: it is up to you---the scientist---to conduct these activities.

Muler is intended to reduce the friction of getting started with near-IR echelle spectroscopy. Typical spectral analyses become only 2 or 3 lines of compact `muler` code, rather than hundreds of cells of Jupyter notebooks just to load and post-process a spectrum.

Plotting a sky-subtracted, flattened spectrum:

```Python
spectrum = HPFSpectrum(file=file, order=15)
spectrum.remove_nans().sky_subtract().deblaze().normalize().plot()
```

Measuring an [equivalent width](https://en.wikipedia.org/wiki/Equivalent_width):

```Python
spectrum = HPFSpectrum(file=file, order=15)
clean_spectrum = spectrum.remove_nans().sky_subtract().deblaze().normalize()
ew = clean_spectrum.measure_ew()
```

### Installation: `pip` and development version

We currently offer seamless installation with `pip` and `conda`! You can install `muler` in one line with:

```bash
pip install muler
```
or 
```bash
conda install -c conda-forge muler
```

`muler` constantly changes and benefits from new community contributions like yours. We therefore recommend the slightly more tedious installation from the raw source code described on our [Installation webpage](https://muler.readthedocs.io/en/latest/install.html). Installing from source code empowers you to modify the code for your purposes.

### Our mission and your help

`muler` is aimed at collecting common spectral analysis methods into one place, simplifying the interactive process of astronomical discovery. We want to reduce the perceived complexity of working with echelle spectra, so that scientists have more time to focus on frontier science problems. We aspire to become a tool that new and experienced astronomers alike come to use and rely upon. In order to achieve these goals **we need community contributions from practitioners like you.**

That help can take many forms, even more creative outlets than most would guess. A great and easy first step is to :star: our repo or subscribe to notifications so that you can stay up to date as our project evolves. Glance around our [tutorials](https://muler.readthedocs.io/en/latest/tutorials/index.html), overall [documentation](https://muler.readthedocs.io/en/latest/), and [Issues](https://github.com/OttoStruve/muler/issues) pages. Join our welcoming community and help us build some cool tools to make astronomy slightly easier, more reproducible, and more fun.

### Spectrographs we currently support

We currently support the [IGRINS](https://www.as.utexas.edu/astronomy/research/people/jaffe/igrins.html), [HPF](https://hpf.psu.edu/), and [NIRSPEC](https://www2.keck.hawaii.edu/inst/nirspec/) spectrographs. These three near-IR echelle spectrographs are attached to some of the greatest telescopes in the world: [Gemini](https://www.gemini.edu/), [HET](https://mcdonaldobservatory.org/research/telescopes/HET), and [Keck](https://www.keckobservatory.org/) respectively. HPF differs from IGRINS and NIRSPEC in its _precision radial velocity_ (PRV) design. It uses fibers instead of slits. We do not purport to address the extreme precision demands of the PRV community, but we anticipate our package is still useful to a wide range of HPF science use cases. We are open to supporting new spectrographs in the future, but at the present time we are focused on building, testing, and maintaining the core features for these three spectrographs before we add new ones. One way to get a new instrument supported is to make a new GitHub [Issue](https://github.com/OttoStruve/muler/issues) to describe the instrument and rationale for adding it to `muler`, so other community members can join the conversation.
