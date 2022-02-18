---
title: 'Astronomical échelle spectroscopy data analysis with `muler`'
tags:
  - Python
  - astronomy
  - spectroscopy
  - stars
  - echelle
authors:
  - name: Michael A. Gully-Santiago
    orcid: 0000-0002-4020-3457
    affiliation: 1
  - name: Jessica L. Luna # note this makes a footnote saying 'co-first author'
    orcid: 0000-0003-2152-9248
    affiliation: 1
  - name: Caroline V. Morley
    orcid: 0000-0002-4404-0456
    affiliation: 1
  - name: Kyle Kaplan
    orcid: 0000-0001-6909-3856
    affiliation: 2
  - name: Aishwarya Ganesh
    orcid: 0000-0002-1846-196X
    affiliation: 1
  - name: Joel Burke
    affiliation: 1
  - name: Daniel M. Krolikowski
    orcid: 0000-0001-9626-0613
    affiliation: 1
affiliations:
 - name: The University of Texas at Austin Department of Astronomy, Austin, TX, USA
   index: 1
 - name: SOFIA Science Center, Universities Space Research Association, NASA Ames Research Center, Moffett Field, CA, USA
   index: 2
date: 14 February 2022
bibliography: paper.bib

---


# Summary

Modern astronomical échelle spectrographs produce information-rich 2D echellograms that undergo standard reduction procedures to produce extracted 1D spectra. The final post-processing steps of these 1D spectra are often left to the end-user scientists, since the order of operations and algorithm choice may depend on the scientific application. Implementing these post-processing steps from scatch acts as a barrier to entry to newcomers, taxes scientific innovation, and erodes scientific reproducibility. Here we assemble and streamline a collection of spectroscopic data analysis methods into a single easy-to-use Application Programming Interface (API), `muler`. The `specutils`-based fluent interface enables method chaining that yields compact 1-line code for many applications, a dramatic reduction in complexity compared to existing industry practices. When applicable, some algorithms are customized to the pipeline data products of three near-infrared spectrographs: HPF, Keck NIRSPEC, and IGRINS. The framework could be extended to other spectrographs in the future. The tutorials in `muler` may also serve as a central onboarding point for new entrants to the practice of near-infrared échelle spectroscopy data. The open-source permissively licensed Python 3 implementation lowers the barrier to entry and accelerates the investigation of astronomical spectra. 

# Statement of need

In its most elemental form, an astronomical spectrum consists of a list of wavelength coordinates and their flux values. Additional metadata for uncertainties, date, time, sky pointing coordinates, and instrumental configuration are common. The Astropy-affiliated `specutils` package serves as a unifying container for virtually any astronomical spectrum, by providing a way to represent, load, manipulate, and analyze these data and metadata. It is therefore possible for each new practitioner to load a pipeline-produced output spectrum with, `numpy` arrays, `pandas` dataframes, or `specutils` objects and write standard post-processing operations from scratch. This bespoke approach requires expert knowledge, is time-consuming, and may be error prone.  Ideally, we would have an easy-to-use post-processing tool that takes in the raw 1D pipeline output filename and produces a refined spectrum ready for scientific analysis. The `muler` framework fills this role. 

Each spectrograph and pipeline has peculiarities, and so `muler` provides domain-specific layers for individual spectrographs. The `muler` framework therefore enables user code to be compact and easily transferable among projects.

The Python spectroscopy ecosystem has dozens or hundreds of packages. Here we compare `muler` to a few recent examples to show how it fits into that ecosystem.
The `wobble` [@Bedell2019] framework performs advanced modeling on groups of dozens of échelle spectra of the same celestial object, assuming they have already been normalized, continuum-flattened, sky subtracted, etc. So in this case, `muler` serves as the layer that takes in the pipeline output, processes each spectrum identically, and then feeds these refined spectra into `wobble`. Similarly, the `starfish` framework [@czekala15] expects spectra to reside in an HDF5 format with telluric lines either corrected or masked. `muler` can satisfy these requirements. You can therefore view `muler` as taking care of these necessary pre-processing steps to get from the pipeline products to the inputs of virtually any other Python spectroscopy analysis package. 

Some packages such as `igrins_rv` [@IGRINSrv2021] conduct pre-processing steps as a matter of course in their analyses. These packages could hypothetically be refactored to make `muler` a dependency, with the boilerplate steps taken over by `muler`. This refactoring step would simplify the code, possibly making it more maintainable in the long term. In other cases, popular methods in `muler` may move upstream into `specutils`.

`muler` depends on `astropy` [@astropy13; @astropy18], `numpy` [@harris2020array], `specutils`, `scipy` [@scipy2020], and others.

# Architecture and design considerations

The `design` of `muler` shares three key inspirations from `lightkurve` [@lightkurve2018], which provides a well-liked open-source fluent interface to NASA Kepler/K2/TESS data. First, pipeline products have reliable and predictable formats that make them a *de facto* standard. You can then make an API that relies on that standard for accessing attributes and metadata, and build methods that combine the data, metadata, and optionally user inputs to conduct essentially all routine operations. 

The second key idea is that the output of an operation is usually self-similar: a `lightkurve` operation usually takes in a lightcurve and outputs a lightcurve. Here a `muler` operation takes in a `Spectrum1D`-like object and returns a `Spectrum1D`-like. This key idea makes method chaining both possible and desirable.

Finally, `lightkurve` objects inherit key behaviors from the Astropy `TimeSeries` module, reducing the duplication of code relevant to time series data. Here in `muler` we gain similar extensibility through an object-oriented design with `specutils`, abstracting away many routine spectroscopy tasks, and keeping the `muler` code lean.


# Supported spectrographs

We currently support custom operations for three spectrographs: the Immersion Grating Infrared Spectrograph, IGRINS [@park14; @mace2018] reduced with the `plp` [@IGRINSplp2017]; the Habitable Zone Planet Finder Spectrograph, HPF [@hpf2012; @hpf2014; @hpf2018] with either the `Goldilocks` or Penn State pipelines; and the Keck Near-Infrared Spectrograph, NIRSPEC [@mclean1998; @mclean2000] with the `NSDRP` pipeline. 

In principle, `muler` could be extended with devoted classes for other échelle spectrographs, so long as the pipeline outputs deliver standardized files with common metadata, such as date, time, celestial coordinates, etc.


# Acknowledgements

This research has made use of NASA's Astrophysics Data System Bibliographic Services. This material is based upon work supported by the National Aeronautics and Space Administration under Grant Numbers 80NSSC21K0650 for the NNH20ZDA001N-ADAP:D.2; and 80NSSC20K0257 for the XRP issued through the Science Mission Directorate.

# References