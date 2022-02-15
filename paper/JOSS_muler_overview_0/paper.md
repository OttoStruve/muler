---
title: 'Astronomical echelle spectroscopy data analysis with `muler`'
tags:
  - Python
  - astronomy
  - spectroscopy
  - stars
  - echelle
authors:
  - name: Michael A. Gully-Santiago
    orcid: 0000-0002-4020-3457
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
  - name: Author with no affiliation^[corresponding author]
    affiliation: 3
affiliations:
 - name: The University of Texas at Austin Department of Astronomy
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 14 February 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Modern astronomical échelle spectrographs produce information-rich 2D echellograms that undergo standard reduction procedures to produce extracted 1D spectra.  The final post-processing steps of these 1D spectra are often left to the end-user scientists, since the order of opertations and algorithm choice may depend on the scientific application.  Implementing these post-processing steps from scatch acts as a barrier to entry to newcomers, taxes scientific innovation, and erodes scientific reproducibility.  Here we assemble and streamline a collection of spectroscopic data analysis methods into a single easy-to-use Application Programming Interface (API), `muler`.  The `specutils`-based fluent interface enables method chaining that yields compact 1-line code for many applications, a dramatic reduction in complexity compared to existing industry practices.  When applicable, some algorithms are customized to the pipeline data products of three near infrared spectrographs: HPF, Keck NIRSPEC, and IGRINS.  The framework could be extended to other spectrographs in the future.  The tutorials in `muler` may also serve as a central onboarding point for new entrants to the practice of near-infrared échelle spectroscopy data.  The open source permissively licensed Python 3 implementation lowers the barrier to entry and accelerates the investigation of astronomical spectra.  

# Statement of need

In its most elemental form, an astronomical spectrum consists of a list of wavelength coordinates and their flux values.  Additional metadata for uncertainties, date, time, sky pointing coordinates, and instrumental configuration are common.  The Astropy-affiliated `specutils` package serves as a uniting container for virtually any astronomical spectrum, by providing a way to represent, load, manipulate, and analyze these data and metadata.  It is therefore possible for each new practitioner to load a pipeline-produced output spectrum with, say, `specutils` and write standard post-processing operations from scratch.  This bespoke approach requires expert knowledge, is time consuming, and may be error prone.  Ideally we would have an easy-to-use post-processing tool that takes in the raw 1D pipeline output spectrum and produces a refined spectrum ready for scientific analysis.  The `muler` framework fills this role.  Furthermore, each spectrograph and pipeline has its own peculiarities, and so `muler` provides---when needed---a domain-specific layer for individual spectrographs.  The `muler` framework therefore enables user code to be compact and easily transferable among projects.

The Python spectroscopy ecosystem has dozens or hundreds of packages.  Here we compare `muler` to a few recent examples to show how it fits into that ecosystem.
The `wobble` [@Bedell2019] framework performs advanced modeling on groups of dozens of échelle spectra of the same celestial object, assuming they have already been normalized, continuum-flattened, sky subtracted, etc.  So `muler` can serve as the layer that takes in the pipeline output and processes each spectrum identically to input into `wobble`.  Similarly, the  `starfish` framework [@czekala15] expects spectra to reside in an HDF5 format with telluric lines either corrected or masked.  `muler` can satisfy these requirements.  You can therefore view muler as taking care of these necessary pre-processing steps to get from the pipeline products to the inputs of virtually any other Python spectroscopy analysis package.  

Some packages such as `igrins_rv` [@IGRINSrv2021] conduct pre-processing steps as a matter of course in their analyses.  These packages could hypothetically be refactored to make `muler` a dependency, with the boilerplate steps taken over by `muler`.  This refactoring step would simplify the code, possibly making it more maintainable in the long term.  In other cases, popular methods in `muler` may move upstream into `specutils`.

`muler` depends on `astropy` [@astropy13; @astropy18], `numpy` [@harris2020array], `specutils`, `scipy` [@scipy2020], and others.

# Supported spectrographs

We currently support custom operations for three spectrographs: the Immersion Grating Infrared Spectrograph, IGRINS [@park14; @mace2018] reduced with the `plp` [@IGRINSplp2017]; the Habitable Zone Planet Finder Spectrograph, HPF [@hpf2012; @hpf2014; @hpf2018] with either the `Goldilocks` or Penn State pipelines; and the Keck Near-Infrared Spectrograph, NIRSPEC [@mclean1998; @mclean2000] with the `NSDRP` pipeline.  

In princple, `muler` could be extended with devoted classes for other échelle spectrographs, so long as the pipeline outputs deliver standardized files with common metadata, such as date, time, celestial coordinates, etc.


# Acknowledgements

This research has made use of NASA's Astrophysics Data System Bibliographic Services.  

# References