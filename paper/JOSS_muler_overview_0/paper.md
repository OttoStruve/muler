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

Modern échelle spectrographs produce information-rich 2-dimensional (2D) raw echellograms that undergo standard reduction procedures to produce extracted 1D spectra, typically chunked into echelle orders.  The final post-processing steps for echelle orders are often left to the end-user scientists, since the order of opertations and algorithm choice may depend on the scientific application.  These steps---while uncontroversial and routine---act as a barrier to entry to newcomers, and act as a tax on scientific innovation since teams have to re-invent the wheel to overcome implementation complexity, before getting to the already inherently complex scientific enterprise.  Here we assemble and streamline a collection of standard post-processing algorithms into a single easy-to-use Application Programming Interface (API).  The open source permissively licensed Python 3 implementation `muler` lowers the barrier to entry and accelerates the investigation of astronomical spectra.  The framework currently supports data from the HPF, Keck NIRSPEC, and IGRINS spectrographs, and is extensible to others.  

# Statement of need

The `specutils` framework is:  

> a Python package for representing, loading, manipulating, and analyzing astronomical spectroscopic data. The generic data containers and accompanying modules provide a toolbox that the astronomical community can use to build more domain-specific packages.

Here we explain muler.

Here are some other python packages for echelle spectroscopy: `wobble` [@Bedell2019], and `starfish` [@czekala15].  You can view muler as taking care of the necessary pre-processing steps to get from the pipeline products to the inputs of those packages.

We depend on `astropy` [@astropy13; @astropy18], `numpy` [@harris2020array], and others.

# Supported spectrographs

We currently support custom operations for three spectrographs: the Immersion Grating Infrared Spectrograph, IGRINS [@park14; @mace2018]; the Habitable Zone Planet Finder Spectrograph, HPF [@hpf2012; @hpf2014; @hpf2018] ; and the Keck Near-Infrared Spectrograph, NIRSPEC [@mclean1998; @mclean2000].  

In princple, `muler` could be extended with devoted classes for other échelle spectrographs, so long as the pipeline outputs deliver standardized files with common metadata, such as date, time, celestial coordinates, etc.


# Acknowledgements

This research has made use of NASA's Astrophysics Data System Bibliographic Services.  




# References