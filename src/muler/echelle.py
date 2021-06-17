r"""
Echelle Spectrum
---------------

An abstract base class for a high resolution spectrum, for some echelle order :math:`m \in ` out of :math:`M` total orders, each with vectors for wavelength, flux, and uncertainty, e.g. :math:`F_m(\lambda)`.  This class is a subclass of specutils' Spectrum1D and is intended to have its methods inherited by specific instrument classes.


EchelleSpectrum
###############
"""

import warnings
import logging
import numpy as np
import astropy
import pandas as pd
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty
from scipy.stats import median_abs_deviation
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from specutils.analysis import equivalent_width
from scipy.interpolate import UnivariateSpline
from astropy.constants import R_jup, R_sun, G, M_jup, R_earth, c

# from barycorrpy import get_BC_vel
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from celerite2 import terms
import celerite2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import copy

from specutils.spectra.spectral_region import SpectralRegion
from specutils.analysis import equivalent_width


log = logging.getLogger(__name__)

from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter("ignore", category=VerifyWarning)

#  See Issue: https://github.com/astropy/specutils/issues/779
warnings.filterwarnings(
    "ignore", category=astropy.utils.exceptions.AstropyDeprecationWarning
)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
# See Issue: https://github.com/astropy/specutils/issues/800
warnings.filterwarnings("ignore", category=RuntimeWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from specutils import Spectrum1D
    from specutils import SpectrumList


class EchelleSpectrum(Spectrum1D):
    r"""
    An abstract base class to provide common methods that will be inherited by instrument-specific classes
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def estimate_barycorr(self):
        """Estimate the Barycentric Correction from the Date and Target Coordinates

        Returns
        -------
        barycentric_corrections : (float, float)
            Tuple of floats for the barycentric corrections for target and LFC
        """
        obstime = self.astropy_time
        loc = EarthLocation.of_site(self.site_name)
        sc = SkyCoord(ra=self.RA, dec=self.DEC)
        barycorr = sc.radial_velocity_correction(obstime=obstime, location=loc)
        return barycorr

    def measure_ew(self, mu):
        """Measure the equivalent width of a given spectrum
        
        Parameters
        ----------
        mu : scalar/float
            The center wavelength of given line
        
        Returns
        -------
        equivalent width : (scalar)
        """
        log.warning("Experimental method")

        left_bound = 0.999 * mu * u.Angstrom
        right_bound = 1.001 * mu * u.Angstrom
        ew = equivalent_width(self, regions=SpectralRegion(left_bound, right_bound))
        return ew

    def normalize(self):
        """Normalize spectrum by its median value

        Returns
        -------
        normalized_spec : (KeckNIRSPECSpectrum)
            Normalized Spectrum
        """
        median_flux = np.nanmedian(self.flux)

        # Each ancillary spectrum (e.g. sky) should also be normalized
        meta_out = copy.deepcopy(self.meta)
        if self.ancillary_spectra is not None:
            for ancillary_spectrum in self.ancillary_spectra:
                meta_out[ancillary_spectrum] = meta_out[ancillary_spectrum].divide(
                    median_flux, handle_meta="first_found"
                )

        self.meta = meta_out
        return self.divide(median_flux, handle_meta="first_found")

    def deblaze(self, method="spline"):
        """Remove blaze function from spectrum by interpolating a spline function

        Note: It is recommended to remove NaNs before running this operation,
                otherwise edge effects can be appear from zero-padded edges.

        Returns
        -------
        blaze corrrected spectrum
        """

        if method == "spline":
            if np.any(np.isnan(self.flux)):
                log.warning(
                    "your spectrum contains NaNs, "
                    "it is highly recommended to run `.remove_nans()` before deblazing"
                )

            spline = UnivariateSpline(self.wavelength, np.nan_to_num(self.flux), k=5)
            interp_spline = spline(self.wavelength) * self.flux.unit

            no_blaze = self.divide(interp_spline, handle_meta="first_found")

            if "sky" in self.meta.keys():
                new_sky = self.sky.divide(interp_spline, handle_meta="first_found")
                no_blaze.meta["sky"] = new_sky

            return no_blaze

        else:
            raise NotImplementedError

    def barycentric_correct(self):
        """shift spectrum by barycenter velocity

        Returns
        -------
        barycenter corrected Spectrum : (KeckNIRSPECSpectrum)
        """
        bcRV = self.estimate_barycorr()

        try:
            self.radial_velocity = bcRV
        except:
            log.error(
                "rv shift requires specutils version >= 1.2, you have: {}".format(
                    specutils.__version__
                )
            )
            raise
        return self
