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

    def remove_nans(self):
        """Remove data points that have NaN fluxes

        By default the method removes NaN's from target, sky, and lfc fibers.

        Returns
        -------
        finite_spec : (KeckNIRSPECSpectrum)
            Spectrum with NaNs removed
        """

        # Todo: probably want to check that all NaNs are in the mask

        def remove_nans_per_spectrum(spectrum):
            if spectrum.uncertainty is not None:
                masked_unc = StdDevUncertainty(
                    spectrum.uncertainty.array[~spectrum.mask]
                )
            else:
                masked_unc = None

            meta_out = copy.deepcopy(spectrum.meta)
            meta_out["x_values"] = meta_out["x_values"][~spectrum.mask]

            return self._copy(
                spectral_axis=spectrum.wavelength[~spectrum.mask],
                flux=spectrum.flux[~spectrum.mask],
                mask=spectrum.mask[~spectrum.mask],
                uncertainty=masked_unc,
                meta=meta_out,
            )

        new_self = remove_nans_per_spectrum(self)
        if "sky" in self.meta.keys():
            new_sky = remove_nans_per_spectrum(self.sky)
            new_self.meta["sky"] = new_sky
        # if "lfc" in self.meta.keys():
        #    new_lfc = remove_nans_per_spectrum(self.lfc)
        #    new_self.meta["lfc"] = new_lfc

        return new_self

    def smooth_spectrum(self):
        """Smooth the spectrum using Gaussian Process regression

        Returns
        -------
        smoothed_spec : (EchelleSpectrum)
            Smooth version of input Spectrum
        """
        if self.uncertainty is not None:
            unc = self.uncertainty.array
        else:
            unc = np.repeat(np.nanmedian(self.flux.value) / 100.0, len(self.flux))

        kernel = terms.SHOTerm(sigma=0.03, rho=15.0, Q=0.5)
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        gp.compute(self.wavelength)

        # Construct the GP model with celerite
        def set_params(params, gp):
            gp.mean = params[0]
            theta = np.exp(params[1:])
            gp.kernel = terms.SHOTerm(sigma=theta[0], rho=theta[1], Q=0.5)
            gp.compute(self.wavelength.value, yerr=unc + theta[2], quiet=True)
            return gp

        def neg_log_like(params, gp):
            gp = set_params(params, gp)
            return -gp.log_likelihood(self.flux.value)

        initial_params = [np.log(1), np.log(0.001), np.log(5.0), np.log(0.01)]
        soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
        opt_gp = set_params(soln.x, gp)

        mean_model = opt_gp.predict(self.flux.value, t=self.wavelength.value)

        meta_out = copy.deepcopy(self.meta)
        meta_out["x_values"] = meta_out["x_values"][~self.mask]

        return self._copy(
            spectral_axis=self.wavelength,
            flux=mean_model * self.flux.unit,
            mask=np.zeros_like(mean_model, dtype=np.bool),
            meta=meta_out,
        )

    def plot(self, ax=None, ylo=0.6, yhi=1.2, figsize=(10, 4), **kwargs):
        """Plot a quick look of the spectrum"

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        ylo : scalar
            Lower limit of the y axis
        yhi : scalar
            Upper limit of the y axis
        figsize : tuple
            The figure size for the plot
        label : str
            The legend label to for plt.legend()

        Returns
        -------
        ax : (`~matplotlib.axes.Axes`)
            The axis to display and/or modify
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.set_ylim(ylo, yhi)
            ax.set_xlabel("$\lambda \;(\AA)$")
            ax.set_ylabel("Flux")
            ax.step(self.wavelength, self.flux, **kwargs)
        else:
            ax.step(self.wavelength, self.flux, **kwargs)

        return ax

    def remove_outliers(self, threshold=5):
        """Remove outliers above threshold

        Parameters
        ----------
        threshold : float
            The sigma-clipping threshold (in units of sigma)


        Returns
        -------
        clean_spec : (KeckNIRSPECSpectrum)
            Cleaned version of input Spectrum
        """
        residual = self.flux - self.smooth_spectrum().flux
        mad = median_abs_deviation(residual.value)
        mask = np.abs(residual.value) > threshold * mad

        spectrum_out = copy.deepcopy(self)
        spectrum_out._mask = mask
        spectrum_out.flux[mask] = np.NaN

        return spectrum_out.remove_nans()

    def trim_edges(self, limits=None):
        """Trim the order edges, which falloff in SNR

        This method applies limits on absolute x pixel values, regardless
        of the order of previous destructive operations, which may not
        be the intended behavior in some applications.

        Parameters
        ----------
        limits : tuple
            The index bounds (lo, hi) for trimming the order

        Returns
        -------
        trimmed_spec : (EchelleSpectrum)
            Trimmed version of input Spectrum
        """
        if limits is None:
            limits = self.noisy_edges
        lo, hi = limits
        meta_out = copy.deepcopy(self.meta)
        x_values = meta_out["x_values"]
        mask = (x_values < lo) | (x_values > hi)

        if self.uncertainty is not None:
            masked_unc = StdDevUncertainty(self.uncertainty.array[~mask])
        else:
            masked_unc = None

        meta_out["x_values"] = x_values[~mask]

        return self._copy(
            spectral_axis=self.wavelength[~mask],
            flux=self.flux[~mask],
            mask=self.mask[~mask],
            uncertainty=masked_unc,
            meta=meta_out,
        )
