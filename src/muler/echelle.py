r"""
Echelle Spectrum
----------------

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
from astropy.modeling.physical_models import BlackBody
import specutils

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
        barycentric_corrections : float
            Barycentric correction for targets in units of m/s
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
        if hasattr(self, "ancillary_spectra"):
            if self.ancillary_spectra is not None:
                for ancillary_spectrum in self.ancillary_spectra:
                    if ancillary_spectrum in meta_out.keys():
                        meta_out[ancillary_spectrum] = meta_out[
                            ancillary_spectrum
                        ].divide(median_flux, handle_meta="first_found")

        self.meta = meta_out
        return self.divide(median_flux, handle_meta="first_found")

    def flatten_by_black_body(self, Teff):
        """Flatten the spectrum by a scaled black body, usually after deblazing
        Note: This method applies mostly to high-bandwidth stellar spectra.

        Parameters
        ----------
        Teff : float
            The effective temperature of the black body in Kelvin units
        """
        blackbody = BlackBody(temperature=Teff * u.K)(self.wavelength)
        blackbody = blackbody / np.mean(blackbody)
        wl_scaled = self.wavelength
        wl_scaled = wl_scaled / np.median(wl_scaled)
        try:
            return self.divide(blackbody / wl_scaled ** 2, handle_meta="first_found")
        except u.UnitConversionError:
            return self.divide(
                blackbody / wl_scaled ** 2 * self.unit, handle_meta="first_found"
            )

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
        bcRV = +1.0 * self.estimate_barycorr()

        try:
            self.radial_velocity = bcRV
            return self._copy(
                spectral_axis=self.wavelength.value * self.wavelength.unit
            )

        except:
            log.error(
                "rv shift requires specutils version >= 1.2, you have: {}".format(
                    specutils.__version__
                )
            )
            raise

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
            net_mask = spectrum.mask | (spectrum.flux.value != spectrum.flux.value)
            if spectrum.uncertainty is not None:
                masked_unc = StdDevUncertainty(spectrum.uncertainty.array[~net_mask])
            else:
                masked_unc = None

            meta_out = copy.deepcopy(spectrum.meta)
            meta_out["x_values"] = meta_out["x_values"][~net_mask]

            return self._copy(
                spectral_axis=spectrum.wavelength[~net_mask],
                flux=spectrum.flux[~net_mask],
                mask=spectrum.mask[~net_mask],
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
            if hasattr(self, "spectrographname"):
                ax.set_title(self.spectrographname + " Spectrum")
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

    def estimate_uncertainty(self):
        """Estimate the uncertainty based on residual after smoothing


        Returns
        -------
        uncertainty : (np.float)
            Typical uncertainty
        """
        residual = self.flux - self.smooth_spectrum().flux
        return median_abs_deviation(residual.value)

    def to_HDF5(self, path, file_basename):
        """Export to spectral order to HDF5 file format
        This format is required for per-order Starfish input

        Parameters
        ----------
        path : str
            The directory destination for the HDF5 file
        file_basename : str
            The basename of the file to which the order number and extension
            are appended.  Typically source name that matches a database entry.
        """
        grating_order = self.meta["m"]
        out_path = path + "/" + file_basename + "_m{:03d}.hdf5".format(grating_order)

        # The mask should be ones everywhere
        mask_out = np.ones(len(self.wavelength), dtype=int)
        f_new = h5py.File(out_path, "w")
        f_new.create_dataset("fls", data=self.flux.value)
        f_new.create_dataset("wls", data=self.wavelength.to(u.Angstrom).value)
        f_new.create_dataset("sigmas", data=self.uncertainty.array)
        f_new.create_dataset("masks", data=mask_out)
        f_new.close()


class EchelleSpectrumList(SpectrumList):
    r"""
    An enhanced container for a list of Echelle spectral orders
    """

    def __init__(self, *args, **kwargs):
        self.normalization_order_index = 0
        super().__init__(*args, **kwargs)

    def normalize(self, order_index=0):
        """Normalize all orders to one of the other orders
        """
        index = self.normalization_order_index
        median_flux = copy.deepcopy(np.nanmedian(self[index].flux))
        for i in range(len(self)):
            self[i] = self[i].divide(median_flux, handle_meta="first_found")

        return self

    def remove_nans(self):
        """Remove all the NaNs
        """
        # TODO: is this in-place overriding of self allowed?
        # May have unintended consequences?
        # Consider making a copy instead...
        for i in range(len(self)):
            self[i] = self[i].remove_nans()

        return self

    def remove_outliers(self, threshold=5):
        """Remove all the outliers

        Parameters
        ----------
        threshold : float
            The sigma-clipping threshold (in units of sigma)
        """
        for i in range(len(self)):
            self[i] = self[i].remove_outliers(threshold=threshold)

        return self

    def trim_edges(self, limits=None):
        """Trim all the edges
        """
        for i in range(len(self)):
            self[i] = self[i].trim_edges(limits)

        return self

    def deblaze(self, method="spline"):
        """Remove blaze function from all orders by interpolating a spline function

        Note: It is recommended to remove NaNs before running this operation,
                otherwise  effects can be appear from zero-padded edges.
        """
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i].deblaze(method=method)
        return spec_out


    def flatten_by_black_body(self, Teff):
        """Flatten by black body"""
        spec_out = copy.deepcopy(self)
        index = self.normalization_order_index
        median_wl = copy.deepcopy(np.nanmedian(self[index].wavelength))

        blackbody_func = BlackBody(temperature=Teff * u.K)
        blackbody_ref = blackbody_func(median_wl)

        for i in range(len(spec_out)):
            blackbody = (
                blackbody_func(spec_out[i].wavelength)
                / blackbody_ref
                / (spec_out[i].wavelength / median_wl) ** 2
            )
            try:
                spec_out[i] = spec_out[i].divide(blackbody, handle_meta="first_found")
            except u.UnitConversionError:
                spec_out[i] = spec_out[i].divide(
                    blackbody * self.unit, handle_meta="first_found"
                )

        return spec_out

    def to_HDF5(self, path, file_basename):
        """Save all spectral orders to the HDF5 file format
        """
        for i in range(len(self)):
            self[i].to_HDF5(path, file_basename)

    def stitch(self):
        """Stitch all the spectra together, assuming zero overlap in wavelength.  
        """
        wls = np.hstack([self[i].wavelength for i in range(len(self))])
        fluxes = np.hstack([self[i].flux for i in range(len(self))])
        # unc = np.hstack([self[i].uncertainty.array for i in range(len(self))])
        # unc_out = StdDevUncertainty(unc)

        return self[0].__class__(spectral_axis=wls, flux=fluxes)

    def plot(self, **kwargs):
        """Plot the entire spectrum list
        """
        if not "ax" in kwargs:
            ax = self[0].plot(figsize=(25, 4), **kwargs)
            for i in range(1, len(self)):
                self[i].plot(ax=ax, **kwargs)
            return ax
        else:
            for i in range(1, len(self)):
                self[i].plot(**kwargs)

        

    def __add__(self, other):
        """Bandmath addition
        """
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] + other[i]
        return spec_out

    def __sub__(self, other):
        """Bandmath subtraction
        """
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] - other[i]
        return spec_out

    def __mul__(self, other):
        """Bandmath multiplication
        """
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] * other[i]
        return spec_out

    def __truediv__(self, other):
        """Bandmath division
        """
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] / other[i]
        return spec_out
