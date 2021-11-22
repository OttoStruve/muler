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
from astropy.units import Quantity
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty
from scipy.stats import median_abs_deviation
from scipy.interpolate import InterpolatedUnivariateSpline
from specutils.analysis import equivalent_width
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from astropy.constants import R_jup, R_sun, G, M_jup, R_earth, c
from astropy.modeling.physical_models import BlackBody
import specutils
from muler.utilities import apply_numpy_mask, resample_list

# from barycorrpy import get_BC_vel
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time


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

        self.ancillary_spectra = None
        super().__init__(*args, **kwargs)

    @property
    def snr(self):
        """The Signal-to-Noise Ratio :math:`\frac{S}{N}`, the flux divided by the uncertainty

        The spectrum should have an input uncertainty, otherwise returns NaNs
        """

        if self.uncertainty is not None:
            if self.uncertainty.uncertainty_type == "std":
                snr_estimate = self.flux / self.uncertainty.quantity
            elif self.uncertainty.uncertainty_type == "ivar":
                snr_estimate = self.flux * np.sqrt(self.uncertainty.quantity)
            else:
                message = "SNR only supports standard deviation and inverse variance uncertainty"
                raise NotImplementedError(message)
        else:
            snr_estimate = np.repeat(np.NaN, len(self.flux)) * u.dimensionless_unscaled

        return snr_estimate

    @property
    def available_ancillary_spectra(self):
        """The list of available ancillary spectra"""

        output = []
        if hasattr(self, "ancillary_spectra"):
            if self.ancillary_spectra is not None:
                output = [
                    ancillary_spectrum
                    for ancillary_spectrum in self.ancillary_spectra
                    if ancillary_spectrum in self.meta.keys()
                ]
        return output

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
        spec = self._copy(
            spectral_axis=self.wavelength.value * self.wavelength.unit, wcs=None
        )
        median_flux = np.nanmedian(spec.flux.value)

        # Each ancillary spectrum (e.g. sky) should also be normalized
        meta_out = copy.deepcopy(spec.meta)
        for ancillary_spectrum in self.available_ancillary_spectra:
            meta_out[ancillary_spectrum] = meta_out[ancillary_spectrum].divide(
                median_flux * spec.flux.unit, handle_meta="ff"
            )

        # spec.meta = meta_out
        return spec.divide(
            median_flux * spec.flux.unit, handle_meta="first_found"
        )._copy(meta=meta_out)

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

    def flatten(
        self,
        window_length=101,
        polyorder=2,
        return_trend=False,
        break_tolerance=5,
        niters=3,
        sigma=3,
        mask=None,
        **kwargs,
    ):
        """Removes the low frequency trend using scipy's Savitzky-Golay filter.
        This method wraps `scipy.signal.savgol_filter`.  Abridged from the
        `lightkurve` method with the same name for flux time series.

        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e. the number of coefficients).
            ``window_length`` must be a positive odd integer.
        polyorder : int
            The order of the polynomial used to fit the samples. ``polyorder``
            must be less than window_length.
        return_trend : bool
            If `True`, the method will return a tuple of two elements
            (flattened_spec, trend_spec) where trend_spec is the removed trend.
        break_tolerance : int
            If there are large gaps in wavelength, flatten will split the flux into
            several sub-spectra and apply `savgol_filter` to each
            individually. A gap is defined as a region in wavelength larger than
            `break_tolerance` times the median gap.  To disable this feature,
            set `break_tolerance` to None.
        niters : int
            Number of iterations to iteratively sigma clip and flatten. If more than one, will
            perform the flatten several times, removing outliers each time.
        sigma : int
            Number of sigma above which to remove outliers from the flatten
        mask : boolean array with length of self.wavelength
            Boolean array to mask data with before flattening. Flux values where
            mask is True will not be used to flatten the data. An interpolated
            result will be provided for these points. Use this mask to remove
            data you want to preserve, e.g. spectral regions of interest.
        **kwargs : dict
            Dictionary of arguments to be passed to `scipy.signal.savgol_filter`.
        Returns
        -------
        flatten_spec : `EchelleSpectrum`
            New light curve object with long-term trends removed.
        If ``return_trend`` is set to ``True``, this method will also return:
        trend_spec : `EchelleSpectrum`
            New light curve object containing the trend that was removed.
        """
        if mask is None:
            mask = np.ones(len(self.wavelength), dtype=bool)
        else:
            # Deep copy ensures we don't change the original.
            mask = copy.deepcopy(~mask)
        # No NaNs
        mask &= np.isfinite(self.flux)
        # No outliers
        mask &= np.nan_to_num(np.abs(self.flux - np.nanmedian(self.flux))) <= (
            np.nanstd(self.flux) * sigma
        )
        for iter in np.arange(0, niters):
            if break_tolerance is None:
                break_tolerance = np.nan
            if polyorder >= window_length:
                polyorder = window_length - 1
                log.warning(
                    "polyorder must be smaller than window_length, "
                    "using polyorder={}.".format(polyorder)
                )
            # Split the lightcurve into segments by finding large gaps in time
            dlam = self.wavelength.value[mask][1:] - self.wavelength.value[mask][0:-1]
            with warnings.catch_warnings():  # Ignore warnings due to NaNs
                warnings.simplefilter("ignore", RuntimeWarning)
                cut = np.where(dlam > break_tolerance * np.nanmedian(dlam))[0] + 1
            low = np.append([0], cut)
            high = np.append(cut, len(self.wavelength[mask]))
            # Then, apply the savgol_filter to each segment separately
            trend_signal = Quantity(
                np.zeros(len(self.wavelength[mask])), unit=self.flux.unit
            )
            for l, h in zip(low, high):
                # Reduce `window_length` and `polyorder` for short segments;
                # this prevents `savgol_filter` from raising an exception
                # If the segment is too short, just take the median
                if np.any([window_length > (h - l), (h - l) < break_tolerance]):
                    trend_signal[l:h] = np.nanmedian(self.flux[mask][l:h])
                else:
                    # Scipy outputs a warning here that is not useful, will be fixed in version 1.2
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        trsig = savgol_filter(
                            x=self.flux.value[mask][l:h],
                            window_length=window_length,
                            polyorder=polyorder,
                            **kwargs,
                        )
                        trend_signal[l:h] = Quantity(trsig, trend_signal.unit)
            # Ignore outliers;
            # Note that it's possible numerical noise can cause outliers...
            # If this happens you can add `1e-14` below to avoid detecting
            # outliers which are merely caused by numerical noise.
            mask1 = np.nan_to_num(np.abs(self.flux[mask] - trend_signal)) < (
                np.nanstd(self.flux[mask] - trend_signal)
                * sigma
                # + Quantity(1e-14, self.flux.unit)
            )
            f = interp1d(
                self.wavelength.value[mask][mask1],
                trend_signal[mask1],
                fill_value="extrapolate",
            )
            trend_signal = Quantity(f(self.wavelength.value), self.flux.unit)
            mask[mask] &= mask1

        flatten_spec = copy.deepcopy(self)
        trend_spec = self._copy(flux=trend_signal)
        with warnings.catch_warnings():
            # ignore invalid division warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            flatten_spec = flatten_spec.divide(trend_spec, handle_meta="ff")
        if return_trend:
            return flatten_spec, trend_spec
        else:
            return flatten_spec

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
        return self.rv_shift(bcRV)

    def rv_shift(self, velocity):
        """
        Shift velocity of spectrum in astropy units (or km/s if input velocity is just a float)
        """
        if (
            type(velocity) == float
        ):  # If supplied velocity is not using astropy units, default to km/s
            velocity = velocity * (u.km / u.s)
        try:
            self.radial_velocity = velocity
            return self._copy(
                spectral_axis=self.wavelength.value * self.wavelength.unit,
                wcs=None,
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
        keep_indices = (self.mask == False) & (self.flux == self.flux)
        return self.apply_boolean_mask(keep_indices)

    def smooth_spectrum(
        self, return_model=False, optimize_kernel=False, bandwidth=150.0
    ):
        """Smooth the spectrum using Gaussian Process regression

        Parameters
        -------
        return_model : (bool)
            Whether or not to return the gp model, which takes a wavelength axis
            as input and outputs the smooth trend
        optimize_kernel : (bool)
            Whether to optimize the GP hyperparameters: correlation scale and amplitude
        bandwidth : (float)
            The smoothing bandwidth in Angstroms.  Defaults to 150 Angstrom lengthscale.

        Returns
        -------
        smoothed_spec : (EchelleSpectrum)
            Smooth version of input Spectrum
        """
        try:
            from celerite2 import terms
            import celerite2
        except ImportError:
            raise ImportError(
                "You need to install celerite2 to use the smoothing='celerite' method."
            )
        if self.uncertainty is not None:
            unc = self.uncertainty.array
        else:
            unc = np.repeat(np.nanmedian(self.flux.value) / 100.0, len(self.flux))

        # TODO: change rho to depend on the bandwidth
        kernel = terms.SHOTerm(sigma=0.01, rho=bandwidth, Q=0.25)
        gp = celerite2.GaussianProcess(kernel, mean=0.0)
        gp.compute(self.wavelength, yerr=unc)

        if optimize_kernel:
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
        else:
            opt_gp = gp

        mean_model = opt_gp.predict(self.flux.value, t=self.wavelength.value)

        smoothed_spectrum = self.__class__(
            spectral_axis=self.wavelength.value * self.wavelength.unit,
            flux=mean_model * self.flux.unit,
            uncertainty=None,
            mask=np.zeros_like(mean_model, dtype=np.bool),
            meta=copy.deepcopy(self.meta),
            wcs=None,
        )

        if return_model:
            gp_model = lambda wl: opt_gp.predict(self.flux.value, t=wl)
            return (smoothed_spectrum, gp_model)
        else:
            return smoothed_spectrum

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
            ax.step(self.wavelength, self.flux, **kwargs, where="mid")
        else:
            ax.step(self.wavelength, self.flux, **kwargs, where="mid")

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
        mad = median_abs_deviation(residual.value, nan_policy="omit")
        keep_indices = (np.abs(residual.value) < (threshold * mad)) == True

        return self.apply_boolean_mask(keep_indices)

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
        if hasattr(self.meta, "x_values"):
            x_values = self.meta["x_values"]
        else:
            x_values = np.arange(len(self.wavelength))
        keep_indices = (x_values > lo) & (x_values < hi)

        return self.apply_boolean_mask(keep_indices)

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
        try:
            import h5py
        except ImportError:
            raise ImportError("You need to install h5py to export to the HDF5 format.")

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

    def resample_list(self, specList, **kwargs):
        """
        Resample a single EchelleSpectrum object into a EchelleSpectrumList object.
        Useful for converting models into echelle spectra with multiple orders.
        """
        return resample_list(self, specList, **kwargs)

    def apply_boolean_mask(self, mask):
        """Apply a boolean mask to the spectrum and any available ancillary spectra

        Parameters
        ----------
        mask: boolean mask, typically a numpy array
            The boolean mask with numpy-style masking: True means "keep" that index and
            False means discard that index
        """

        spec = apply_numpy_mask(self, mask)

        for ancillary_spectrum in self.available_ancillary_spectra:
            spec.meta[ancillary_spectrum] = apply_numpy_mask(
                spec.meta[ancillary_spectrum], mask
            )

        return spec


class EchelleSpectrumList(SpectrumList):
    r"""
    An enhanced container for a list of Echelle spectral orders
    """

    def __init__(self, *args, **kwargs):
        self.normalization_order_index = 0
        super().__init__(*args, **kwargs)

    def normalize(self, order_index=0):
        """Normalize all orders to one of the other orders"""
        index = self.normalization_order_index
        median_flux = copy.deepcopy(np.nanmedian(self[index].flux))
        for i in range(len(self)):
            self[i] = self[i].divide(median_flux, handle_meta="first_found")

        return self

    def remove_nans(self):
        """Remove all the NaNs"""
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
        """Trim all the edges"""
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
        """Save all spectral orders to the HDF5 file format"""
        for i in range(len(self)):
            self[i].to_HDF5(path, file_basename)

    def stitch(self):
        """Stitch all the spectra together, assuming zero overlap in wavelength."""
        spec = copy.deepcopy(self)
        wls = (
            np.hstack([spec[i].wavelength.value for i in range(len(spec))])
            * spec[0].wavelength.unit
        )
        fluxes = (
            np.hstack([spec[i].flux.value for i in range(len(spec))])
            * spec[0].flux.unit
        )
        if spec[0].uncertainty is not None:
            # HACK We assume if one order has it, they all do, and that it's StdDev
            unc = np.hstack([spec[i].uncertainty.array for i in range(len(self))])
            unc_out = StdDevUncertainty(unc)
        else:
            unc_out = None

        # Stack the x_values:
        x_values = np.hstack([spec[i].meta["x_values"] for i in range(len(spec))])

        meta_out = copy.deepcopy(spec[0].meta)
        meta_out["x_values"] = x_values
        for ancillary_spectrum in spec[0].available_ancillary_spectra:
            if spec[0].meta[ancillary_spectrum].meta is not None:
                meta_of_meta = spec[0].meta[ancillary_spectrum].meta
                x_values = np.hstack(
                    [
                        spec[i].meta[ancillary_spectrum].meta["x_values"]
                        for i in range(len(spec))
                    ]
                )
                meta_of_meta["x_values"] = x_values
            else:
                meta_of_meta = None
            wls_anc = np.hstack(
                [spec[i].meta[ancillary_spectrum].wavelength for i in range(len(spec))]
            )
            fluxes_anc = np.hstack(
                [spec[i].meta[ancillary_spectrum].flux for i in range(len(spec))]
            )

            meta_out[ancillary_spectrum] = spec[0].__class__(
                spectral_axis=wls_anc, flux=fluxes_anc, meta=meta_of_meta
            )

        return spec[0].__class__(
            spectral_axis=wls, flux=fluxes, uncertainty=unc_out, meta=meta_out, wcs=None
        )

    def plot(self, **kwargs):
        """Plot the entire spectrum list"""
        if not "ax" in kwargs:
            ax = self[0].plot(figsize=(25, 4), **kwargs)
            for i in range(1, len(self)):
                self[i].plot(ax=ax, **kwargs)
            return ax
        else:
            for i in range(1, len(self)):
                self[i].plot(**kwargs)

    def __add__(self, other):
        """Bandmath addition"""
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] + other[i]
        return spec_out

    def __sub__(self, other):
        """Bandmath subtraction"""
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] - other[i]
        return spec_out

    def __mul__(self, other):
        """Bandmath multiplication"""
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] * other[i]
        return spec_out

    def __truediv__(self, other):
        """Bandmath division"""
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i] / other[i]
        return spec_out

    def rv_shift(self, velocity):
        """
        Shift velocity of spectrum in km s^-1
        """
        spec_out = copy.deepcopy(self)
        for i in range(len(self)):
            spec_out[i] = self[i].rv_shift(velocity)
        return spec_out
