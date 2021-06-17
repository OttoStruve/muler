r"""
KeckNIRSPEC Spectrum
---------------

A container for a Keck NIRSPEC high resolution spectrum, for some echelle order :math:`m \in ` out of :math:`M` total orders, each with vectors for wavelength, flux, and uncertainty, e.g. :math:`F_m(\lambda)`.  KeckNIRSPEC already has been sky subtracted, but the subtracted sky signal is contained as the spectrum.sky attribute for reference.


KeckNIRSPECSpectrum
###################
"""

import warnings
import logging
from muler.echelle import EchelleSpectrum
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


class KeckNIRSPECSpectrum(EchelleSpectrum):
    r"""
    A container for Keck NIRSPEC spectra

    Args:
        file (str): A path to a reduced Keck NIRSPEC spectrum from NSDRP
    """

    def __init__(self, *args, file=None, order=63, **kwargs):

        self.site_name = "Keck Observatory"

        if file is not None:
            file_basename = file.split("/")[-1]
            assert (
                file_basename[0:3] == "NS."
            ), "Only NSDRP spectra are currently supported"
            pipeline = "NSDRP"
            assert (
                "_flux_tbl.fits" in file_basename
            ), "Only fits table files are currently supported"
            file_stem = file_basename.split("_flux")[0]
            grating_order = int(file_stem[-2:])

            assert os.path.exists(file), "The file must exist"

            hdu = fits.open(file)
            hdu0 = hdu[1]

            ## Target Spectrum
            lamb = hdu0.data["wave (A)"].astype(np.float64) * u.AA
            flux = hdu0.data["flux (cnts)"].astype(np.float64) * u.ct
            unc = hdu0.data["noise (cnts)"].astype(np.float64) * u.ct

            uncertainty = StdDevUncertainty(unc)
            mask = (
                np.isnan(flux) | np.isnan(uncertainty.array) | (uncertainty.array <= 0)
            )

            # Attempt to read-in the header:
            fits_with_full_header = file.replace("/fitstbl/", "/fits/").replace(
                "_flux_tbl.", "_flux."
            )
            if os.path.exists(fits_with_full_header):
                hdu_hdr = fits.open(fits_with_full_header)
                hdr = hdu_hdr[0].header
                wcs = WCS(hdr)
            else:
                wcs = None
                hdr = None

            meta_dict = {
                "x_values": hdu0.data["col"].astype(np.int),
                "pipeline": pipeline,
                "m": grating_order,
                "header": hdr,
            }

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=wcs,
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs
            )

            ## Sky Spectrum
            flux = hdu0.data["sky (cnts)"].astype(np.float64) * u.ct

            sky_spectrum = KeckNIRSPECSpectrum(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict.copy(),
                **kwargs
            )

            self.meta["sky"] = sky_spectrum
        else:
            super().__init__(*args, **kwargs)

    @property
    def pipeline(self):
        """Which pipeline does this spectrum originate from?"""
        return self.meta["pipeline"]

    @property
    def sky(self):
        """Sky fiber spectrum stored as its own KeckNIRSPECSpectrum object"""
        return self.meta["sky"]

    @property
    def flat(self):
        """Flat spectrum stored as its own KeckNIRSPECSpectrum object"""
        return self.meta["flat"]

    @property
    def RA(self):
        """The right ascension from header files"""
        self.meta["header"]["RA"] * u.hourangle

    @property
    def DEC(self):
        """The declination from header files"""
        self.meta["header"]["DEC"] * u.deg

    @property
    def astropy_time(self):
        """The astropy time based on the header"""
        mjd = self.meta["header"]["MJD-OBS"]
        time_out = Time(mjd, format="mjd", scale="utc")
        return time_out

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

    def normalize(self):
        """Normalize spectrum by its median value

        Returns
        -------
        normalized_spec : (KeckNIRSPECSpectrum)
            Normalized Spectrum
        """
        median_flux = np.nanmedian(self.flux)

        # return self.divide(median_flux, handle_meta="first_found")
        meta_out = copy.deepcopy(self.meta)
        meta_out["sky"] = meta_out["sky"].divide(median_flux, handle_meta="first_found")
        # meta_out["flat"] = meta_out["flat"].divide(median_flux, handle_meta="first_found")
        return KeckNIRSPECSpectrum(
            spectral_axis=self.wavelength,
            flux=self.flux,
            meta=meta_out,
            mask=self.mask,
            uncertainty=self.uncertainty,
        ).divide(median_flux, handle_meta="first_found")

    def sky_subtract(self, force=False):
        """Subtract science spectrum from sky spectrum

        Note: This operation does not wavelength shift or scale the sky spectrum

        Returns
        -------
        sky_subtractedSpec : (KeckNIRSPECSpectrum)
            Sky subtracted Spectrum
        """
        if force:
            log.warn(
                "NIRSPEC data are already natively sky subtracted! "
                "Proceeding with a forced sky subtraction anyways..."
            )
            return self.subtract(self.sky, handle_meta="first_found")
        else:
            log.error(
                "NIRSPEC data are already natively sky subtracted! "
                "To proceed anyway, state `force=True`."
            )
            return self

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

    def blaze_divide_spline(self):
        """Remove blaze function from spectrum by interpolating a spline function

        Note: It is recommended to remove NaNs before running this operation,
                otherwise edge effects can be appear from zero-padded edges.

        Returns
        -------
        blaze corrrected spectrum : (KeckNIRSPECSpectrum)
        """
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

        # if "flat" in self.meta.keys():
        #    new_flat = self.flat.divide(interp_spline, handle_meta="first_found")
        #    no_blaze.meta["flat"] = new_flat

        return no_blaze

    def barycentric_correct(self):
        """shift spectrum by barycenter velocity

        Returns
        -------
        barycenter corrected Spectrum : (KeckNIRSPECSpectrum)
        """
        log.error("Not yet tested on NIRSPEC data, should fail!")
        meta_out = copy.deepcopy(self.meta)

        bcRV = self.estimate_barycorr()

        new_wave = self.wavelength * (1.0 + (bcRV.value / c.value))

        return KeckNIRSPECSpectrum(
            spectral_axis=new_wave,
            flux=self.flux,
            mask=self.mask,
            uncertainty=self.uncertainty,
            meta=meta_out,
        )

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

            return KeckNIRSPECSpectrum(
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
        smoothed_spec : (KeckNIRSPECSpectrum)
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

        return KeckNIRSPECSpectrum(
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

    def trim_edges(self, limits=(10, 1000)):
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
        trimmed_spec : (KeckNIRSPECSpectrum)
            Trimmed version of input Spectrum
        """
        lo, hi = limits
        meta_out = copy.deepcopy(self.meta)
        x_values = meta_out["x_values"]
        mask = (x_values < lo) | (x_values > hi)

        if self.uncertainty is not None:
            masked_unc = StdDevUncertainty(self.uncertainty.array[~mask])
        else:
            masked_unc = None

        meta_out["x_values"] = x_values[~mask]

        return KeckNIRSPECSpectrum(
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


class KeckNIRSPECSpectrumList(SpectrumList):
    r"""
    An enhanced container for a list of KeckNIRSPEC spectral orders

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def read(files):
        """Read in a SpectrumList from a file

        Parameters
        ----------
        file : (str)
            A path to a reduced KeckNIRSPEC spectrum from plp
        """
        n_orders = len(files)

        list_out = []
        for i in range(n_orders):
            assert ".flux_tbl.fits" in files[i]
            spec = KeckNIRSPECSpectrum(file=files[i])
            list_out.append(spec)
        return KeckNIRSPECSpectrumList(list_out)

    def normalize(self):
        """Normalize the all spectra to order 14's median
        """
        median_flux = copy.deepcopy(np.nanmedian(self[0].flux))
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

    def trim_edges(self):
        """Trim all the edges
        """
        for i in range(len(self)):
            self[i] = self[i].trim_edges()

        return self

    def to_HDF5(self, path, file_basename):
        """Save all spectral orders to the HDF5 file format
        """
        for i in range(len(self)):
            self[i].to_HDF5(path, file_basename)

    def stitch(self):
        """Stitch all the spectra together, assuming zero overlap in wavelength.  
        """
        log.warning("Experimental method")
        wls = np.hstack([self[i].wavelength for i in range(len(self))])
        fluxes = np.hstack([self[i].flux for i in range(len(self))])
        # unc = np.hstack([self[i].uncertainty.array for i in range(len(self))])
        # unc_out = StdDevUncertainty(unc)

        return KeckNIRSPECSpectrum(spectral_axis=wls, flux=fluxes)

    def plot(self, **kwargs):
        """Plot the entire spectrum list
        """
        ax = self[0].plot(figsize=(25, 4), **kwargs)
        for i in range(1, len(self)):
            self[i].plot(ax=ax, **kwargs)

        return ax
