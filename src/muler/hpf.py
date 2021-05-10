r"""
HPF Spectrum
---------------

A container for an HPF spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.  HPF additionally has a sky fiber and optionally a Laser Frequency Comb fiber.  Our experimental API currently ignores the LFC fiber.  The sky fiber can be accessed by passing the `sky=True` kwarg when retrieving the


HPFSpectrum
##############
"""

import warnings
import numpy as np
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty
from scipy.stats import median_abs_deviation
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from astropy.constants import R_jup, R_sun, G, M_jup, R_earth, c

# from barycorrpy import get_BC_vel
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

# from barycorrpy.utils import get_stellar_data

# from specutils.io.registers import data_loader
from celerite2 import terms
import celerite2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import copy

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

# Convert FITS running index number to echelle order m
grating_order_offsets = {"Goldilocks": 0, "HPF": 0}  # Not implemented yet


class HPFSpectrum(Spectrum1D):
    r"""
    A container for HPF spectra

    Args:
        file (str): A path to a reduced HPF spectrum from Goldilocks *or* the HPF instrument team
        order (int): which spectral order to read
        cached_hdus (list) :
            A pre-loaded HDU to reduce file I/O for multiorder access.
            If provided, must give both HDUs.  Optional, default is None.
    """

    def __init__(
        self,
        *args,
        file=None,
        order=19,
        cached_hdus=None,
        sky=False,
        lfc=False,
        **kwargs
    ):

        if file is not None:
            if "Goldilocks" in file:
                pipeline = "Goldilocks"
            elif "Slope" in file:
                pipeline = "HPF"
            else:
                raise NameError("Cannot identify file as an HPF spectrum")
            grating_order = grating_order_offsets[pipeline] + order

            if cached_hdus is not None:
                hdus = cached_hdus[0]
            else:
                hdus = fits.open(str(file))
            hdr = hdus[0].header

            if sky:
                lamb = hdus[8].data[order].astype(np.float64) * u.AA
                flux = hdus[2].data[order].astype(np.float64) * u.ct
                unc = hdus[5].data[order].astype(np.float64) * u.ct
            elif lfc:
                lamb = hdus[9].data[order].astype(np.float64) * u.AA
                flux = hdus[3].data[order].astype(np.float64) * u.ct
                unc = hdus[6].data[order].astype(np.float64) * u.ct
            else:
                lamb = hdus[7].data[order].astype(np.float64) * u.AA
                flux = hdus[1].data[order].astype(np.float64) * u.ct
                unc = hdus[4].data[order].astype(np.float64) * u.ct

            (barycorr, lfccorr) = self._estimate_barycorr(hdr, pipeline)

            meta_dict = {
                "x_values": np.arange(0, 2048, 1, dtype=np.int),
                "pipeline": pipeline,
                "m": grating_order,
                "header": hdr,
                "BCcorr": barycorr,
                "LFCcorr": lfccorr,
            }

            uncertainty = StdDevUncertainty(unc)
            mask = (
                np.isnan(flux) | np.isnan(uncertainty.array) | (uncertainty.array <= 0)
            )

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=WCS(hdr),
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs
            )
        else:
            super().__init__(*args, **kwargs)

    def _estimate_barycorr(self, hdr, pipeline):
        """Estimate the Barycentric Correction from the Date and Target Coordinates
        
        Parameters
        ----------
        hdr : FITS HDU header
            The FITS header from either pipeline
        pipeline:
            Which HPF pipeline

        Returns
        -------
        barycentric_corrections : (float, float)
            Tuple of floats for the barycentric corrections for target and LFC
        """
        ## Compute RV shifts
        time_obs = hdr["DATE-OBS"]
        obstime = Time(time_obs, format="isot", scale="utc")
        obstime.format = "jd"

        ## TODO: Which is the right RA, Dec to put here?
        ## QRA and QDEC is also available.  Which is correct?
        RA = hdr["RA"]
        DEC = hdr["DEC"]

        if pipeline == "Goldilocks":
            lfccorr = hdr["LRVCORR"] * u.m / u.s
        else:
            lfccorr = 0.0 * u.m / u.s

        loc = EarthLocation.from_geodetic(
            -104.0147, 30.6814, height=2025.0
        )  # HET coordinates
        sc = SkyCoord(ra=RA, dec=DEC, unit=(u.hourangle, u.deg))
        barycorr = sc.radial_velocity_correction(obstime=obstime, location=loc)
        return (barycorr, lfccorr)

    def normalize(self):
        """Normalize spectrum by its median value

        Returns
        -------
        normalized_spec : (HPFSpectrum)
            Normalized Spectrum
        """
        median_flux = np.nanmedian(self.flux)
        # median_sky = np.nanmedian(self.sky)

        return self.divide(median_flux, handle_meta="first_found")

    def sky_subtract(self, sky):
        """Subtract science spectrum from sky spectrum

        Note: This operation does not wavelength shift or scale the sky spectrum

        Parameters
        ----------
        sky : vector
            The sky spectrum to subtract from the target spectrum.

        Returns
        -------
        sky_subtractedSpec : (HPFSpectrum)
            Sky subtracted Spectrum
        """
        new_flux = self.flux - sky

        return HPFSpectrum(
            spectral_axis=self.wavelength,
            flux=new_flux,
            meta=self.meta,
            mask=self.mask,
            uncertainty=self.uncertainty,
        )

    def measure_ew(self):
        """Measure the equivalent width of a given spectrum
        
        Returns
        -------
        equivalent width : (scalar)"""
        return 0 # for now

    def blaze_divide_spline(self):
        """Remove blaze function from spectrum by interpolating a spline function

        Returns
        -------
        blaze corrrected spectrum : (HPFSpectrum)
        """
        new_spec = self.normalize()
        spline = UnivariateSpline(self.wavelength, np.nan_to_num(new_spec.flux), k=5)
        interp_spline = spline(self.wavelength)

        no_blaze = new_spec / interp_spline

        return HPFSpectrum(
            spectral_axis=self.wavelength,
            flux=no_blaze.flux,
            meta=self.meta,
            mask=self.mask,
        )

    def blaze_subtract_flats(self, flat, order=19):
        """Remove blaze function from spectrum by subtracting by flat spectrum

        Returns
        -------
        blaze corrrected spectrum using flat fields : (HPFSpectrum)

        """
        new_flux = self.normalize()

        flat_wv = flat[0]
        flat_flux = flat[1]
        if len(flat) == 2:
            flat_err = flat[2]

        master_flat = flat_flux[order] / np.nanmedian(flat_flux[order])

        flat_spline = InterpolatedUnivariateSpline(
            flat_wv[order], np.nan_to_num(master_flat), k=5
        )
        interp_flat = flat_spline(self.wavelength)

        no_flat = new_flux / interp_flat

        return HPFSpectrum(
            spectral_axis=self.wavelength,
            flux=no_flat.flux,
            meta=self.meta,
            mask=self.mask,
        )

    def shift_spec(self, absRV=0):
        """shift spectrum by barycenter velocity

        Returns
        -------
        barycenter corrected Spectrum : (HPFSpectrum)
        """
        meta_out = copy.deepcopy(self.meta)

        bcRV = meta_out["BCcorr"]
        lfcRV = meta_out["LFCcorr"]
        absRV = absRV * u.m / u.s

        vel = bcRV + lfcRV + absRV

        new_wave = self.wavelength * (1.0 + (vel.value / c.value))

        return HPFSpectrum(
            spectral_axis=new_wave,
            flux=self.flux,
            mask=self.mask,
            uncertainty=self.uncertainty,
            meta=meta_out,
        )

    def remove_nans(self):
        """Remove data points that have NaN fluxes

        Returns
        -------
        finite_spec : (HPFSpectrum)
            Spectrum with NaNs removed
        """
        if self.uncertainty is not None:
            masked_unc = StdDevUncertainty(self.uncertainty.array[~self.mask])
        else:
            masked_unc = None

        meta_out = copy.deepcopy(self.meta)
        meta_out["x_values"] = meta_out["x_values"][~self.mask]

        return HPFSpectrum(
            spectral_axis=self.wavelength[~self.mask],
            flux=self.flux[~self.mask],
            mask=self.mask[~self.mask],
            uncertainty=masked_unc,
            meta=meta_out,
        )

    def smooth_spectrum(self):
        """Smooth the spectrum using Gaussian Process regression

        Returns
        -------
        smoothed_spec : (HPFSpectrum)
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

        return HPFSpectrum(
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
        clean_spec : (HPFSpectrum)
            Cleaned version of input Spectrum
        """
        residual = self.flux - self.smooth_spectrum().flux
        mad = median_abs_deviation(residual.value)
        mask = np.abs(residual.value) > threshold * mad

        spectrum_out = copy.deepcopy(self)
        spectrum_out._mask = mask
        spectrum_out.flux[mask] = np.NaN

        return spectrum_out.remove_nans()

    def trim_edges(self, limits=(450, 1950)):
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
        trimmed_spec : (HPFSpectrum)
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

        return HPFSpectrum(
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


class HPFSpectrumList(SpectrumList):
    r"""
    An enhanced container for a list of HPF spectral orders

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def read(file, precache_hdus=True):
        """Read in a SpectrumList from a file

        Parameters
        ----------
        file : (str)
            A path to a reduced HPF spectrum from plp
        """
        assert ".spectra.fits" in file

        hdus = fits.open(file, memmap=False)
        cached_hdus = [hdus]  # , sn_hdus]

        n_orders, n_pix = hdus[7].data.shape

        list_out = []
        for i in range(n_orders):
            spec = HPFSpectrum(file=file, order=i, cached_hdus=cached_hdus)
            list_out.append(spec)
        return HPFSpectrumList(list_out)

    def normalize(self):
        """Normalize the all spectra to order 14's median
        """
        median_flux = copy.deepcopy(np.nanmedian(self[14].flux))
        for i in range(len(self)):
            self[i] = self[i].divide(median_flux, handle_meta="first_found")

        return self

    # def sky_subtract(self):
    #     """Sky subtract all orders
    #     """
    #     flux = copy.deepcopy(self.flux)
    #     sky = copy.deepcopy(self.sky)
    #     for i in range(len(self)):
    #         self[i] = flux[i] - sky[i]

    #     return self

    def remove_nans(self):
        """Remove all the NaNs
        """
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

    def plot(self, **kwargs):
        """Plot the entire spectrum list
        """
        ax = self[0].plot(figsize=(25, 4), **kwargs)
        for i in range(1, len(self)):
            self[i].plot(ax=ax, **kwargs)

        return ax
