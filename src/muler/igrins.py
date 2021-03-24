r"""
IGRINS Spectrum
---------------

A container for an IGRINS spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.


IGRINSSpectrum
##############
"""

import warnings
import numpy as np
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from scipy.stats import median_abs_deviation
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

# See Issue: https://github.com/astropy/specutils/issues/800
warnings.filterwarnings("ignore", category=RuntimeWarning)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from specutils import Spectrum1D


class IGRINSSpectrum(Spectrum1D):
    r"""
    A container for IGRINS spectra

    Args:
        file (str): A path to a reduced IGRINS spectrum from plp
        order (int): which spectral order to read
    """

    def __init__(self, *args, file=None, order=10, **kwargs):

        if file is not None:
            hdus = fits.open(str(file))
            lamb = hdus["WAVELENGTH"].data[order].astype(np.float64) * u.micron
            flux = hdus["SPEC_DIVIDE_A0V"].data[order].astype(np.float64) * u.ct
            sn_file = file[:-13] + "sn.fits"
            meta_dict = {"x_values": np.arange(0, 2048, 1, dtype=np.int)}
            if os.path.exists(sn_file):
                hdus = fits.open(sn_file)
                sn = hdus[0].data[10]
                unc = np.abs(flux / sn)
                uncertainty = StdDevUncertainty(unc)
                mask = np.isnan(flux) | np.isnan(uncertainty.array)
            else:
                uncertainty = None
                mask = np.isnan(flux)

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs
            )
        else:
            super().__init__(*args, **kwargs)

    def normalize(self):
        """Normalize spectrum by its median value

        Returns:
            (IGRINSSpectrum): Normalized Spectrum
        """
        median_flux = np.nanmedian(self.flux)

        return self.divide(median_flux, handle_meta="first_found")

    def remove_nans(self):
        """Remove data points that have NaN fluxes
        
        Returns:
            (IGRINSSpectrum): Spectrum with NaNs removed
        """
        if self.uncertainty is not None:
            masked_unc = StdDevUncertainty(self.uncertainty.array[~self.mask])
        else:
            masked_unc = None

        meta_out = copy.deepcopy(self.meta)
        meta_out["x_values"] = meta_out["x_values"][~self.mask]

        return IGRINSSpectrum(
            spectral_axis=self.wavelength[~self.mask],
            flux=self.flux[~self.mask],
            mask=self.mask[~self.mask],
            uncertainty=masked_unc,
            meta=meta_out,
        )

    def smooth_spectrum(self):
        """Smooth the spectrum using Gaussian Process regression
        
        Returns:
            (IGRINSSpectrum): Smooth version of input Spectrum
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

        return IGRINSSpectrum(
            spectral_axis=self.wavelength,
            flux=mean_model * self.flux.unit,
            mask=np.zeros_like(mean_model, dtype=np.bool),
            meta=meta_out,
        )

    def plot(self, ax=None, ylo=0.6, yhi=1.2, figsize=(10, 4), label=None):
        """Plot a quick look of the spectrum"
        
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        ylo : scalar
            Lower limit of the y axis
        yhi : sca;ar
            Upper limit of the y axis
        figsize : tuple
            The figure size for the plot
        label : str
            The legend label to for plt.legend()

        Returns: 
            (`~matplotlib.axes.Axes`): The axis to display and/or modify
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.set_ylim(ylo, yhi)
            ax.set_xlabel("$\lambda \;(\AA)$")
            ax.set_ylabel("Flux")
            ax.step(self.wavelength, self.flux, label=label)
        else:
            ax.step(self.wavelength, self.flux, label=label)

        return ax

    def remove_outliers(self, threshold=5):
        """Remove outliers above threshold
        
        Parameters
        ----------
        threshold : float
            The sigma-clipping threshold (in units of sigma)

        Returns:
            (IGRINSSpectrum): Cleaned version of input Spectrum
        """
        residual = self.flux - self.smooth_spectrum().flux
        mad = median_abs_deviation(residual.value)
        mask = np.abs(residual.value) > threshold * mad

        spectrum_out = copy.deepcopy(self)
        spectrum_out._mask = mask
        spectrum_out.flux[mask] = np.NaN

        return spectrum_out.remove_nans()

    def trim_edges(self, limits=(450, 1950)):
        """Trim the order edges, which tend to be noisy

        This method applies limits on absolute x pixel values, regardless
        of the order of previous destructive operations, which may not
        be the intended behavior in some applications.
        
        Parameters
        ----------
        limits : tuple
            The index bounds (lo, hi) for trimming the order

        Returns:
            (IGRINSSpectrum): Trimmed version of input Spectrum
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

        return IGRINSSpectrum(
            spectral_axis=self.wavelength[~mask],
            flux=self.flux[~mask],
            mask=self.mask[~mask],
            uncertainty=masked_unc,
            meta=meta_out,
        )

    def estimate_uncertainty(self):
        """Estimate the uncertainty based on residual after smoothing
        

        Returns:
            (float): Typical uncertainty
        """
        residual = self.flux - self.smooth_spectrum().flux
        return median_abs_deviation(residual.value)
