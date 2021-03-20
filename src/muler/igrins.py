r"""
IGRINS Spectrum
---------------

A container for an IGRINS spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.


IGRINSSpectrum
##############
"""

import warnings
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from celerite2 import terms
import celerite2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

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
            if os.path.exists(sn_file):
                hdus = fits.open(sn_file)
                sn = hdus[0].data[10]
                unc = flux / sn
                uncertainty = StdDevUncertainty(unc)
            mask = np.isnan(flux) | np.isnan(unc)

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                uncertainty=uncertainty,
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

        return self.divide(median_flux)

    def remove_nans(self):
        """Remove data points that have NaN fluxes"""
        if self.uncertainty is not None:
            masked_unc = StdDevUncertainty(self.uncertainty.array[~self.mask])
        else:
            masked_unc = None
        return IGRINSSpectrum(
            spectral_axis=self.wavelength[~self.mask],
            flux=self.flux[~self.mask],
            mask=self.mask[~self.mask],
            uncertainty=masked_unc,
        )

    def smooth_spectrum(self):
        """Smooth the spectrum using Gaussian Process regression"""
        if self.uncertainty is not None:
            unc = self.uncertainty.array
        else:
            unc = np.zeros_like(self.flux.value)

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

        return IGRINSSpectrum(
            spectral_axis=self.wavelength,
            flux=mean_model * self.flux.unit,
            mask=np.zeros_like(mean_model, dtype=np.bool),
        )
