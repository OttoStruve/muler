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
import copy

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
            mask = np.isnan(flux)
            super().__init__(spectral_axis=lamb, flux=flux, mask=mask, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def normalize(self):
        """Normalize spectrum by its median value

        Returns:
            (IGRINSSpectrum): Normalized Spectrum
        """
        median_flux = np.nanmedian(self.flux)

        return self.divide(median_flux)
