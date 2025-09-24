r"""
MagE Spectrum
---------------

A container for an MagE spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.


MagESpectrum
##############
"""
import logging
import warnings
import astropy
from matplotlib import pyplot as plt
from muler.echelle import EchelleSpectrum, EchelleSpectrumList
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy import units as u
from astropy.nddata import StdDevUncertainty
import glob
import numpy as np


#  See Issue: https://github.com/astropy/specutils/issues/779
warnings.filterwarnings(
    "ignore", category=astropy.utils.exceptions.AstropyDeprecationWarning
)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
# See Issue: https://github.com/astropy/specutils/issues/800
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MagESpectrum(EchelleSpectrum):
    r"""
    A container for MagE spectra

    Args:
        file (str): A path to a reduced MagE spectrum order
    """
    def __init__(self, *args, file=None, **kwargs):
        if file is not None:
            #Open FITS file
            hdul = fits.open(file)
            hdu0 = hdul[0]
            hdr = hdu0.header
            ## Target Spectrum flux
            wcs = WCS(hdr) #Read WCS from header
            x = np.arange(hdu0.data.shape[1]) #Pixel array
            lamb = wcs.pixel_to_world(x,0)[0].value * u.AA #Get wavelength array from wcs stored in header
            flux = hdu0.data.astype(float) * u.ct
            hdul.close()
            #Target spectrum uncertainty
            hdul = fits.open(file.replace('sum.fits', 'sumsig.fits'))
            hdu0 = hdul[0]
            unc = StdDevUncertainty(hdu0.data.astype(float) * u.ct)
            hdul.close()
            mask = np.isnan(flux) | np.isnan(unc.array)

            meta_dict = {
                # "x_values": np.arange(0, 2048, 1, dtype=int),
                "header": hdr,
            }

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=unc,
                meta=meta_dict,
                **kwargs,
            )

        else:
            super().__init__(*args, **kwargs)

class MagESpectrumList(EchelleSpectrumList):
    r"""
    An enhanced container for a list of MagE spectral orders

    """

    def __init__(self, *args, **kwargs):
        self.normalization_order_index = 5
        super().__init__(*args, **kwargs)
    @staticmethod
    def read(path=''):
        target_name = path.split('/')[-1]
        file_paths = sorted(glob.glob(path+'/'+target_name+'*sum.fits'))
        n_orders = len(file_paths)
        list_out = []
        for i in range(n_orders):
            spec = MagESpectrum(
                    file=file_paths[i],
                )
            list_out.append(spec)
        list_out.reverse() #Invert the order
        specList = MagESpectrumList(list_out)
        return specList



