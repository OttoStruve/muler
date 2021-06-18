r"""
IGRINS Spectrum
---------------

A container for an IGRINS spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.


IGRINSSpectrum
##############
"""

import warnings
from astropy.time import Time
import numpy as np
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty

import copy
from muler.echelle import EchelleSpectrum

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

# Convert PLP index number to echelle order m
## Note that these technically depend on grating temperature
## For typical operating temperature, offsets should be exact.
grating_order_offsets = {"H": 98, "K": 71}


class IGRINSSpectrum(EchelleSpectrum):
    r"""
    A container for IGRINS spectra

    Args:
        file (str): A path to a reduced IGRINS spectrum from plp
        order (int): which spectral order to read
        cached_hdus (list) :
            List of two fits HDUs, one for the spec_a0v.fits, and one for the
            sn.fits file, to reduce file I/O for multiorder access.
            If provided, must give both HDUs.  Optional, default is None.
    """

    def __init__(self, *args, file=None, order=10, cached_hdus=None, **kwargs):

        self.ancillary_spectra = None
        self.noisy_edges = (450, 1950)
        self.default_resolution = 45_000.0

        if file is not None:
            # Determine the band
            if "SDCH" in file:
                band = "H"
            elif "SDCK" in file:
                band = "K"
            else:
                raise NameError("Cannot identify file as an IGRINS spectrum")
            grating_order = grating_order_offsets[band] + order

            sn_file = file[:-13] + "sn.fits"
            if cached_hdus is not None:
                hdus = cached_hdus[0]
                sn_hdus = cached_hdus[1]
            else:
                hdus = fits.open(str(file))
                try:
                    sn_hdus = fits.open(sn_file)
                except:
                    sn_hdus = None
            hdr = hdus[0].header
            lamb = hdus["WAVELENGTH"].data[order].astype(np.float64) * u.micron
            flux = hdus["SPEC_DIVIDE_A0V"].data[order].astype(np.float64) * u.ct
            meta_dict = {
                "x_values": np.arange(0, 2048, 1, dtype=np.int),
                "m": grating_order,
                "header": hdr,
            }
            if sn_hdus is not None:
                sn = sn_hdus[0].data[10]
                unc = np.abs(flux / sn)
                uncertainty = StdDevUncertainty(unc)
                mask = np.isnan(flux) | np.isnan(uncertainty.array)
            else:
                uncertainty = None
                mask = np.isnan(flux)

            super().__init__(
                spectral_axis=lamb.to(u.Angstrom),
                flux=flux,
                mask=mask,
                wcs=WCS(hdr),
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs,
            )
        else:
            super().__init__(*args, **kwargs)

    @property
    def site_name(self):
        """Which pipeline does this spectrum originate from?"""
        # TODO: add a check lookup dictionary for other telescopes
        # to ensure astropy compatibility
        return self.meta["header"]["TELESCOP"]

    @property
    def RA(self):
        """The right ascension from header files"""
        return self.meta["header"]["OBJRA"] * u.deg

    @property
    def DEC(self):
        """The declination from header files"""
        return self.meta["header"]["OBJDEC"] * u.deg

    @property
    def astropy_time(self):
        """The astropy time based on the header"""
        mjd = self.meta["header"]["MJD-OBS"]
        return Time(mjd, format="mjd", scale="utc")


class IGRINSSpectrumList(SpectrumList):
    r"""
    An enhanced container for a list of IGRINS spectral orders

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def read(file, precache_hdus=True):
        """Read in a SpectrumList from a file

        Parameters
        ----------
        file : (str)
            A path to a reduced IGRINS spectrum from plp
        """
        assert ".spec_a0v.fits" in file
        hdus = fits.open(file, memmap=False)
        sn_file = file[:-13] + "sn.fits"
        sn_hdus = fits.open(sn_file, memmap=False)
        cached_hdus = [hdus, sn_hdus]

        n_orders, n_pix = hdus["WAVELENGTH"].data.shape

        list_out = []
        for i in range(n_orders):
            spec = IGRINSSpectrum(file=file, order=i, cached_hdus=cached_hdus)
            list_out.append(spec)
        return IGRINSSpectrumList(list_out)

    def normalize(self):
        """Normalize the all spectra to order 14's median
        """
        median_flux = copy.deepcopy(np.nanmedian(self[14].flux))
        for i in range(len(self)):
            self[i] = self[i].divide(median_flux, handle_meta="first_found")

        return self

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
