r"""
KeckNIRSPEC Spectrum
--------------------

A container for a Keck NIRSPEC high resolution spectrum, for some echelle order :math:`m \in ` out of :math:`M` total orders, each with vectors for wavelength, flux, and uncertainty, e.g. :math:`F_m(\lambda)`.  KeckNIRSPEC already has been sky subtracted, but the subtracted sky signal is contained as the spectrum.sky attribute for reference.


KeckNIRSPECSpectrum
###################
"""

import warnings
import logging
from muler.echelle import EchelleSpectrum, EchelleSpectrumList
import numpy as np
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty
from astropy.constants import R_jup, R_sun, G, M_jup, R_earth, c

# from barycorrpy import get_BC_vel
from astropy.time import Time

import os
import copy


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


class KeckNIRSPECSpectrum(EchelleSpectrum):
    r"""
    A container for Keck NIRSPEC spectra

    Args:
        file (str): A path to a reduced Keck NIRSPEC spectrum from NSDRP
    """

    def __init__(self, *args, file=None, order=63, **kwargs):

        self.site_name = "Keck Observatory"
        self.ancillary_spectra = ["sky"]
        self.noisy_edges = (10, 1000)
        self.instrumental_resolution = 20_000.0

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
                **kwargs,
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
                **kwargs,
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
        return self.meta["header"]["RA"] * u.hourangle

    @property
    def DEC(self):
        """The declination from header files"""
        return self.meta["header"]["DEC"] * u.deg

    @property
    def astropy_time(self):
        """The astropy time based on the header"""
        mjd = self.meta["header"]["MJD-OBS"]
        return Time(mjd, format="mjd", scale="utc")

    def sky_subtract(self, force=False):
        """Subtract sky spectrum from science spectrum

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


class KeckNIRSPECSpectrumList(EchelleSpectrumList):
    r"""
    An enhanced container for a list of KeckNIRSPEC spectral orders

    """

    def __init__(self, *args, **kwargs):
        # Todo: put Keck NIRSPEC specific content and attributes here
        self.normalization_order_index = 0
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
            assert (
                files[i].find("_flux_tbl.fits") != -1
            ), "{} should be a fits_tbl".format(files[i])
            spec = KeckNIRSPECSpectrum(file=files[i])
            list_out.append(spec)
        return KeckNIRSPECSpectrumList(list_out)
