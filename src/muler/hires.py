r"""
KeckHIRES Spectrum
--------------------

A container for a Keck HIRES high resolution spectrum, for some echelle order :math:`m \in ` out of :math:`M` total orders, each with vectors for wavelength, flux, and uncertainty, e.g. :math:`F_m(\lambda)`. 


KeckHIRESSpectrum
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


class KeckHIRESSpectrum(EchelleSpectrum):
    r"""
    A container for Keck HIRES spectra

    Args:
        file (str): A path to a reduced Keck HIRES spectrum from KOA
    """

    def __init__(self, *args, file=None, **kwargs):

        self.site_name = "Keck Observatory"
        # self.ancillary_spectra = ["sky"]
        self.noisy_edges = (6, 4050)
        self.instrumental_resolution = 48_000.0
        self.deckname = ""

        if file is not None:
            file_basename = os.path.basename(file)
            assert (
                file_basename[0:3] == "HI."
            ), "Only KoA spectra are currently supported"
            pipeline = "KoA"
            assert (
                "_flux.fits" in file_basename
            ), "Only fits files are currently supported"
            file_stem = file_basename.split("_flux")[0]
            grating_order = int(file_stem[-2:])

            assert os.path.exists(file), "The file must exist"

            hdu = fits.open(file)
            hdu0 = hdu[1]

            ## Target Spectrum
            lamb = hdu0.data["wave"].astype(float) * u.AA
            flux = hdu0.data["Flux"].astype(float) * u.ct
            unc = hdu0.data["Error"].astype(float) * u.ct

            uncertainty = StdDevUncertainty(unc)
            mask = np.array(
                (
                    np.isnan(flux)
                    | np.isnan(uncertainty.array)
                    | (uncertainty.array <= 0)
                ),
                dtype=bool,
            )

            # Attempt to read-in the header:
            hdr = hdu[0].header

            meta_dict = {
                "x_values": hdu0.data["col"].astype(int),
                "pipeline": pipeline,
                "m": grating_order,
                "header": hdr,
            }

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask.astype(bool),
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs,
            )

            ## Sky Spectrum
            flux = hdu0.data["Background"].astype(float) * u.ct

            sky_spectrum = KeckHIRESSpectrum(
                spectral_axis=lamb,
                flux=flux,
                mask=mask.astype(bool),
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict.copy(),
                **kwargs,
            )

            self.meta["sky"] = sky_spectrum

            ## Flat Spectrum
            flux = hdu0.data["Flat"].astype(float) * u.ct

            flat_spectrum = KeckHIRESSpectrum(
                spectral_axis=lamb,
                flux=flux,
                mask=mask.astype(bool),
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict.copy(),
                **kwargs,
            )

            self.meta["flat"] = flat_spectrum

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
    def ancillary_spectra(self):
        """The list of conceivable ancillary spectra"""
        return ["sky"]

    @property
    def flat(self):
        """Flat spectrum stored as its own object"""
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
        mjd = self.meta["header"]["MJD"]
        return Time(mjd, format="mjd", scale="utc")

    def sky_subtract(self, force=False):
        """Subtract sky spectrum from science spectrum

        Returns
        -------
        sky_subtractedSpec : (KeckHIRESSpectrum)
            Sky subtracted Spectrum
        """
        if force:
            log.warn(
                "HIRES data are already natively sky subtracted! "
                "Proceeding with a forced sky subtraction anyways..."
            )
            return self.subtract(self.sky, handle_meta="first_found")
        else:
            log.error(
                "HIRES data are already natively sky subtracted! "
                "To proceed anyway, state `force=True`."
            )
            return self


class KeckHIRESSpectrumList(EchelleSpectrumList):
    r"""
    An enhanced container for a list of KeckHIRES spectral orders

    """

    def __init__(self, *args, **kwargs):
        # Todo: put Keck HIRES specific content and attributes here
        self.normalization_order_index = 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def read(files):
        """Read in a SpectrumList from a file

        Parameters
        ----------
        file : (str)
            A path to a reduced KeckHIRES spectrum from KoA
        """
        n_orders = len(files)

        list_out = []
        for i in range(n_orders):
            assert files[i].find("_flux.fits") != -1, "{} should be a fits".format(
                files[i]
            )
            spec = KeckHIRESSpectrum(file=files[i])
            list_out.append(spec)
        return KeckHIRESSpectrumList(list_out)
