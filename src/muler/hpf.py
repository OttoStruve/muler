r"""
HPF Spectrum
---------------

A container for an HPF spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.  HPF additionally has a sky fiber and optionally a Laser Frequency Comb fiber.  Our experimental API currently ignores the LFC fiber.  The sky fiber can be accessed by passing the `sky=True` kwarg when retrieving the


HPFSpectrum
##############
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
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.constants import R_jup, R_sun, G, M_jup, R_earth, c
from astropy.time import Time
import copy
from importlib_resources import files
from specutils.manipulation import LinearInterpolatedResampler
from . import templates
import pandas as pd

log = logging.getLogger(__name__)

for category in [
    astropy.utils.exceptions.AstropyDeprecationWarning,
    FITSFixedWarning,
    RuntimeWarning,
]:
    warnings.filterwarnings("ignore", category=category)


# Convert FITS running index number to echelle order m
grating_order_offsets = {"Goldilocks": 0, "HPF": 0}  # Not implemented yet

# Blaze function template
static_blaze_file = files(templates).joinpath("HPF_blaze_templates.csv")
STATIC_BLAZE_DATAFRAME = pd.read_csv(static_blaze_file)


class HPFSpectrum(EchelleSpectrum):
    r"""
    A container for HPF spectra

    Args:
        file (str): A path to a reduced HPF spectrum from Goldilocks *or* the HPF instrument team
        order (int): which spectral order to read
        cached_hdus (list) :
            A pre-loaded HDU to reduce file I/O for multiorder access.
            If provided, must give both HDUs.  Optional, default is None.
    """

    def __init__(self, *args, file=None, order=19, cached_hdus=None, **kwargs):

        self.site_name = "mcdonald"
        self.ancillary_spectra = ["sky", "lfc"]
        self.noisy_edges = (3, 2045)
        self.instrumental_resolution = 55_000.0

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

            ## Target Spectrum
            lamb = hdus[7].data[order].astype(np.float64) * u.AA
            flux = hdus[1].data[order].astype(np.float64) * u.ct
            unc = hdus[4].data[order].astype(np.float64) * u.ct

            meta_dict = {
                "x_values": np.arange(0, 2048, 1, dtype=np.int),
                "pipeline": pipeline,
                "m": grating_order,
                "header": hdr,
            }

            uncertainty = StdDevUncertainty(unc)
            mask = (
                np.isnan(flux) | np.isnan(uncertainty.array) | (uncertainty.array <= 0)
            )

            super().__init__(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs,
            )

            ## Sky Spectrum
            lamb = hdus[8].data[order].astype(np.float64) * u.AA
            flux = hdus[2].data[order].astype(np.float64) * u.ct
            unc = hdus[5].data[order].astype(np.float64) * u.ct
            uncertainty = StdDevUncertainty(unc)
            mask = (
                np.isnan(flux) | np.isnan(uncertainty.array) | (uncertainty.array <= 0)
            )
            sky_spectrum = HPFSpectrum(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict.copy(),
                **kwargs,
            )

            ## LFC Spectrum
            lamb = hdus[9].data[order].astype(np.float64) * u.AA
            flux = hdus[3].data[order].astype(np.float64) * u.ct
            unc = hdus[6].data[order].astype(np.float64) * u.ct
            uncertainty = StdDevUncertainty(unc)
            mask = (
                np.isnan(flux) | np.isnan(uncertainty.array) | (uncertainty.array <= 0)
            )
            lfc_spectrum = HPFSpectrum(
                spectral_axis=lamb,
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict.copy(),
                **kwargs,
            )

            ## We could optionally enable lfc and sky metadata for these referece spectra
            ## That's slightly redundant, it enables antipatterns like:
            # `spectrum.sky.lfc` rather than simply `spectrum.lfc`

            # sky_spectrum.meta["lfc"] = lfc_spectrum
            # lfc_spectrum.meta["sky"] = sky_spectrum

            sky_spectrum.meta["provenance"] = "Sky fiber"
            lfc_spectrum.meta["provenance"] = "Laser Frequency Comb"
            self.meta["provenance"] = "Target fiber"

            self.meta["sky"] = sky_spectrum
            self.meta["lfc"] = lfc_spectrum

        else:
            super().__init__(*args, **kwargs)

    @property
    def provenance(self):
        """What is the provenance of each spectrum?"""
        return self.meta["provenance"]

    @property
    def pipeline(self):
        """Which pipeline does this spectrum originate from?"""
        return self.meta["pipeline"]

    @property
    def spectrographname(self):
        """What's the name of the spectrograph?"""
        return "HPF"

    @property
    def sky(self):
        """Sky fiber spectrum stored as its own HPFSpectrum object"""
        return self.meta["sky"]

    @property
    def lfc(self):
        """Sky fiber spectrum stored as its own HPFSpectrum object"""
        return self.meta["lfc"]

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
        mjd = self.meta["header"]["DATE-OBS"]
        return Time(mjd, format="isot", scale="utc")

    def get_static_blaze_template(self, method="Goldilocks"):
        """Get the static blaze template for HPF, as estimated by Goldilocks
        
        Parameters
        ----------
        file : (str)
            A path to a reduced HPF spectrum from plp
        method : (Str)
            Either "Goldilocks" or "2021_median" (default: Goldilocks)
        """
        type_dict = {"Goldilocks": "blaze_Goldilocks", "2021_median": "blaze_2021"}
        assert method in type_dict.keys()
        blaze_type = type_dict[method]

        # Watch out! Some HPFSpectrum methods *will not work* on this calibration spectrum!
        return HPFSpectrum(
            spectral_axis=STATIC_BLAZE_DATAFRAME.wavelength_Angstrom.values
            * u.Angstrom,
            flux=STATIC_BLAZE_DATAFRAME[blaze_type].values * u.dimensionless_unscaled,
        )

    def _deblaze_by_template(self):
        """Deblazing with a template-based method"""
        blaze_template = self.get_static_blaze_template(method="Goldilocks")
        resampler = LinearInterpolatedResampler()
        resampled_blaze = resampler(blaze_template, self.wavelength)
        return self.divide(resampled_blaze, handle_meta="first_found")

    def deblaze(self, method="template"):
        """Override the default spline deblazing with HPF-custom blaze templates.
        
        Parameters
        ----------
        method : (Str)
            Either "template" or "spline" (default: template)
        """
        if method == "template":
            return self._deblaze_by_template()
        else:
            log.error("This method is deprecated!  Please use the new deblaze method")
            raise NotImplementedError

    def sky_subtract(self):
        """Subtract science spectrum from sky spectrum

        Note: This operation does not wavelength shift or scale the sky spectrum

        Returns
        -------
        sky_subtractedSpec : (HPFSpectrum)
            Sky subtracted Spectrum
        """
        log.warning(
            "This method is known to oversubtract the sky.  Please see GitHub Issues for more info."
        )
        return self.subtract(self.sky, handle_meta="first_found")

    def blaze_divide_flats(self, flat, order=19):
        """Remove blaze function from spectrum by dividing by flat spectrum

        Returns
        -------
        blaze corrrected spectrum using flat fields : (HPFSpectrum)

        """
        log.warning("This method is deprecated!  Please use the new deblaze method")
        raise NotImplementedError


class HPFSpectrumList(EchelleSpectrumList):
    r"""
    An enhanced container for a list of HPF spectral orders

    """

    def __init__(self, *args, **kwargs):
        self.normalization_order_index = 14
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
        cached_hdus = [hdus]

        n_orders, n_pix = hdus[7].data.shape

        list_out = []
        for i in range(n_orders):
            spec = HPFSpectrum(file=file, order=i, cached_hdus=cached_hdus)
            list_out.append(spec)
        return HPFSpectrumList(list_out)

    def deblaze(self):
        """Deblaze the entire spectrum"""
        spec_out = copy.copy(self)
        for i in range(len(spec_out)):
            spec_out[i] = spec_out[i].deblaze()

        return spec_out

    # def sky_subtract(self):
    #     """Sky subtract all orders
    #     """
    #     flux = copy.deepcopy(self.flux)
    #     sky = copy.deepcopy(self.sky)
    #     for i in range(len(self)):
    #         self[i] = flux[i] - sky[i]

    #     return self
