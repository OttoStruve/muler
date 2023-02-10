r"""
IGRINS Spectrum
---------------

A container for an IGRINS spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.


IGRINSSpectrum
##############
"""
import logging
import warnings
from muler.echelle import EchelleSpectrum, EchelleSpectrumList
from astropy.time import Time
import numpy as np
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty

import copy
import os

log = logging.getLogger(__name__)

#  See Issue: https://github.com/astropy/specutils/issues/779
warnings.filterwarnings(
    "ignore", category=astropy.utils.exceptions.AstropyDeprecationWarning
)
warnings.filterwarnings("ignore", category=FITSFixedWarning)
# See Issue: https://github.com/astropy/specutils/issues/800
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Convert PLP index number to echelle order m
## Note that these technically depend on grating temperature
## For typical operating temperature, offsets should be exact.
grating_order_offsets = {"H": 98, "K": 71}


class IGRINSSpectrum(EchelleSpectrum):
    r"""
    A container for IGRINS spectra

    Args:
        file (str): A path to a reduced IGRINS spectrum from plp of file type .spec.fits
            or .spec_a0v.fits.
        order (int): which spectral order to read
        cached_hdus (list) :
            List of two or three fits HDUs, one for the spec.fits/spec_a0v.fits, one for the
            variance.fits file, and one optional one for the .wave.fits file
            to reduce file I/O for multiorder access.
            If provided, must give both (or three) HDUs.  Optional, default is None.
        wavefile (str):  A path to a reduced IGRINS spectrum storing the wavelength solution
            of file type .wave.fits.
    """

    def __init__(
        self, *args, file=None, order=10, cached_hdus=None, wavefile=None, **kwargs
    ):

        # self.ancillary_spectra = None
        self.noisy_edges = (450, 1950)
        self.instrumental_resolution = 45_000.0
        sn_fits_used = False #False if variance.fits file used for uncertainity, true if sn.fits file used for uncertainity

        if file is not None:
            assert (".spec_a0v.fits" in file) or (".spec.fits" in file) or (".spec_flattened.fits")
            # Determine the band
            if "SDCH" in file:
                band = "H"
            elif "SDCK" in file:
                band = "K"
            else:
                raise NameError("Cannot identify file as an IGRINS spectrum")
            grating_order = grating_order_offsets[band] + order

            if cached_hdus is not None:
                hdus = cached_hdus[0]
                if "rtell" in file:
                    sn = hdus["SNR"].data[order]
                    uncertainity_hdus = None
                else:
                    uncertainity_hdus = cached_hdus[1]
                if wavefile is not None:
                    wave_hdus = cached_hdus[2]
            else: #Read in files if cached_hdus are not provided
                hdus = fits.open(str(file))
                if wavefile is not None:
                    if os.path.exists(wavefile): #Check if user provided path to wavefile exists
                        wave_hdus = fits.open(wavefile)
                    else: #If not, check file name inside directory from file
                        base_path = os.path.dirname(file)
                        full_path = base_path + '/' + os.path.basename(wavefile)
                        wave_hdus = fits.open(full_path)
                if "rtell" not in file:           
                    if ".spec_a0v.fits" in file: #Grab base file name for the uncertainity file
                        uncertainity_file_base = file[:-13]
                    elif ".spec_flattened.fits" in file:
                        uncertainity_file_base = file[:-19]
                    elif ".spec.fits" in file:
                        uncertainity_file_base = file[:-9] 
                    uncertainity_file_sn = uncertainity_file_base + 'sn.fits' #Construct file names to check for both possible sn.fits and variance.fits uncertainity files
                    uncertainity_file_variance = uncertainity_file_base + 'variance.fits'
                    if os.path.exists(uncertainity_file_variance): #Check if variance.fits exists, if it does use that for the uncertainity
                        uncertainity_hdus = fits.open(uncertainity_file_variance)
                    elif os.path.exists(uncertainity_file_sn): #If variance.fits does not exist, try using sn.fits for the uncertainity
                        uncertainity_hdus = fits.open(uncertainity_file_sn)
                        sn_fits_used = True #true if sn.fits file used for uncertainity, else it is false
                    else:
                        raise Exception(
                            "Neither variance.fits or sn.fits exists to get the uncertainity.  Please provide one of these files in the same directory as your spectrum file."
                            )             
                else:
                    sn = hdus["SNR"].data[order]
                    uncertainity_hdus = None
            hdr = hdus[0].header
            if ("spec_a0v.fits" in file) and (wavefile is not None):
                log.warn(
                    "You have passed in a wavefile and a spec_a0v format file, which has its own wavelength solution.  Ignoring the wavefile."
                )
                lamb = hdus["WAVELENGTH"].data[order].astype(np.float64) * u.micron
                flux = hdus["SPEC_DIVIDE_A0V"].data[order].astype(np.float64) * u.ct
            elif ".spec_a0v.fits" in file:
                lamb = hdus["WAVELENGTH"].data[order].astype(float) * u.micron
                flux = hdus["SPEC_DIVIDE_A0V"].data[order].astype(float) * u.ct
            elif (("spec.fits" in file) or ("spec_flattened.fits" in file)) and (wavefile is not None):
                lamb = (
                    wave_hdus[0].data[order].astype(float) * 1e-3 * u.micron
                )  # Note .wave.fits and .wavesol_v1.fts files store their wavelenghts in nm so they need to be converted to microns
                flux = hdus[0].data[order].astype(float) * u.ct
            elif (("spec.fits" in file) or ("spec_flattened.fits" in file)) and (wavefile is None):
                raise Exception(
                    "wavefile must be specified when passing in spec.fits files, which do not come with an in-built wavelength solution."
                )
            else:
                raise Exception(
                    "File "
                    + file
                    + " is the wrong file type.  It must be either .spec_a0v.fits or .spec.fits."
                )
            meta_dict = {
                "x_values": np.arange(0, 2048, 1, dtype=int),
                "m": grating_order,
                "header": hdr,
            }
            if "rtell" in file:
                stddev_per_resolution_element = np.abs(flux / sn) #Get uncertainity per resoultion element
                dw = np.gradient(lamb) #Divide out stuff the IGRINS PLP did to calculate the uncertainity per resolution element to get the uncertainity per pixel
                pixel_per_res_element = (lamb/40000.)/dw
                stddev = np.abs(sn / flux)
                stddev = stddev_per_resolution_element / pixel_per_res_element
                uncertainty = StdDevUncertainty(stddev)
                mask = np.isnan(flux) | np.isnan(uncertainty.array)
            elif uncertainity_hdus is not None:
                if not sn_fits_used: #If .variance.fits used
                    variance = uncertainity_hdus[0].data[order].astype(np.float64)
                    stddev = np.sqrt(variance)
                else: #Else if .sn.fits used
                    sn = uncertainity_hdus[0].data[order].astype(np.float64)
                    stddev_per_resolution_element = np.abs(flux / sn) #Get uncertainity per resoultion element
                    dw = np.gradient(lamb) #Divide out stuff the IGRINS PLP did to calculate the uncertainity per resolution element to get the uncertainity per pixel
                    pixel_per_res_element = (lamb/40000.)/dw
                    stddev = stddev_per_resolution_element / pixel_per_res_element
                uncertainty = StdDevUncertainty(stddev)
                mask = np.isnan(flux) | np.isnan(uncertainty.array)
            else:
                uncertainity = None
                mask = np.isnan(flux)
            super().__init__(
                spectral_axis=lamb.to(u.Angstrom),
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs,
            )
        else:
            super().__init__(*args, **kwargs)
            if not ("header" in self.meta.keys()):
                log.warn(
                    "The spectrum metadata appears to be missing.  "
                    "The functionality of muler may be impaired without metadata.  "
                    "See discussion at https://github.com/OttoStruve/muler/issues/79."
                )

    @property
    def site_name(self):
        """Which pipeline does this spectrum originate from?"""
        # TODO: add a check lookup dictionary for other telescopes
        # to ensure astropy compatibility
        return self.meta["header"]["TELESCOP"]

    @property
    def ancillary_spectra(self):
        """The list of conceivable ancillary spectra"""
        return []

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


class IGRINSSpectrumList(EchelleSpectrumList):
    r"""
    An enhanced container for a list of IGRINS spectral orders

    """

    def __init__(self, *args, **kwargs):
        self.normalization_order_index = 14
        super().__init__(*args, **kwargs)

    @staticmethod
    def read(file, precache_hdus=True, wavefile=None):
        """Read in a SpectrumList from a file

        Parameters
        ----------
        file : (str)
            A path to a reduced IGRINS spectrum from plp
        wavefile : (str)

        """
        # still works
        assert (".spec_a0v.fits" in file) or (".spec.fits" in file) or (".spec_flattened.fits" in file)
        hdus = fits.open(file, memmap=False)
        if ".spec_a0v.fits" in file:
            variance_file = file[:-13] + "variance.fits"
        elif ".spec.fits" in file:
            variance_file = file[:-9] + "variance.fits"
        elif ".spec_flattened.fits" in file:
            variance_file = file[:-19] + "variance.fits"
        if "rtell" not in file:
            variance_hdus = fits.open(variance_file, memmap=False)
            cached_hdus = [hdus, variance_hdus]
        else:
            cached_hdus = [hdus]
        if wavefile is not None:
            if os.path.exists(wavefile): #Check if user provided path to wavefile exists
                wave_hdus = fits.open(wavefile, memmap=False)
            else: #If not, check file name inside directory from file
                base_path = os.path.dirname(file)
                full_path = base_path + '/' + os.path.basename(wavefile)
                wave_hdus = fits.open(full_path, memmap=False)
            cached_hdus.append(wave_hdus)

        n_orders, n_pix = hdus[0].data.shape

        list_out = []
        for i in range(n_orders - 1, -1, -1):
            spec = IGRINSSpectrum(
                file=file, wavefile=wavefile, order=i, cached_hdus=cached_hdus
            )
            list_out.append(spec)
        return IGRINSSpectrumList(list_out)

