r"""
IGRINS Spectrum
---------------

A container for an IGRINS spectrum of :math:`M=28` total total orders :math:`m`, each with vectors for wavelength flux and uncertainty, e.g. :math:`F_m(\lambda)`.


IGRINSSpectrum
##############
"""
import logging
import warnings
import json
from muler.echelle import EchelleSpectrum, EchelleSpectrumList
from muler.utilities import Slit, concatenate_orders
from astropy.time import Time
import numpy as np
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty
from specutils.manipulation import LinearInterpolatedResampler
LinInterpResampler = LinearInterpolatedResampler()

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


def readPLP(plppath, date, frameno, waveframeno, dim='1D'):
    """Convience function for easily reading in the full IGRINS Spectrum (both H and K bands)
    from the IGRINS PLP output

    Parameters
    ----------
    plppath: string
        Path to the IGRINS PLP (e.g. "/Users/Username/Desktop/plp/")
    date: int or string
        Date for night of IGIRNS observation in format of YYYYMMDD (e.g. "201401023")
    frameno: int or string
        Number of frame denoting target as specified as the first frame in the
        recipes file for the night (e.g. 54 or "0054")
    waveframeno: int or string
        Number of frame denoting target as specified as the first frame in the
        recipes file for the wavelength solution (e.g. 54 or "0054") from a wvlsol_v1 file.
        This is usually the first frame number for the sky.
    dim: string
        Set to "1D" to read in the 1D extracted spectrum from the .spec.fits files
        or "2D" to read in the rectified 2D spectrum from the .spec2d.fits files

    Returns
    -------
    IGRINSSpectrumList containing all the orders for the H and K bands for the specified target
    """
    if type(date) is not str: #Converhet dates and frame numbers to the proper string format
        date = '%.8d' % int(date)
    if type(frameno) is not str:
        frameno = '%.4d' % int(frameno)
    if type(waveframeno) is not str:
        waveframeno = '%.4d' % int(waveframeno)
    if dim.upper() == '1D': #Use proper filename for 1D or 2D extractions
        suffix = '.spec.fits'
    elif dim.upper() == '2D':
        suffix = '.spec2d.fits'
    else:
        raise Exception(
            "Argument 'dim' must be '1D' for .spec.fits files or '2D' for .spec2d.fits files."
            )
    spec_H = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/'+'SDCH_'+date+'_'+frameno+suffix, #Read in H band
                                wavefile=plppath+'calib/primary/'+date +'/SKY_SDCH_'+date+'_'+waveframeno+'.wvlsol_v1.fits')
    spec_K = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/'+'SDCK_'+date+'_'+frameno+suffix, #Read in K band
                                wavefile=plppath+'calib/primary/'+date +'/SKY_SDCK_'+date+'_'+waveframeno+'.wvlsol_v1.fits')
    spec_all = concatenate_orders(spec_H, spec_K) #Combine H and K bands
    return spec_all


def getUncertaintyFilepath(filepath):
    """Returns path for uncertainty file (.variance.fits or .sn.fits)

        Will first search for a .variance.fits file but if that does not exist
        will search for a .sn.fits file.

    Parameters
    ----------
    filepath: Filepath to fits file storing the data.  Can be .spec.fits or .spec_a0v.fits.

    Returns
    -------
    uncertaintyFilepath: string
        Returns the file path to the uncertianity (.variance.fits or .sn.fits) file.

    """
    if ".spec_a0v.fits" in filepath: #Grab base file name for the uncertainty file
        path_base = filepath[:-14]
    elif ".spec_flattened.fits" in filepath:
        path_base = filepath[:-20]
    elif ".spec.fits" in filepath:
        path_base = filepath[:-10]
    elif ".spec2d.fits" in filepath:
        path_base = filepath[:-12]
    if ".spec2d.fits" in filepath:
        if os.path.exists(path_base + '.var2d.fits'):
            return path_base + '.var2d.fits'
        else:
            raise Exception(
                "The file .var2d.fits does not exist in the same path as the spectrum file to get the uncertainty.  Please provide one of these files in the same directory as your spectrum file."
                )             
    else:
        if os.path.exists(path_base + '.variance.fits'): #Prefer .variance.fits file
            return path_base + '.variance.fits'
        elif os.path.exists(path_base + '.sn.fits'): #If no .variance.fits file found, try using the .sn.fits file
            return path_base + '.sn.fits'
        else:
            raise Exception(
                "Neither .variance.fits or .sn.fits exists in the same path as the spectrum file to get the uncertainty.  Please provide one of these files in the same directory as your spectrum file."
                )             

def getSlitProfile(filepath, band, slit_length):
    """Returns the path for the slit profile.  Will first look for a 2D
    spectrum .spec2d.fits file to calculate the profile from.  If a spec2d.fits
    file does not exist, will look for a .slit_profile.json.

    Parameters
    ----------
    filepath: string
        Filepath to fits file storing the data.  Can be .spec.fits or .spec_a0v.fits.
    band: string
        'H' or 'K' specifying which band
    slit_length: float
        Length of the slit on the sky in arcsec.

    Returns
    -------
    x: float
        Distance in arcsec along the slit
    y: float
        Flux of beam profile across the slit
    """
    if ".spec_a0v.fits" in filepath: #Grab base file name for the uncertainty file
        path_base = filepath[:-14]
    elif ".spec_flattened.fits" in filepath:
        path_base = filepath[:-20]
    elif ".spec.fits" in filepath:
        path_base = filepath[:-10]
    elif ".spec2d.fits" in filepath:
        path_base = filepath[:-12]
    path_base = path_base.replace('SDCH', 'SDC'+band).replace('SDCK', 'SDC'+band)
    spec2d_filepath = path_base + '.spec2d.fits'
    json_filepath = path_base + '.slit_profile.json'
    if os.path.exists(filepath): #First try to use the 2D spectrum in a .spec2d.fits file to estimate the slit proflie
        spec2d = fits.getdata(spec2d_filepath)
        long_spec2d = spec2d[2,:,1000:1300] #Chop off order edges at columns 800 and 1200
        for i in range(3, len(spec2d)-2):
            long_spec2d = np.concatenate([long_spec2d, spec2d[i,:,1000:1300]], axis=1)
        y = np.nanmedian(long_spec2d, axis=1)
        x = np.arange(len(y)) * (slit_length / len(y))
    elif os.path.exists(json_filepath): #If no 2D spectrum exists, try using the PLP estimate in .slit_profile.json
        json_file = open(filepath)
        json_obj = json.load(json_file)
        x = np.array(json_obj['profile_x']) * slit_length
        y = np.array(json_obj['profile_y'])
        json_file.close()
    else:
        raise Exception(
            "Need either .spec2d.fits or .slit_profile.json file in the same directory as "
            + filepath
            + " in order to get an estimate of the slit profile.  .spec2d.fits or .slit_profile.json are missing."
        )        
    return x, y



def getIGRINSSlitThroughputABBACoefficients(file, slit_length=14.8, PA=90, guiding_error=1.5, print_info=True, plot=False):
    """Estimate the wavelength dependent fractional slit throughput for a point source nodded ABBA on the IGRINS slit and return the 
    coefficients of a linear fit.

    Parameters
    ----------
    file:
        Path to fits file (e.g. spec.fits) from which the slit_profile.json file is also in the same directory.
        These should all be in the same IGRINS PLP output directory.
    slit_length: float
        Length of the slit on the sky in arcsec.
    PA: float
        Position angle of the slit on the sky in degrees.  Measured counterclockwise from North to East.
    guilding_error: float
        Estimate of the guiding error in arcsec.  This smears out the PSF fits in the East-West direction.
        This should be used carefully and only for telescopes on equitorial mounts.
    print_info: bool
        Print information about the fit.
    plot: bool
        Visualize slit throughput calculations.

    Returns
    -------
    m, b:
        Coefficients for a fit of a linear trend of m*(1/wavelength)+b to the fractional slit throughput with the
        wavelength units in microns.

    """
    igrins_slit = Slit(length=slit_length, width=slit_length*(1/14.8), PA=PA, guiding_error=guiding_error)
    #Get throughput for H band
    x, y = getSlitProfile(file, band='H', slit_length=slit_length) #Get slit profile
    igrins_slit.clear()
    igrins_slit.ABBA(y, x=x, print_info=print_info, plot=plot)
    if plot:
        print('2D plot of H-band')
        igrins_slit.plot2d()
        #breakpoint()
    f_through_slit_H = igrins_slit.estimate_slit_throughput()
    #Get throughput for K band
    x, y = getSlitProfile(file, band='K', slit_length=slit_length) #Get slit profile
    igrins_slit.clear()
    igrins_slit.ABBA(y, x=x, print_info=print_info, plot=plot)
    if plot:
        print('2D plot of K-band')
        igrins_slit.plot2d()
        breakpoint()
    f_through_slit_K = igrins_slit.estimate_slit_throughput()
    #Fit linear trend through slit throughput as function of wavelength and using fitting a line through two points
    m = (f_through_slit_K - f_through_slit_H) / ((1/2.2) - (1/1.65))
    b = f_through_slit_H - m*(1/1.65)
    if print_info:
        # log.info('H-band slit throughput: ', f_through_slit_H)
        # log.info('K-band slit throughput:', f_through_slit_K)
        # log.info('m: ', m)
        # log.info('b: ', b)
        print('H-band slit throughput: ', f_through_slit_H)
        print('K-band slit throughput:', f_through_slit_K)
        print('m: ', m)
        print('b: ', b)
    return m, b



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
        self, *args, file=None, order=10, sn_used = False, cached_hdus=None, wavefile=None, **kwargs
    ):

        # self.ancillary_spectra = None
        self.noisy_edges = (450, 1950)
        self.instrumental_resolution = 45_000.0
        self.file = file

         #False if variance.fits file used for uncertainty, true if sn.fits file used for uncertainty

        if file is not None:
            
            assert (".spec_a0v.fits" in file) or (".spec.fits" in file) or (".spec_flattened.fits" in file) or ('.spec2d.fits' in file)
            # Determine the band
            if "SDCH" in file:
                band = "H"
            elif "SDCK" in file:
                band = "K"
            else:
                raise NameError("Cannot identify file as an IGRINS spectrum")
            grating_order = grating_order_offsets[band] + order

            uncertainty_hdus = None #Default values
            uncertainty = None
            if cached_hdus is not None:
                hdus = cached_hdus[0]
                if "rtell" in file:
                    sn = hdus["SNR"].data[order]
                else:
                    uncertainty_hdus = cached_hdus[1]
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
                if "rtell" not in file and "spec_a0v" not in file:  
                    uncertainty_filepath = getUncertaintyFilepath(file)
                    uncertainty_hdus = fits.open(uncertainty_filepath, memmap=False)   
                    if '.sn.fits' in uncertainty_filepath:
                        sn_used = True
                elif "rtell" in file: #If rtell file is used, grab SNR stored in extension
                    sn = hdus["SNR"].data[order]
                    sn_used = True
            hdr = hdus[0].header
            if ("spec_a0v.fits" in file) and (wavefile is not None):
                log.warn(
                    "You have passed in a wavefile and a spec_a0v format file, which has its own wavelength solution.  Ignoring the wavefile."
                )
                lamb = hdus["WAVELENGTH"].data[order].astype(np.float64) * u.micron
                flux = hdus["SPEC_DIVIDE_A0V"].data[order].astype(np.float64) * u.ct
                try:
                    uncertainty_hdus = [hdus["SPEC_DIVIDE_A0V_VARIANCE"]]
                    sn_used = False
                except:
                    print("Warning: Using older PLP versions of .spec_a0v.fits files which have no variance saved.  Will grab .variance.fits file.")
            elif ".spec_a0v.fits" in file:
                lamb = hdus["WAVELENGTH"].data[order].astype(float) * u.micron
                flux = hdus["SPEC_DIVIDE_A0V"].data[order].astype(float) * u.ct
                try:
                    uncertainty_hdus = [hdus["SPEC_DIVIDE_A0V_VARIANCE"]]
                    sn_used = False
                except:
                    print("Warning: Using older PLP versions of .spec_a0v.fits files which have no variance saved.  Will grab .variance.fits file.")
            elif (("spec.fits" in file) or ("spec_flattened.fits" in file) or ('.spec2d.fits' in file)) and (wavefile is not None):
                lamb = (
                    wave_hdus[0].data[order].astype(float) * u.micron
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
            if uncertainty_hdus is not None or ("rtell" in file):
                if not sn_used: #If .variance.fits used
                    variance = uncertainty_hdus[0].data[order].astype(np.float64)
                    stddev = np.sqrt(variance)
                    if ("rtell" in file): #If using a rtell or spec_a0v file with a variance file, scale the stddev to preserve signal-to-noise
                        unprocessed_flux = hdus["TGT_SPEC"].data[order].astype(np.float64)
                        stddev *= (flux.value / unprocessed_flux)
                else: #Else if .sn.fits (or SNR HDU in rtell file) used
                    if not "rtell" in file:
                        sn = uncertainty_hdus[0].data[order].astype(np.float64)
                    dw = np.gradient(lamb) #Divide out stuff the IGRINS PLP did to calculate the uncertainty per resolution element to get the uncertainty per pixel
                    pixel_per_res_element = (lamb/40000.)/dw
                    sn_per_pixel =  sn / np.sqrt(pixel_per_res_element)
                    stddev = flux.value / sn_per_pixel.value
                uncertainty = StdDevUncertainty(np.abs(stddev))
                mask = np.isnan(flux) | np.isnan(uncertainty.array)
            else:
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
            # if not ("header" in self.meta.keys()):
            #     log.warn(
            #         "The spectrum metadata appears to be missing.  "
            #         "The functionality of muler may be impaired without metadata.  "
            #         "See discussion at https://github.com/OttoStruve/muler/issues/79."
            #     )

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

    def getSlitThroughput(self, slit_length=14.8, PA=90, guiding_error=1.5, print_info=True, plot=False):
        """Estimate the wavelength dependent fractional slit throughput for a point source nodded ABBA on the IGRINS slit.

        Parameters
        ----------
        h_band_slitprofile_filepath:
            Filepath to *.slit_profile.json file outputted by the IGRINS PLP storing the spatial
            profile of the target along the slit for the H band.
        k_band_slitprofile_filepath:
            Filepath to *.slit_profile.json file outputted by the IGRINS PLP storing the spatial
            profile of the target along the slit for the K band.
        slit_length: float
            Length of the slit on the sky in arcsec.
        PA: float
            Position angle of the slit on the sky in degrees.  Measured counterclockwise from North to East.
        guilding_error: float
            Estimate of the guiding error in arcsec.  This smears out the PSF fits in the East-West direction.
            This should be used carefully and only for telescopes on equitorial mounts.
        print_info: bool
            Print information about the fit.

        Returns
        -------
        Returns array of fractional slit throughput as a function of wavelength
        """

        m, b = getIGRINSSlitThroughputABBACoefficients(self.file, slit_length=slit_length, PA=PA, guiding_error=guiding_error, print_info=print_info, plot=plot)
        return m*(1/self.wavelength.um) + b






class IGRINSSpectrumList(EchelleSpectrumList):
    r"""
    An enhanced container for a list of IGRINS spectral orders

    """

    def __init__(self, *args, **kwargs):
        self.file = None
        self.normalization_order_index = 14
        super().__init__(*args, **kwargs)

    @staticmethod
    def read(file, precache_hdus=True, wavefile=None):
        """Read in a SpectrumList from a file

        Parameters
        ----------
        file : (str)
            A path to a reduced IGRINS spectrum from plp.
        wavefile : (str)
            Path to a file storing a wavelength soultion for a night from the plp.
            Wave files are found in the IGRINS PLP callib/primary/DATE/ directory with
            the extension wvlsol_v1.fits.

        """
        # still works
        assert (".spec_a0v.fits" in file) or (".spec.fits" in file) or (".spec_flattened.fits" in file) or (".spec2d.fits" in file)
        
        sn_used = False #Default
        hdus = fits.open(file, memmap=False)

        #hdus["SPEC_DIVIDE_A0V_VARIANCE"] 
        if "SPEC_DIVIDE_A0V_VARIANCE" in hdus:
            cached_hdus = [hdus, [hdus["SPEC_DIVIDE_A0V_VARIANCE"]]] 
        elif "rtell" not in file: #Default, if no rtell file is used
            uncertainty_filepath = getUncertaintyFilepath(file)
            uncertainty_hdus = fits.open(uncertainty_filepath, memmap=False)    
            cached_hdus = [hdus, uncertainty_hdus]   
            if '.sn.fits' in uncertainty_filepath:
                sn_used = True     
        else: #If rtell file is used
            cached_hdus = [hdus]
            sn_used = True
        if wavefile is not None:
            if os.path.exists(wavefile): #Check if user provided path to wavefile exists
                wave_hdus = fits.open(wavefile, memmap=False)
            else: #If not, check file name inside directory from file
                base_path = os.path.dirname(file)
                full_path = base_path + '/' + os.path.basename(wavefile)
                wave_hdus = fits.open(full_path, memmap=False)
            cached_hdus.append(wave_hdus)

        if hdus[0].data is not None:
            hdus0_shape = hdus[0].data.shape #Normally we read from the 0th extension
        else:
            hdus0_shape = hdus[1].data.shape #To insure compatibility with new version of the PLP for spec_a0v.fits files
        if len(hdus0_shape) == 2: #1D spectrum
            n_orders, n_pix = hdus0_shape
        elif len(hdus0_shape) == 3: #2D spectrum
            n_orders, n_height, n_pix = hdus0_shape

        list_out = []
        for i in range(n_orders - 1, -1, -1):
            spec = IGRINSSpectrum(
                file=file, wavefile=wavefile, order=i, sn_used=sn_used, cached_hdus=cached_hdus
            )
            list_out.append(spec)
        specList = IGRINSSpectrumList(list_out)
        specList.file = file
        return specList
    def getSlitThroughput(self, slit_length=14.8, PA=90, guiding_error=1.5,  print_info=True, plot=False):
        """Estimate the wavelength dependent fractional slit throughput for a point source nodded ABBA on the IGRINS slit.

        Parameters
        ----------
        h_band_slitprofile_filepath:
            Filepath to *.slit_profile.json file outputted by the IGRINS PLP storing the spatial
            profile of the target along the slit for the H band.
        k_band_slitprofile_filepath:
            Filepath to *.slit_profile.json file outputted by the IGRINS PLP storing the spatial
            profile of the target along the slit for the K band.
        slit_length: float
            Length of the slit on the sky in arcsec.
        PA: float
            Position angle of the slit on the sky in degrees.  Measured counterclockwise from North to East.
        guilding_error: float
            Estimate of the guiding error in arcsec.  This smears out the PSF fits in the East-West direction.
            This should be used carefully and only for telescopes on equitorial mounts.
        print_info: bool
            Print information about the fit.

        Returns
        -------
        Returns list of arrays of fractional slit throughput as a function of wavelength
        """

        m, b = getIGRINSSlitThroughputABBACoefficients(self.file, slit_length=slit_length, PA=PA, guiding_error=guiding_error, print_info=print_info, plot=plot)
        f_throughput = []
        for i in range(len(self)):
            f_throughput.append(m*(1/self[i].wavelength.um) + b)
        return f_throughput
