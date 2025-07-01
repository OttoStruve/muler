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
from matplotlib import pyplot as plt
from muler.echelle import EchelleSpectrum, EchelleSpectrumList
from muler.utilities import Slit, concatenate_orders, resample_list, roll_along_axis, edge_normalize, isolate_and_normalize_hi_order, round_to_multiple
from astropy.time import Time
import numpy as np
import astropy
from scipy.ndimage import binary_dilation
from astropy.io import fits, ascii
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import StdDevUncertainty
from astroquery.simbad import Simbad
#Simbad.add_votable_fields('flux(V)', 'flux(B)', 'flux(J)', 'flux(H)', 'flux(K)', 'parallax')
Simbad.add_votable_fields('V', 'B', 'J', 'H', 'K', 'parallax')
from specutils.manipulation import LinearInterpolatedResampler
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
LinInterpResampler = LinearInterpolatedResampler()
from gollum.phoenix import PHOENIXGrid, PHOENIXSpectrum
from importlib_resources import files
from . import templates

import copy
import os

log = logging.getLogger("logger")
log.setLevel(logging.DEBUG)



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


def readIGRINS(spec_filepath, wave_filepath=''):
    """Convience function for easily reading in the full IGRINS Spectrum (both H and K bands) given
    the path to a single .spec.fits or .spec2d.fits file and a single wavelength solution file (.wvlsol_v1.fits).
    You only need to provide the path to a file for the H or K band.  It will automatically find the files for the other band.
    The associated .variance.fits or .var2d.fits files will also be automatically read in, if they exist in the same directory.
    Use this to easily read in data downloaded from RRISA.

    Parameters
    ----------    
    spec_filepath: string
        Path to a single spec.fits or spec2d.fits file
        (e.g. "/Path/to/IGRINS/data/SDCH_20220521_0064.spec.fits")
    wave_filepath: string (optional)
         Path to a single wavelength solution file (.wvlsol_v1.fits)
        (e.g. "/Path/to/IGRINS/data/SKY_SDCH_20220521_0055.wvlsol_v1.fits")
        The default behavior is to use the wavelength solution stored in the .spec.fits .spec2d.fits or .spec_a0v.fits
        but the user can provide their own wavelength solution here

    """
    spec_filename = spec_filepath.split('/')[-1] #To handle only changing the band in the filename, not any paths
    spec_filepath = spec_filepath.split(spec_filename)[0]
    if wave_filepath != '': #Use user specified wavelength solution
        wave_filename = wave_filepath.split('/')[-1]
        wave_filepath = wave_filepath.split(wave_filename)[0]
        spec_H = IGRINSSpectrumList.read(spec_filepath+spec_filename.replace('SDCK_', 'SDCH_').replace('_K.', '_H.'), #Read in H band
                                wavefile=wave_filepath+wave_filename.replace('SDCK_', 'SDCH_').replace('_K.', '_H.'))
        spec_K = IGRINSSpectrumList.read(spec_filepath+spec_filename.replace('SDCH_', 'SDCK_').replace('_H.', '_K.'), #Read in K band
                                wavefile=wave_filepath+wave_filename.replace('SDCH_', 'SDCK_').replace('_H.', '_K.'))
    else: #Use wavelength solution built into each fits file (default)
        spec_H = IGRINSSpectrumList.read(spec_filepath+spec_filename.replace('SDCK_', 'SDCH_').replace('_K.', '_H.')) #Read in H band
        spec_K = IGRINSSpectrumList.read(spec_filepath+spec_filename.replace('SDCH_', 'SDCK_').replace('_H.', '_K.')) #Read in K band       
    spec_all = concatenate_orders(spec_H, spec_K) #Combine H and K bands
    return spec_all



def readPLP(plppath, date, frameno, waveframeno='', dim='1D'):
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

    if os.path.exists(plppath+'outdata/'+date +'/N'+date+'S'+frameno+'_H'+suffix):
        gemini=True
    elif os.path.exists(plppath+'outdata/'+date +'/'+'SDCH_'+date+'_'+frameno+suffix):
        gemini = False
    else:
        raise Exception(
            "Provided plppath, date, or frameno is incorrect.  Check these."
        )

    if gemini:
        if waveframeno=='':
            spec_H = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/N'+date+'S'+frameno+'_H'+suffix) #Read in H band
            spec_K = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/N'+date+'S'+frameno+'_K'+suffix) #Read in K band  
        else:        
            spec_H = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/N'+date+'S'+frameno+'_H'+suffix, #Read in H bandgemini
                                        wavefile=plppath+'calib/primary/'+date +'/SKY_N'+date+'S'+waveframeno+'_H.wvlsol_v1.fits')
            spec_K = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/N'+date+'S'+frameno+'_K'+suffix, #Read in K band
                                        wavefile=plppath+'calib/primary/'+date +'/SKY_N'+date+'S'+waveframeno+'_K.wvlsol_v1.fits')
    else:
        if waveframeno=='':
            spec_H = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/'+'SDCH_'+date+'_'+frameno+suffix) #Read in H band
            spec_K = IGRINSSpectrumList.read(plppath+'outdata/'+date +'/'+'SDCK_'+date+'_'+frameno+suffix) #Read in K band  
        else:        
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
    filepath: Filepath to fits file storing the data.  Can be .spec.fits, spec2d.fits, or .spec_a0v.fits.

    Returns
    -------
    uncertaintyFilepath: string
        Returns the file path to the uncertianity (.variance.fits or .sn.fits) file.

    """
    path_base = filepath.replace('".spec_a0v.fits','').replace('.spec.fits','').replace('.spec2d.fits','').replace('.spec_flattened.fits','')
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
        # elif os.path.exists(path_base + '.sn.fits'): #If no .variance.fits file found, try using the .sn.fits file
        #     return path_base + '.sn.fits'
        else:
            raise Exception(
                #"Neither .variance.fits or .sn.fits exists in the same path as the spectrum file to get the uncertainty.  Please provide one of these files in the same directory as your spectrum file."
                "The .variance.fits file does not exist in the same path as the spectrum file to get the uncertainty.  Please provide one of these files in the same directory as your spectrum file."
                )             













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


    # def __init__(
    #     self, *args, file=None, order=10, sn_used = False, cached_hdus=None, wavefile=None, **kwargs
    # ):
    def __init__(
        self, *args, file='', wavefile=None, order=10, band='', cached_hdus=None, **kwargs):
     
        if cached_hdus is not None:

            hdr = cached_hdus[0].header
            grating_order = grating_order_offsets[band] + order
            flux = cached_hdus[0].data[order].astype(float) * u.ct
            variance = cached_hdus[1].data[order].astype(np.float64)
            uncertainty = StdDevUncertainty(np.sqrt(variance))
            lamb = cached_hdus[2].data[order].astype(np.float64) * u.micron
            mask = np.isnan(flux) | np.isnan(uncertainty.array)

            meta_dict = {
                "x_values": np.arange(0, 2048, 1, dtype=int),
                "m": grating_order,
                "header": hdr,
            }

            super().__init__(
                spectral_axis=lamb.to(u.Angstrom),
                flux=flux,
                mask=mask,
                wcs=None,
                uncertainty=uncertainty,
                meta=meta_dict,
                **kwargs,
            )
        elif file != '':
            specList = IGRINSSpectrumList.read(file, wavefile=wavefile)
            spec = specList[order]
            super().__init__(
                spectral_axis=spec.spectral_axis,
                flux=spec.flux,
                mask=spec.mask,
                wcs=None,
                uncertainty=spec.uncertainty,
                meta=spec.meta,
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
            Optional. Path to a file storing a wavelength soultion for a night from the plp.
            Wave files are found in the IGRINS PLP callib/primary/DATE/ directory with
            the extension wvlsol_v1.fits.

        """
        # still works
        assert (".spec_a0v.fits" in file) or (".spec.fits" in file) or (".spec_flattened.fits" in file) or (".spec2d.fits" in file)
        hdus = fits.open(file, memmap=False)
        if ("SDCH_" in file) or ("SDCK_" in file): #Normal IGRINS PLP file naming convention and format
            
            if  (".spec.fits" in file) or (".spec_flattened.fits" in file) or (".spec2d.fits" in file):  #For regular .spec.fits and .spec2d.fits files
                flux_hdu = hdus[0]
                wave_hdu = hdus[1]
                if wavefile is not None:
                    wave_hdus = fits.open(wavefile)
                    wave_hdu = wave_hdus[0]
                uncertainty_filepath = getUncertaintyFilepath(file)
                uncertainty_hdus = fits.open(uncertainty_filepath, memmap=False)
                variance_hdu = uncertainty_hdus[0]
            elif (".spec_a0v.fits" in file): #For .spec_a0v.fits files
                flux_hdu = hdus[0]
                variance_hdu = hdus[1]
                wave_hdu = hdus[2]
            if wavefile is not None: #Check if user provided path to wavefile exists, if it does, use that instead
                wave_hdus = fits.open(wavefile)
                wave_hdu = wave_hdus[0]
        elif ("_H." in file) or ("_K." in file): #Gemini IGRINS 
            flux_hdu = hdus[1]
            variance_hdu = hdus[2]
            wave_hdu = hdus[3]
        if wavefile is not None: #Check if user provided path to wavefile exists, if it does, use that instead
            wave_hdus = fits.open(wavefile)
            wave_hdu = wave_hdus[0]
        cached_hdus = [flux_hdu, variance_hdu, wave_hdu] #Set up hdus in the cached_hdus list
        hdus0_shape = cached_hdus[0].data.shape #Normally we read from the 0th extension
        if len(hdus0_shape) == 2: #1D spectrum
            n_orders, n_pix = hdus0_shape
        elif len(hdus0_shape) == 3: #2D spectrum
            n_orders, n_height, n_pix = hdus0_shape
        
        list_out = []

        if ("SDCH" in file) or ("_H." in file):
            band = "H"
        elif ("SDCK" in file) or ("_K." in file):
            band = "K"
        for i in range(n_orders - 1, -1, -1):
            spec = IGRINSSpectrum(
                #file=file, wavefile=wavefile, order=i, sn_used=sn_used, cached_hdus=cached_hdus
                order=i, cached_hdus=cached_hdus, band=band,
            )
            list_out.append(spec)
        specList = IGRINSSpectrumList(list_out)
        specList.file = file
        return specList


    def getSlitThroughput(self, slit_length=14.8, PA=90, guiding_error=1.5, col1=1200, col2=1300, wave_min=1.4, wave_max = 2.6,
        plot=False, plot_order=10, pdfobj=None):
        """Estimate the wavelength dependent fractional slit throughput for a point source nodded ABBA on the IGRINS slit and return the 
        coefficients of a linear fit.

        Parameters
        ----------
        slit_length: float
            Length of the slit on the sky in arcsec.
        PA: float
            Position angle of the slit on the sky in degrees.  Measured counterclockwise from North to East.
        guilding_error: float
            Estimate of the guiding error in arcsec.  This smears out the PSF fits in the East-West direction.
            This should be used carefully and only for telescopes on equitorial mounts.
        col1: int
            Left limit of detector column to collapse to estimate PSF along slit.
        col2: int
            Right limit of detector column to collapse to estimate PSF along slit.
        wave_min: float
            Lower limit on wavelength range for orders for the estimate.
        wave_max: float
            Upper limit on wavelnegth range for orders used for the estimate.
        plot: bool
            Visualize slit throughput calculations.
        plot_order: int
            Make diagnostic plot of this order specific order showing the fit.
        pdfobj: 
            PdfPages object can be provided for saving diagnostic plots.
        Returns
        -------
        m, b:
            Coefficients for a fit of a linear trend of m*(1/wavelength)+b to the fractional slit throughput with the
            wavelength units in microns.

        """
        path_base = self.file.replace('.spec_a0v.fits','').replace('.spec.fits','').replace('.spec2d.fits','').replace('.spec_flattened.fits','')
        path_H = path_base.replace('SDCK', 'SDCH') + '.spec2d.fits'
        path_K = path_base.replace('SDCH', 'SDCK') + '.spec2d.fits'

        if os.path.exists(path_H): #Check if 2D spectrum in a .spec2d.fits file exists
            spec2d_H = fits.getdata(path_H)[::-1,:] #Read in spec2d.fits file if it exists
        else: #If file does not exist, raise exception
            raise Exception(
                "Need .spec2d.fits file in the same directory as "
                + self.file
                + " in order to get an estimate of the slit profile.  .spec2d.fits is missing."
            )  
        if os.path.exists(path_K): #Check if 2D spectrum in a .spec2d.fits file exists
            spec2d_K = fits.getdata(path_K)[::-1,:] #Read in spec2d.fits file if it exists
        else: #If file does not exist, raise exception
            raise Exception(
                "Need .spec2d.fits file in the same directory as "
                + self.file
                + " in order to get an estimate of the slit profile.  .spec2d.fits is missing."
            )  

        spec2d_list = []#Combine both bands into a python list
        for order in range(len(spec2d_H)):    
            spec2d_list.append(spec2d_H[order])
        for order in range(len(spec2d_K)):    #Combine both bands into a python list
            spec2d_list.append(spec2d_K[order])
        igrins_slit = Slit(length=slit_length, width=slit_length*(1/14.8), PA=PA, guiding_error=guiding_error, n_axis=2500) #Initialize Slit object    
        n_orders = len(spec2d_list) #Count number of orders in the combined bands
        f_through_slit = np.zeros(n_orders)   #Store the slit throughput and associated wavelengths in arrays, where each entry is each order
        wave = np.zeros(n_orders)
        for order in range(n_orders):  #Estimate throughput for each order using the median between columns col1 and col2 and save the result and median wavelength in arrays
            normed_spec2d_order = spec2d_list[order] / np.nansum(np.abs(spec2d_list[order]), axis=0)  #normalize continuum
            y = np.nanmedian(normed_spec2d_order[:,col1:col2], axis=1) #Median collapse columns between col1 and col2 to estimate the slit profile in each order
            x = np.arange(len(y)) * (slit_length / len(y)) #x stores the distance along the slit      
            y[np.isnan(y)] = 0. #Zero out nans
            igrins_slit.clear()
            if plot and order==plot_order:
                igrins_slit.ABBA(y, x=x, print_info=True, plot=True, pdfobj=pdfobj)
                print('2D plot')
                igrins_slit.plot2d()
                if pdfobj is not None: #Save figure to file if PdfPages object is provided
                    pdfobj.savefig()
                #breakpoint()
            else:
                igrins_slit.ABBA(y, x=x, print_info=False, plot=False)
            f_through_slit[order] = igrins_slit.estimate_slit_throughput()
            wave[order] = np.nanmedian(self[order].wavelength.um[col1:col2])
        good_orders = np.isfinite(f_through_slit)  #mask out nans
        f_through_slit = f_through_slit[good_orders]
        wave = wave[good_orders]
        init_line = models.Linear1D() #Fit throughput across orders with a linear fit with x = 1/wavelength (1/microns)
        fitter = fitting.LinearLSQFitter()
        outlier_fitter = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0) #Sigma
        i = (wave >= wave_min) & (wave <= wave_max)
        #fitted_line = fitter(init_line, 1/wave[i], f_through_slit[i])
        fitted_line = outlier_fitter(init_line, 1/wave[i], f_through_slit[i])
        m = fitted_line[0].slope.value
        b = fitted_line[0].intercept.value

        if plot:
            plt.figure()
            plt.plot(wave, f_through_slit, '.')
            plt.plot(wave, fitted_line[0](1/wave))
            plt.ylim([-0.2, 1.2])
            plt.xlabel('Wavelength (micron)')
            plt.ylabel('Estimated Slit Throughput')
            if pdfobj is not None: #Save figure to file if PdfPages object is provided
                pdfobj.savefig()
            plt.figure()
            plt.plot(1/wave, f_through_slit, '.')
            plt.plot(1/wave, fitted_line[0](1/wave))
            plt.ylim([-0.2, 1.2])
            plt.xlabel('Inverse Wavelength (1/micron)')
            plt.ylabel('Estimated Slit Throughput')
            if pdfobj is not None: #Save figure to file if PdfPages object is provided
                pdfobj.savefig()
            print('m: ', m)
            print('b: ', b)


        f_throughput = [] #Calculate and return throughput as a function of all wavelengths (columns) based on the fit above
        for i in range(len(self)):
            f_throughput.append(m*(1/self[i].wavelength.um) + b)
        return f_throughput
    def fitTellurics(self, verbose=True, plot=False):
        """ Do a crude telluric fit using a telluric model from the Planetary Spectrum Generator.
        This is meant to be carried out on standard stars to remove tellurics before fitting
        stellar atmosphere models.  The molecule CO2, H2O, CH4, and NO2 abundances are iteratively
        fit to the spectrum over the center of the K-band.  This fit provides enough correction to the
        telluric lines so the star's spectrum can later be easily smoothed.

        Parameters
        ----------
        verbose: bool
            If True, print various diagnostic information about the fitting in the terminal.
        plot: bool
            If True, make diagnostic plots.

        Returns
        -------
        final_trans: numpy array
            Stores the estimated transmission from the atmosphere based on the best fit parameters.
            Dividing a spectrum by final_trans will apply a crude telluric correction.
            Rows the corrispond to orders and columns that corrispond to the detector x position
            

        """


        #Read in model tellurics from the Planetary Spectrum Generator
        psg_tellurics_file = files(templates).joinpath("psg_trn_r100000_1.4_2.5_um.txt")
        d = ascii.read(psg_tellurics_file, header_start=6)
        wave_trans = d["Wave/freq"].data
        delta_lambda_trans = np.nanmedian(wave_trans[1:] - wave_trans[:-1])
        trans_resolution = 100000
        wave1d = []
        flux1d = []
        for order in self:
            wave1d.append(order.spectral_axis.value * 1e-4)
            flux1d.append(order.flux.value)
        wave1d = np.array(wave1d)
        flux1d = np.array(flux1d)
        #Try an automated fit
        #Set initial guesses for pixel shift and resolution
        rolled_wave1d = roll_along_axis(wave1d, 0.0, axis=1) #Apply an overall pixel shift
        R = 45000
        fwhm_to_std = 1/2.355
        streatch=1.0
        #Adjust the following limits and parameters
        order_range = (54-16, 54-7)
        # order_range = (30,49)
        x_range = (350, 1858)
        #x_shifts = np.arange(-2.0,2.0,0.01)
        x_shifts = np.arange(0,0.1,0.1)
        resolutions = np.arange(45000.0, 45200.0, 200.0)
        streatches = np.arange(1.000, 1.001, 0.001)
        g1 = Gaussian1DKernel(stddev=5) #For correction
        rolled_wave1ds = []
        for l, x_shift in enumerate(x_shifts):
            rolled_wave1ds.append(roll_along_axis(wave1d, x_shift, axis=1))
        interpolation_kind = "linear"
        molecules = ['H2O',  'CO2','CH4', 'N2O'] #Seems to be the only three molecules that matter for the H & K bands
        # molecules = [  'CO2','CH4', 'H2O'] #Seems to be the only three molecules that matter for the H & K bands
        alphas = np.arange(-1.0,1.5,0.005) #Range of alphas to test
        best_fit_alphas = np.zeros(len(molecules))
        rolled_wave1d = wave1d
        central_wavelength = np.nanmedian(rolled_wave1d)
        n_iterations = 4
        for iteration in range(n_iterations):
            print('Iteration ', iteration)
            convolution_resolution = (R**(-2) - trans_resolution**(-2))**-0.5
            convolution_std = (central_wavelength / convolution_resolution) * fwhm_to_std / delta_lambda_trans
            g = Gaussian1DKernel(stddev= convolution_std)
            chisq = np.zeros([len(molecules), len(alphas)]) #Store chisq for each fit
            chisq[:] = 1e99
            n_orders = len(wave1d)
            orders = np.arange(n_orders)
            total_trans = np.ones(np.shape(wave1d))
            molec_original_grid_trans = np.ones([len(molecules), len(d[molecules[0]].data)])
            interp_molec_original_grid_trans = []
            total_original_grid_trans = np.ones(len(d[molecules[0]].data))
            corrected_flux = np.zeros(np.shape(wave1d))
            smoothed_corrected_flux = np.zeros(np.shape(wave1d))
            lambda2 = rolled_wave1d[:,-1]
            streached_rolled_wave1d = rolled_wave1d - (rolled_wave1d - lambda2[:,np.newaxis])*(streatch-1)
            for i, molecule in enumerate(molecules):
                trans_other_molecules_best_fit = np.ones(np.shape(wave1d))
                for j in range(len(molecules)):
                    if j != i:
                        interp_obj = interp1d(d["Wave/freq"].data,   convolve(d[molecules[j]].data**best_fit_alphas[j], g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False) #Create interopolation object to convert to IGRINS wavelength/pixel space
                        #trans_other_molecules_best_fit *= interp_obj(rolled_wave1d)
                        trans_other_molecules_best_fit *= interp_obj(streached_rolled_wave1d)
                for j, alpha in enumerate(alphas):
                    if molecule == 'H2O': #Scale down alpha for water
                        alpha = alpha * 0.05
                    interp_obj_molec_trans = interp1d(d["Wave/freq"].data, convolve(d[molecule].data**alpha, g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False) #Create interopolation object to convert to IGRINS wavelength/pixel space
                    for order in range(order_range[0], order_range[1]):
                        #corrected_flux[order] = convolve(std_flux_divided_by_synthetic_model[order], g1, normalize_kernel=False) / convolve(interp_obj_molec_trans(rolled_wave1d[order])*total_trans[order], g1, normalize_kernel=False)
                        corrected_flux[order] = convolve(flux1d[order], g1, normalize_kernel=False) / convolve(interp_obj_molec_trans(streached_rolled_wave1d[order])*trans_other_molecules_best_fit[order], g1, normalize_kernel=False)
                        smoothed_corrected_flux[order] = median_filter(corrected_flux[order], size=100)
                    chisq[i, j] = np.nansum(  np.abs((corrected_flux[order_range[0]:order_range[1], x_range[0]:x_range[1]] - smoothed_corrected_flux[order_range[0]:order_range[1], x_range[0]:x_range[1]]))**2  )
                chisq[chisq == 0.] = np.nan #Mask out garbage
                best_fit_alpha = alphas[chisq[i] == np.nanmin(chisq[i])][0]
                if molecule == 'H2O': #Scale down alpha for water
                    best_fit_alpha = best_fit_alpha * 0.05                 
                best_fit_alphas[i] = best_fit_alpha
                interp_obj_molec_trans = interp1d(d["Wave/freq"].data,   convolve(d[molecule].data**best_fit_alpha, g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False) #Create interopolation object to convert to IGRINS wavelength/pixel space
                total_trans *= interp_obj_molec_trans(streached_rolled_wave1d)
                #total_trans *= interp_obj_molec_trans(rolled_wave1d)
                molec_original_grid_trans[i] = d[molecule].data**best_fit_alpha
                interp_molec_original_grid_trans.append(interp1d(d["Wave/freq"].data,   convolve(molec_original_grid_trans[i], g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False))
                total_original_grid_trans *= molec_original_grid_trans[i]
                print('Best fit alpha for '+molecule+' = ', best_fit_alpha)
            if iteration < n_iterations - 1:
                order1, order2 = order_range[0], order_range[1]
                x1, x2 = x_range[0], x_range[1]
                n_x_shifts, n_resolutions, n_streatches = len(x_shifts), len(resolutions), len(streatches)
                chisq = np.zeros([n_resolutions, n_x_shifts, n_streatches])
        
                chunk_flattened_std = (flux1d[order,x1:x2] / smoothed_corrected_flux[order1:order2,x1:x2])
                for i in range(n_resolutions):
                    convolution_resolution = (resolutions[i]**(-2) - trans_resolution**(-2))**-0.5
                    convolution_std = (central_wavelength / convolution_resolution) * fwhm_to_std / delta_lambda_trans
                    g = Gaussian1DKernel(stddev= convolution_std)
                    interp_obj = interp1d(d["Wave/freq"].data,   convolve(total_original_grid_trans, g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False) #Create interopolation object to convert to IGRINS wavelength/pixel space
                
                    for j in range(n_x_shifts):
                        rolled_wave1d = rolled_wave1ds[j][order1:order2,x1:x2]
                        #lambda1 = rolled_wave1d[:,0]
                        lambda2 = rolled_wave1d[:,-1]
                        for k in range(n_streatches):
                            streatched_rolled_wave1d = rolled_wave1d - (rolled_wave1d - lambda2[:,np.newaxis])*(streatches[k]-1)
                            chunk_interpolated_total_trans = interp_obj(streatched_rolled_wave1d)
                            chisq[i,j,k] = np.nansum(np.abs(chunk_flattened_std - chunk_interpolated_total_trans)**2)
                ii, jj, kk = np.where(chisq == np.nanmin(chisq))
                best_fit_resolution = resolutions[ii][0]
                best_fit_x_shift = x_shifts[jj][0]
                best_fit_streatch = streatches[kk][0]
                if verbose:
                    print('Best fit R = ', best_fit_resolution)
                    print('Best fit pixel shift = ', best_fit_x_shift)
                    print('Best fit streatch = ', best_fit_streatch)
                R = best_fit_resolution
                rolled_wave1d = roll_along_axis(wave1d, best_fit_x_shift, axis=1) #Apply an overall pixel shift
                streatch = best_fit_streatch
        convolution_resolution = (R**(-2) - trans_resolution**(-2))**-0.5
        convolution_std = (central_wavelength / convolution_resolution) * fwhm_to_std / delta_lambda_trans
        g = Gaussian1DKernel(stddev= convolution_std)
        interp_obj = interp1d(d["Wave/freq"].data,   convolve(total_original_grid_trans, g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False) #Create interopolation object to convert to IGRINS wavelength/pixel space
        interp_obj_total_original_grid_trans = interp1d(d["Wave/freq"].data,   convolve(total_original_grid_trans, g, normalize_kernel=False), kind=interpolation_kind, bounds_error=False) #Create interopolation object to convert to IGRINS wavelength/pixel space
        

        best_fit_rolled_wave1d = roll_along_axis(wave1d, best_fit_x_shift, axis=1) #Apply an overall pixel shift
        lambda2 = lambda2 = best_fit_rolled_wave1d[:,-1]
        best_fit_streatched_rolled_wave1d = best_fit_rolled_wave1d - (best_fit_rolled_wave1d - lambda2[:,np.newaxis])*(best_fit_streatch-1)
        final_trans = interp_obj_total_original_grid_trans(best_fit_streatched_rolled_wave1d)

        #final_trans = interp_obj_total_original_grid_trans(wave1d)
        if plot:
            corrected_flux = np.zeros(np.shape(wave1d))
            smoothed_corrected_flux = np.zeros(np.shape(wave1d))
            plt.figure(figsize=[10,5])
            #for order in range(order_range[0], order_range[1]): #range(len(wave1d)):
            for order in range(len(wave1d)):
                #corrected_flux[order] = flux1d[order] / total_trans[order]
                corrected_flux[order] = flux1d[order] / final_trans[order]
                smoothed_corrected_flux[order] = convolve(corrected_flux[order],g, normalize_kernel=False)
                plt.plot(wave1d[order][100:1950], flux1d[order][100:1950], label='A0V spec file', color='silver')
                plt.plot(wave1d[order][100:1950], corrected_flux[order][100:1950], color='black')
                #plt.plot(wave1d[order][100:1950], smoothed_corrected_flux[order][100:1950], label='Estimated Blaze', color='black')
            max_y = np.nanmax(flux1d)
            plt.ylim([-0.2*max_y, 1.2*max_y])
            plt.show()
            #flattened_std_flux_divided_by_synthetic_model = flux1d / smoothed_corrected_flux   
        return final_trans
    def fitStandardStar(self, name, plot=False, verbose=True, max_iterations=10, logg_range=(3.0,5.0), z_range=(-1.0,0.0), 
            alpha_range=(0.8,1.5), rotational_broadening_range=(10, 150), radial_velocity_range=(-100, 100)):
        g_large = Gaussian1DKernel(stddev=40.0)
        br14_x1 = 15838
        br14_x2 = 15950
        br10_x1 = 17300
        br10_x2 = 17435
        brgamma_x1 = 21605
        brgamma_x2 = 21722
        #Grab orders with the brackett lines
        brgamma_order = 0
        br10_order = 0
        br14_order = 0
        for order in range(len(self)):
            if (self[order].spectral_axis[0].value < brgamma_x1) and (self[order].spectral_axis[-1].value > brgamma_x2):
                brgamma_order = order
            if (self[order].spectral_axis[0].value < br10_x1) and (self[order].spectral_axis[-1].value > br10_x2):
                br10_order = order
            if (self[order].spectral_axis[0].value < br14_x1) and (self[order].spectral_axis[-1].value > br14_x2):
                br14_order = order
        # if plot:
        #     self[br14_order].plot()
        #     self[br10_order].plot()
        #     self[brgamma_order].plot()


        #RUN TELLURIC CORRECTOIN

        total_trans = self.fitTellurics(verbose=verbose, plot=plot)
        #Fit standard star spectrum
        #Get initial guess
        best_fit_z = -0.5
        best_fit_logg = 4.5
        best_fit_rotational_broadening = 40.0
        best_fit_radial_velocity = 0.
        best_fit_alpha = 1.0
        #Read in color grids
        color_grid_file = files(templates).joinpath("color_grid_full.csv")
        teff, logg, z, B_minus_V, J_minus_V, H_minus_V, K_minus_V = np.loadtxt(color_grid_file, delimiter=',')
        #subgrid_teff, subgrid_logg, subgrid_z, subgrid_B_minus_V, subgrid_J_minus_V, subgrid_H_minus_V, subgrid_K_minus_V =  np.loadtxt('color_grid_sub.csv', delimiter=',')
        # Grab colors of a target from simbad
        query_result = Simbad.query_object(name)
        target_B = query_result['B'][0] 
        target_V = query_result['V'][0]
        target_J = query_result['J'][0]
        target_H = query_result['H'][0]
        target_K = query_result['K'][0]

        #Initial attempt to use colors to constrain stellar parameters
        n = len(teff)
        chisq = np.zeros(n)
        for i in range(n):
            chisq[i] = (target_B - (B_minus_V[i]+target_V))**2 + \
                        (target_J - (J_minus_V[i]+target_V))**2 + \
                        (target_H - (H_minus_V[i]+target_V))**2 +  \
                        (target_K - (K_minus_V[i]+target_V))**2
        # min_chisq = chisq == np.nanmin(chisq[logg==4.5])
        min_chisq = chisq == np.nanmin(chisq[(logg==best_fit_logg) & (z==best_fit_z)])
        best_fit_teff = teff[min_chisq][0]
        # color_best_fit_teff =  teff[min_chisq][0]
        # color_best_fit_logg = logg[min_chisq][0]
        print('min Teff =', teff[min_chisq][0])
        # print('min log(g) =', logg[min_chisq][0])        
        br14_spec = isolate_and_normalize_hi_order(i=br14_order, x1=br14_x1, x2=br14_x2, specobj=copy.deepcopy(self)/total_trans, mask=True) 
        br14_mask =  binary_dilation((br14_spec.flux.value / convolve(br14_spec, g_large)) <  0.85, iterations=5) #Try to mask tellurics
        br10_spec = isolate_and_normalize_hi_order(i=br10_order, x1=br10_x1, x2=br10_x2, specobj=copy.deepcopy(self)/total_trans, mask=True)
        br10_mask =  binary_dilation((br10_spec.flux.value / convolve(br10_spec, g_large)) <  0.85, iterations=5) #Try to mask tellurics
        brgamma_spec = isolate_and_normalize_hi_order(i=brgamma_order, x1=brgamma_x1, x2=brgamma_x2, specobj=copy.deepcopy(self)/total_trans, mask=True) 
        brgamma_mask =  binary_dilation((brgamma_spec.flux.value / convolve(brgamma_spec, g_large)) <  0.85, iterations=5) #Try to mask tellurics
        br14_window = (br14_spec.spectral_axis.value > br14_x1) & (br14_spec.spectral_axis.value <= br14_x2)
        br10_window = (br10_spec.spectral_axis.value > br10_x1) & (br10_spec.spectral_axis.value <= br10_x2)
        brgamma_window = ((brgamma_spec.spectral_axis.value > brgamma_x1) & (brgamma_spec.spectral_axis.value <= brgamma_x2))
        g = Gaussian1DKernel(stddev=7.0) #Do a little bit of smoothing of the blaze functions
        br14_spec_smoothed_flux =  convolve(br14_spec.flux.value, g, mask=br14_mask)
        br10_spec_smoothed_flux =  convolve(br10_spec.flux.value, g, mask=br10_mask)
        brgamma_spec_smoothed_flux =  convolve(brgamma_spec.flux.value, g, mask=brgamma_mask)
        brgamma_spec = brgamma_spec / (brgamma_spec/brgamma_spec_smoothed_flux)
        br10_spec = br10_spec / (br10_spec/br10_spec_smoothed_flux)
        br14_spec = br14_spec / (br14_spec/br14_spec_smoothed_flux)
        #for iteration in range(n_iterations):
        iteration = 0
        last_best_fit_teff = 0
        last_best_fit_logg = 0
        last_best_fit_z = 0
        last_best_fit_rotational_broadening = 0
        last_best_fit_radial_velocity = 0
        last_best_fit_alpha = 0

        #Full grid from gollum
        nearest_best_fit_teff = round_to_multiple(best_fit_teff, 200)
        grid = PHOENIXGrid(teff_range=(nearest_best_fit_teff, nearest_best_fit_teff), logg_range=logg_range, 
                           Z_range=z_range, wl_lo=3450, wl_hi= 25500, download=True)
        #Now create a subgrid by averaging between points on the , AKA new_grid
        new_grid = []
        new_grid_logg = []
        new_grid_z = []
        logg_list = np.arange(logg_range[0], logg_range[1]+0.25, 0.25)
        z_list = np.arange(z_range[0], z_range[1]+0.25, 0.25)
        for logg in logg_list:
            for z in z_list:
                    if logg % 0.5 == 0:
                        if z % 0.5 == 0:
                            entryAvg = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg, metallicity=z))])
                        else:
                            entry1 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg, metallicity=z))])
                            entry2 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg, metallicity=z+0.5))])
                            entryAvg = (entry1 + entry2) / 2
                    else:
                        if z % 0.5 == 0:
                            entry1 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg, metallicity=z))])
                            entry2 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg+0.5, metallicity=z))])
                            entryAvg = (entry1 + entry2) / 2                 
                        else:
                            entry1 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg, metallicity=z))])
                            entry2 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg+0.5, metallicity=z))])
                            entry3 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg, metallicity=z+0.5))])
                            entry4 = copy.deepcopy(grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=logg+0.5, metallicity=z+0.5))])
                            entryAvg = (entry1 + entry2 + entry3 + entry4) / 4
                    entryAvg = grid[0].__class__(entryAvg) #Force class to be gollum model
                    entryAvg.meta['logg'] = logg
                    entryAvg.meta['Z'] = z
                    entryAvg.meta['teff'] = best_fit_teff      
                    
                    new_grid.append(entryAvg)
                    new_grid_logg.append(logg)
                    new_grid_z.append(z) 
        new_grid_logg = np.array(new_grid_logg)
        new_grid_z = np.array(new_grid_z)

        #Iterate until convergence
        while ((best_fit_teff != last_best_fit_teff) or (best_fit_logg != last_best_fit_logg) or (best_fit_z != last_best_fit_z) or \
                    (best_fit_rotational_broadening != last_best_fit_rotational_broadening) or (best_fit_radial_velocity != last_best_fit_radial_velocity) or \
                    (best_fit_alpha != last_best_fit_alpha)) and (iteration < max_iterations):
            last_best_fit_teff = best_fit_teff
            last_best_fit_logg = best_fit_logg
            last_best_fit_z = best_fit_z
            last_best_fit_rotational_broadening = best_fit_rotational_broadening
            last_best_fit_radial_velocity = best_fit_radial_velocity
            last_best_fit_alpha = best_fit_alpha
            if verbose:
                print('ITERATION ', iteration)
                print('teff = ', best_fit_teff)
                print('logg = ', best_fit_logg)
                print('z = ', best_fit_z)
                print('rotational broadening = ', best_fit_rotational_broadening)
                print('radial velocity = ', best_fit_radial_velocity)
                print('alpha = ', best_fit_alpha)
            #FIND RV AND ROTATIONAL VELOCITY
            nearest_best_fit_teff = round_to_multiple(best_fit_teff, 200)
            grid_index = np.where((new_grid_logg == best_fit_logg) & (new_grid_z == best_fit_z))[0][0]
            model_spec = new_grid[grid_index]
            # grid = PHOENIXGrid(teff_range=(nearest_best_fit_teff, nearest_best_fit_teff), logg_range=(best_fit_logg, best_fit_logg), 
            #                    Z_range=(best_fit_z, best_fit_z), wl_lo= 14000, wl_hi= 25000, download=True)
            rotational_broadenings = np.arange(rotational_broadening_range[0], rotational_broadening_range[1]+5, 5)
            radial_velocities = np.arange(radial_velocity_range[0], radial_velocity_range[1]+5, 5)
            chisq = []
            result_rotational_broadening = []
            result_velocities = []
            result_alphas = []
            #for model_spec in grid:
            for rotational_broadening in rotational_broadenings:
                broadened_model_spec = model_spec.rotationally_broaden(rotational_broadening).instrumental_broaden(45000) 
                #shifted_model_spec = model_spec.instrumental_broaden(45000).rotationally_broaden(rotational_broadening)
                for radial_velocity in radial_velocities:
                    shifted_broadened_model_spec = copy.deepcopy(broadened_model_spec)
                    shifted_broadened_model_spec.shift_spectrum_to(radial_velocity = radial_velocity*(u.km/u.s))
                    # #Br-14
                    br14_synth = shifted_broadened_model_spec.resample(br14_spec)
                    br14_synth = edge_normalize(x1=br14_x1, x2=br14_x2, specobj=br14_synth)
                    brgamma_synth = shifted_broadened_model_spec.resample(brgamma_spec)
                    brgamma_synth = edge_normalize(x1=brgamma_x1, x2=brgamma_x2, specobj=brgamma_synth)
                    #Br-gamma
                    #for alpha in [1.3]:
                    for alpha in np.arange(alpha_range[0], alpha_range[1]+0.025, 0.025):
                        diff_br14 = (br14_spec.flux.value-br14_synth.flux.value**alpha)
                        diff_brgamma = (brgamma_spec.flux.value-brgamma_synth.flux.value**alpha)

                        if iteration ==0:
                            diff_br14 = diff_br14 / np.nanmax(diff_br14)
                            diff_brgamma = diff_brgamma / np.nanmax(diff_brgamma)

                        chisq.append(np.nansum((diff_br14[br14_window]**2)) + 
                                        np.nansum((diff_brgamma[brgamma_window]**2)))
                        result_rotational_broadening.append(rotational_broadening)
                        result_velocities.append(radial_velocity)
                        result_alphas.append(alpha)
            chisq = np.array(chisq)
            result_rotational_broadening = np.array(result_rotational_broadening)
            result_velocities = np.array(result_velocities)
            min_i = np.where(chisq == np.nanmin(chisq))[0][0]
            best_fit_rotational_broadening = result_rotational_broadening[min_i]
            best_fit_radial_velocity = result_velocities[min_i]
            best_fit_alpha = result_alphas[min_i]
            #Now with velocities fixed, vary Z and logg
            # nearest_best_fit_teff = round_to_multiple(best_fit_teff, 200)
            # grid = PHOENIXGrid(teff_range=(nearest_best_fit_teff, nearest_best_fit_teff), logg_range=(3.0, 5.0), 
            #                    Z_range=(-1.0, 0.0), wl_lo= 14000, wl_hi= 25000, download=True)
            chisq = []
            result_logg = []
            result_z = []
            count=0
            #for model_spec in grid:
            for model_spec in new_grid:
                count = count+1
                broadened_model_spec = model_spec.rotationally_broaden(best_fit_rotational_broadening).instrumental_broaden(45000)
                shifted_broadened_model_spec = copy.deepcopy(broadened_model_spec)
                shifted_broadened_model_spec.shift_spectrum_to(radial_velocity = best_fit_radial_velocity*(u.km/u.s))
                # #Br-14
                br14_synth = shifted_broadened_model_spec.resample(br14_spec)
                br14_synth = edge_normalize(x1=br14_x1, x2=br14_x2, specobj=br14_synth)
                diff_br14 = (br14_spec.flux.value-br14_synth.flux.value**best_fit_alpha)
                brgamma_synth = shifted_broadened_model_spec.resample(brgamma_spec)
                brgamma_synth = edge_normalize(x1=brgamma_x1, x2=brgamma_x2, specobj=brgamma_synth)
                diff_brgamma = (brgamma_spec.flux.value-brgamma_synth.flux.value**best_fit_alpha)

                if iteration == 0:
                    diff_br14 = diff_br14/np.nanmax(diff_br14)
                    diff_brgamma = diff_brgamma/np.nanmax(diff_brgamma)

                chisq.append(np.nansum((diff_br14[br14_window]**2)) + 
                                np.nansum((diff_brgamma[brgamma_window]**2)))
                result_logg.append(model_spec.logg)
                result_z.append(model_spec.Z)
            chisq = np.array(chisq)
            result_logg = np.array(result_logg)
            min_i = np.where(chisq == np.nanmin(chisq))[0][0]
            best_fit_logg = result_logg[min_i]
            best_fit_z = result_z[min_i]

            # #Use colors to constrain stellar parameters
            # n = len(teff)
            # chisq = np.zeros(n)
            # for i in range(n):
            #     chisq[i] = (target_B - (B_minus_V[i]+target_V))**2 + \
            #                 (target_J - (J_minus_V[i]+target_V))**2 + \
            #                 (target_H - (H_minus_V[i]+target_V))**2 +  \
            #                 (target_K - (K_minus_V[i]+target_V))**2
            # # min_chisq = chisq == np.nanmin(chisq[logg==4.5])
            # min_chisq = chisq == np.nanmin(chisq[(logg==best_fit_logg) & (z==best_fit_z)])

            # best_fit_teff = round_to_multiple(teff[min_chisq][0], 200)
            iteration += 1
        if verbose:
            if iteration == max_iterations:
                print('Maximum number of iterations reached.')
            else:
                print('Fit has converged.')
            print('FINAL RESULTS FOR '+name)
            print('teff = ', best_fit_teff)
            print('logg = ', best_fit_logg)
            print('z = ', best_fit_z)
            print('rotational broadening = ', best_fit_rotational_broadening)
            print('radial velocity = ', best_fit_radial_velocity)
            print('alpha = ', best_fit_alpha)
        #Grab best fit model from grid
        #model_spec = grid[grid.get_index(grid.find_nearest_grid_point(teff=best_fit_teff, logg=best_fit_logg, metallicity=best_fit_z))] \
        #        .rotationally_broaden(best_fit_rotational_broadening)
        grid_index = np.where((new_grid_logg == best_fit_logg) & (new_grid_z == best_fit_z))[0][0]
        model_spec = new_grid[grid_index].rotationally_broaden(best_fit_rotational_broadening)
        model_spec.shift_spectrum_to(radial_velocity=best_fit_radial_velocity*(u.km/u.s))

        x = np.array([1.52, 1.6, 1.62487, 1.66142, 1.7, 1.9, 2.0, 2.1, 2.2, 2.25])*1e4 #Coordinates tracing continuum of Vega, taken between H I lines in the model spectrum vegallpr25.50000resam5
        #y = array([2493670., 1950210., 1584670., 1512410., 1406170. , 1293900., 854857., 706839., 589023., 494054., 417965., 356822., 306391.]) * scale_vega_flux * 1e3
        interp1d_model = interp1d(model_spec.spectral_axis.value, model_spec.flux.value, kind='linear', bounds_error=False)
        continuum_points = interp1d_model(x)
        interp1d_cont = interp1d(x, continuum_points, kind='cubic', bounds_error=False)
        cont = interp1d_cont(model_spec.spectral_axis.value) #grab interpolated continuum once so dn't have to interpolate it again
        blue_side = model_spec.spectral_axis.value < x[0] #Set blue most side of continuum to model spec to avoid weirdness
        red_side = model_spec.spectral_axis.value > x[-1]
        cont[blue_side] = model_spec.flux.value[blue_side]
        cont[red_side] = model_spec.flux.value[red_side]

        if plot:
            #Br-14
            br14_synth = edge_normalize(x1=br14_x1, x2=br14_x2, specobj=model_spec.resample(br14_spec))
            plt.figure()
            plt.plot(br14_spec.spectral_axis, br14_spec.flux, label='Br-14')
            plt.plot(br14_synth.spectral_axis, br14_synth.flux**best_fit_alpha, label='Synthetic Spectrum')
            plt.plot([br14_x1, br14_x2], [1,1], label='Estimated Continuum Level')
            plt.xlim([br14_x1, br14_x2])
            plt.ylim([0.5, 1.25])
            plt.legend()
            plt.xlabel('Wavelength (Ang.)')
            plt.ylabel('Normalized Flux')
            plt.title('Br-14')
            #Br-gamma
            #Br-10
            br10_synth = edge_normalize(x1=br10_x1, x2=br10_x2, specobj=model_spec.resample(br10_spec))
            plt.figure()
            plt.plot(br10_spec.spectral_axis, br10_spec.flux, label='Br-10')
            plt.plot(br10_synth.spectral_axis, br10_synth.flux**best_fit_alpha, label='Synthetic Spectrum')
            plt.plot([br10_x1, br10_x2], [1,1], label='Estimated Continuum Level')
            plt.xlim([br10_x1, br10_x2])
            plt.ylim([0.5, 1.25])
            plt.legend()
            plt.xlabel('Wavelength (Ang.)')
            plt.ylabel('Normalized Flux')
            plt.title('Br-10')
            #Br-gamma
            brgamma_synth = edge_normalize(x1=brgamma_x1, x2=brgamma_x2, specobj=model_spec.resample(brgamma_spec))
            plt.figure()
            plt.plot(brgamma_spec.spectral_axis, brgamma_spec.flux, label='Br-Gamma')
            plt.plot(brgamma_synth.spectral_axis, brgamma_synth.flux**best_fit_alpha, label='Synthetic Spectrum')
            plt.plot([brgamma_x1, brgamma_x2], [1,1], label='Estimated Continuum Level')
            plt.xlim([brgamma_x1, brgamma_x2])
            plt.ylim([0.5, 1.25])
            plt.legend()
            plt.xlabel('Wavelength (Ang.)')
            plt.ylabel('Normalized Flux')
            plt.title('Br-gamma')
            #Plot the non-blaze corrected or normalized spectrum
            plt.figure()
            resampled_model_spec = (model_spec.resample(self[br14_order])).normalize()
            plt.plot(self[br14_order].spectral_axis, self[br14_order].flux, label = 'Uncorrected spectrum')
            plt.plot(self[br14_order].spectral_axis, self[br14_order].flux.value/(resampled_model_spec.flux.value**best_fit_alpha), label = 'Corrected spectrum')
            plt.xlim([br14_x1-50, br14_x2+50])
            resampled_model_spec = model_spec.resample(self[br10_order]).normalize()
            plt.figure()
            plt.plot(self[br10_order].spectral_axis, self[br10_order].flux, label = 'Uncorrected spectrum')
            plt.plot(self[br10_order].spectral_axis, self[br10_order].flux.value/(resampled_model_spec.flux.value**best_fit_alpha), label = 'Corrected spectrum')
            plt.xlim([br10_x1-50, br10_x2+50])
            resampled_model_spec = model_spec.resample(self[brgamma_order]).normalize()
            plt.figure()
            plt.plot(self[brgamma_order].spectral_axis, self[brgamma_order].flux, label = 'Uncorrected spectrum')
            plt.plot(self[brgamma_order].spectral_axis, self[brgamma_order].flux.value/(resampled_model_spec.flux.value**best_fit_alpha), label = 'Corrected spectrum')
            plt.xlim([brgamma_x1-50, brgamma_x2+50])
            #Plot model fit
            # x = np.array([1.52, 1.6, 1.62487, 1.66142, 1.7, 1.9, 2.0, 2.1, 2.2, 2.25])*1e4 #Coordinates tracing continuum of Vega, taken between H I lines in the model spectrum vegallpr25.50000resam5
            # #y = array([2493670., 1950210., 1584670., 1512410., 1406170. , 1293900., 854857., 706839., 589023., 494054., 417965., 356822., 306391.]) * scale_vega_flux * 1e3
            # interp1d_model = interp1d(model_spec.spectral_axis.value, model_spec.flux.value, kind='linear', bounds_error=False)
            # continuum_points = interp1d_model(x)
            # interp1d_cont = interp1d(x, continuum_points, kind='cubic', bounds_error=False)
            # cont = interp1d_cont(model_spec.spectral_axis.value) #grab interpolated continuum once so dn't have to interpolate it again
            # blue_side = model_spec.spectral_axis.value < x[0] #Set blue most side of continuum to model spec to avoid weirdness
            # red_side = model_spec.spectral_axis.value > x[-1]
            # cont[blue_side] = model_spec.flux.value[blue_side]
            # cont[red_side] = model_spec.flux.value[red_side]
            # #Fit a polynomial to the continuum
            plt.figure()
            x = model_spec.spectral_axis.value
            # cont = mask_hydrogen_lines(model_spec)
            # mask = np.isfinite(cont)
            # x1, x2 = 20, -20 #clip edges
            # cont_pfit = np.poly1d(np.polyfit(x[mask][x1:x2], cont[mask][x1:x2], deg=5))
            plt.plot(x, model_spec.flux.value, color='black')
            plt.plot(x, cont, color='blue')
            #plt.plot(x, cont_pfit(x), color='red')
            #Scale spectrum by power law (best fit alpha)
            scaled_model_flux = ((model_spec.flux.value / cont)**(best_fit_alpha))*cont
            plt.plot(x, scaled_model_flux, color='red')
            #plt.xlim([22000,24000])
        scaled_model_spec_flux = ((model_spec.flux.value/ cont)**(best_fit_alpha))*cont
        model_spec = model_spec.__class__(model_spec * (scaled_model_spec_flux / model_spec.flux.value))
        return model_spec
        #return resample_list(model_spec, self)


