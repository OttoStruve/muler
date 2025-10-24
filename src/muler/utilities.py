import logging

import numpy as np
import copy
from specutils.spectra import Spectrum1D
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy.modeling import models, fitting #import the astropy model fitting package
from astropy import units as u
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from specutils.manipulation import LinearInterpolatedResampler
from matplotlib import pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.ndimage import binary_dilation
from astroquery.simbad import Simbad
#Simbad.add_votable_fields('allfluxes')
LinInterpResampler = LinearInterpolatedResampler()
from tynt import FilterGenerator
from astropy.coordinates import SkyCoord

log = logging.getLogger(__name__)



def roll_along_axis(array_to_correct, correction, axis=0): #Apply flexure correction by numpy rolling along an axis and averaging between two rolled arrays to account for sub-pixel shifts
    axis = int(axis)
    integer_correction = np.round(correction) #grab whole number component of correction
    fractional_correction = correction - float(integer_correction) #Grab fractional component of correction (remainder after grabbing whole number out)
    rolled_array =  np.roll(array_to_correct, int(integer_correction), axis=axis) #role array the number of pixels matching the integer correction
    if fractional_correction > 0.: #For a positive correction
        rolled_array_plus_one = np.roll(array_to_correct, int(integer_correction+1), axis=axis) #Roll array an extra one pixel to the right
    else: #For a negative correction
        rolled_array_plus_one = np.roll(array_to_correct, int(integer_correction-1), axis=axis) #Roll array an extra one pixel to the left
    corrected_array = rolled_array*(1.0-np.abs(fractional_correction)) + rolled_array_plus_one*np.abs(fractional_correction) #interpolate over the fraction of a pixel
    return corrected_array

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

def find_nearest(array, value): #
    """Return index of entry in sorted array closest to provided value.
    Modified slightly from: https://stackoverflow.com/a/2566508

    Parameters
    -------
    array:
        Sorted array to search.
    value:
        Value to search for in array.

    Returns
        Index for entry in array closest to value
    -------
    """
    return (np.abs(np.asarray(array) - value)).argmin()



def edge_normalize(x1, x2, specobj, window=20): 
    """
    Draw a line between the fluxes at points x1 and x2 and normalize to that line.
    Used for crude continuum normalization to isolate H I line profiles in a standard star spectrum.
    """
    half_window = round(window / 2)
    x = specobj.spectral_axis.value
    ix1 = find_nearest(x, x1) #Grab points to normalize to
    ix2 = find_nearest(x, x2)
    y1 = np.nanmedian(specobj.flux.value[ix1-half_window:ix1+half_window]) #Normalize to end points using a linear fit that goes through the edges
    y2 = np.nanmedian(specobj.flux.value[ix2-half_window:ix2+half_window])
    m = (y2 - y1) / (x[ix2] - x[ix1]) #Fit for a line through two points
    b = y2 - m * x[ix2]
    #specresult = specobj / (m*x+b)
    #specresult = specobj / ((y1+y2)/2)
    return specobj / (m*x+b)
    

def isolate_and_normalize_hi_order(i, x1, x2, specobj, mask=True):
    """
    Function to aid in isolating and continuum normalizing H I line profiles in a standard star spectrum.
    """
    g_large = Gaussian1DKernel(stddev=40.0)
    g = Gaussian1DKernel(stddev=20.0) #Do a little bit of smoothing of the blaze functions
    if mask:
        left_mask = binary_dilation((specobj[i-1].flux.value / convolve(specobj[i-1].flux.value, g_large)) < 0.80, iterations=5) #Order to the left
        left_order = convolve(convolve(specobj[i-1].flux.value, g_large, mask=left_mask), g, mask=left_mask)
        right_mask = binary_dilation((specobj[i+1].flux.value / convolve(specobj[i+1].flux.value, g_large)) < 0.80, iterations=5) #order to the right
        right_order = convolve(convolve(specobj[i+1].flux.value, g_large, mask=right_mask),  g, mask=right_mask)
    else:
        left_order = convolve(specobj[i-1].flux.value, g)
        right_order = convolve(specobj[i+1].flux.value, g)
    #cont =  convolve(np.nanmean([left_order, right_order], axis=0), g_large) #Average both orders to get some idea of what the continuum should be
    # ix1 = find_nearest(specobj[i].spectral_axis.value, x1) #Grab points to normalize to
    # ix2 = find_nearest(specobj[i].spectral_axis.value, x2)
    # y1 = specresult.flux[ix1] #Normalize to end points using a linear fit that goes through the edges
    # y2 = specresult.flux[ix2]
    # m = (y2 - y1) / (ix2 - ix1)
    # b = y2 - m * ix2
    # specresult = specresult / (m*x+b)
    #return edge_normalize(x1=x1, x2=x2, specobj=specobj[i]/cont)
    return edge_normalize(x1=x1, x2=x2, specobj=specobj[i]/convolve(np.nanmean([left_order, right_order], axis=0), g_large))


def resample_combine_spectra(input_spec, spec_to_match, weights=1.0):
        """Linearly resample input_spectra, which can be a list of spectra, to match specrum_to_match and return an EchelleSpectrum
        or EchelleSpectrumList object with the same spectral axis and naned pixels as specrum_to_match.  One main applications
        for this is to match multiple synthetic spectra generated from stellar atmosphere models to a real spectrum.

        Parameters
        -------
        input_spec :
            A EchelleSpectrumm EchelleSpectrumList, or similar specutils object (or list of objects) to be resampled to match spec_to_match.
        specrum_to_match :
            A EchelleSpectrum or EchelleSpectrumLis spectrum which the input_spec will be resampled to match in both wavelength and naned pixels
        weights :
            A list or array giving the fraction of each spectrum in input_spec that makes up the final resampled spectrum.
            Useful for grid interpolation for stellar atmosphere models or just stacking spectra from multiple objects
            into one spectrum.
    
        Returns
        -------
        An EchelleSpectrum or EchelleSpectrumList object with the same wavelength arrays and naned pixels as spec_to_match.
        """

        if is_list(input_spec): #
            weights = np.array(weights) #Check that weights are a list and their sum equals 1
            sum_weights = np.sum(weights)
            assert (len(weights)==1 and weights[0] == 1) or (len(weights) > 1), "If providing weights, You need to provide a weight for each input spectrum.."
            assert sum_weights == 1, "Total weights in weights list is "+str(sum_weights)+" but total must equal to 1."
           
            if is_list(spec_to_match):
                resampled_spec = resample_list(input_spec[0], spec_to_match)*(weights[0]) #Resample spectra
                for i in range(1, len(input_spec)):
                    if len(weights)==1 and weights[0] == 1:
                        resampled_spec = resampled_spec + resample_list(input_spec[i], spec_to_match)*(weights[i])
                    else:
                        resampled_spec = resampled_spec + resample_list(input_spec[i], spec_to_match)
            else:
                resampled_spec = LinInterpResampler(input_spec[0], spec_to_match.spectral_axis)*(weights[0]) #Resample spectra
                for i in range(1, len(input_spec)):
                    if len(weights)==1 and weights[0] == 1:
                        resampled_spec = resampled_spec + LinInterpResampler(input_spec[i], spec_to_match.spectral_axis)*(weights[i])
                    else:
                        resampled_spec = resampled_spec + LinInterpResampler(input_spec[i], spec_to_match.spectral_axis)
        else:
            if is_list(spec_to_match):
                resampled_spec = resample_list(input_spec, spec_to_match) #Resample spectrum
            else:
                resampled_spec = LinInterpResampler(input_spec, spec_to_match.spectral_axis)
                resampled_spec = spec_to_match.__class__( #Ensure resampled_spec is the same object as spec_to_match
                    spectral_axis=resampled_spec.spectral_axis, flux=resampled_spec.flux, meta=resampled_spec.meta, wcs=None)

        if is_list(spec_to_match): #Propogate nans from spec_to_match to avoid wierd errors
            for i in range(len(spec_to_match)):
                resampled_spec[i].flux[np.isnan(spec_to_match[i].flux.value)] = np.nan
        else:
            resampled_spec.flux[np.isnan(spec_to_match.flux.value)] = np.nan

        return resampled_spec



def combine_spectra(spec_list):
    """Combines spectra assuming they are aligned pixel-by-pixel"""
    spec_final = spec_list[0]
    for i in range(1, len(spec_list)):
        spec_final = spec_final.add(spec_list[i], propagate_uncertainties=True)
    return spec_final


def combine_spectra_misaligned(spec_list, pixel_midpoints=None, propagate_uncertainty=False):
    """Combines spectra that might not be aligned pixel-by-pixel

    Misaligned spectra can arise when significant Radial Velocity shifts have been applied
    before combination.  This method is not suitable for precision radial velocities.

    Parameters
    ----------
    spec_list: list of Spectrum1D-like objects
        A list of spectra, with each spectrum possessing at least some overlap with the others
    propagate_uncertainty: boolean or String
        How to propagate uncertainty: if True and uncertainties are provided, it will propagate them.
        If False, it will determine uncertainties from sample standard deviation of the mean.
        If "max", and uncertainties are provided, it will take whichever is higher.
    pixel_midpoints: numpy.float or astropy.Quantity
        A vector of wavelength coordinates that represent the desired pixel midpoints
        of the output spectrum.  If None, the coordinates are determined from the input,
        using coarse bin spacings from the first input spectrum


    Returns
    -------
    combined_spec: Spectrum1D-like object
        Returns a spectrum of the same subclass as the input spectrum, with the flux values taking
        the weighted mean of the bins defined by pixel_midpoints.  The metadata is copied
        from the first spectrum in the list: we make no attempt to combine metadata
        from the multiple input spectra.  If input spectra have uncertainties, they are propagated
        using a formula for weighting the input uncertainties.  If input spectra do not have uncertainties,
        they are estimated from the sample standard deviation of the mean estimator.

    """
    fiducial_spec = spec_list[0]
    wavelength_unit = fiducial_spec.wavelength.unit  # Angstrom
    flux_unit = fiducial_spec.flux.unit  # dimensionless

    x = np.hstack([spectrum.wavelength.value for spectrum in spec_list])
    y = np.hstack([spectrum.flux.value for spectrum in spec_list])
    if fiducial_spec.uncertainty is None:
        has_uncertainty = False
        unc = np.ones_like(y)  # dummy values
    else:
        has_uncertainty = True
        unc = np.hstack([spectrum.uncertainty.array for spectrum in spec_list])

    # Remove NaNs
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(unc)
    x, y, unc = x[finite_mask], y[finite_mask], unc[finite_mask]

    # Determine pixel midpoints if not provided
    if pixel_midpoints is None:
        # Determine from data
        input_wavelength = fiducial_spec.wavelength.value
        typical_binsize = np.nanmedian(np.diff(input_wavelength))
        pixel_midpoints = np.arange(x.min(), x.max(), typical_binsize)

    # Determine pixel *edges* from pixel midpoints:
    bin_sizes = np.diff(pixel_midpoints)
    bin_sizes = np.diff(pixel_midpoints, prepend=pixel_midpoints[0] - bin_sizes[0])
    left_edges = pixel_midpoints - bin_sizes / 2
    right_edges = pixel_midpoints + bin_sizes / 2
    pixel_edges = np.hstack((left_edges, right_edges[-1]))

    ## Compute the weighted mean in each bin:
    weights = 1.0 * unc ** 2
    weights = weights / np.sum(weights)

    binned_sum_of_flux_times_weights = binned_statistic(
        x=x, values=y * weights, statistic=np.sum, bins=pixel_edges
    )

    binned_sum_of_weights = binned_statistic(
        x=x, values=weights, statistic=np.sum, bins=pixel_edges
    )

    weighted_mean_flux = (
        binned_sum_of_flux_times_weights.statistic / binned_sum_of_weights.statistic
    )

    ## Uncertainty estimate One:
    # Propagate the uncertainty in each bin
    binned_variance = binned_statistic(
        x=x, values=unc ** 2, statistic=np.sum, bins=pixel_edges
    )
    binned_count = binned_statistic(
        x=x, values=y, statistic="count", bins=pixel_edges
    )  # gives combined spectrum
    propagated_uncertainty = np.sqrt(binned_variance.statistic) / binned_count.statistic

    ## Uncertainty estimate Two:
    # Compute sample standard deviation of flux values in each bin
    binned_stddev = binned_statistic(
        x=x, values=y, statistic=np.std, bins=pixel_edges
    )  # gives combined spectrum

    sampled_uncertainty = binned_stddev.statistic / np.sqrt(binned_count.statistic)

    unc_out = sampled_uncertainty
    if has_uncertainty and (propagate_uncertainty == "max"):
        unc_out = np.maximum(propagated_uncertainty, sampled_uncertainty)
    elif has_uncertainty and (propagate_uncertainty == True):
        unc_out = propagated_uncertainty

    mask_out = np.isnan(unc_out)

    return fiducial_spec._copy(
        spectral_axis=pixel_midpoints * wavelength_unit,
        flux=weighted_mean_flux * flux_unit,
        uncertainty=StdDevUncertainty(unc_out),
        mask=mask_out,
        wcs=None,
    )


def apply_numpy_mask(spec, mask):
    """Applies a boolean mask to an input spectrum, numpy-style (True=Keep, False=Discard)


    Parameters
    ----------
    spec: Spectrum1D-like object
        Object storing spectrum
    mask: boolean mask, typically a numpy array
        The boolean mask with numpy-style masking: True means "keep" that index and False means discard that index
    """

    assert isinstance(spec, Spectrum1D), "Input must be a specutils Spectrum1D object"

    assert mask.sum() > 0, "The masked spectrum must have at least one pixel remaining"

    if len(mask) != len(spec.spectral_axis.value):
        raise IndexError(
            "Your boolean mask has {} entries and your spectrum has {} pixels.  "
            " The boolean mask should have the same shape as the spectrum."
        )

    if spec.meta is not None:
        meta_out = copy.deepcopy(spec.meta)
        # if "x_values" in spec.meta.keys():
        #     meta_out["x_values"] = meta_out["x_values"][mask]
    else:
        meta_out = None

    ndim = spec.flux.ndim #Grab dimensionality of spec, can be 1D or 2D
    if ndim == 1: #For 1D spectra
        if spec.uncertainty is not None:
            masked_unc = spec.uncertainty[mask]
        else:
            masked_unc = None

        if spec.mask is not None:
            mask_out = spec.mask[mask]
        else:
            mask_out = None

        return spec.__class__(
            spectral_axis=spec.wavelength.value[mask] * spec.wavelength.unit,
            flux=spec.flux[mask],
            mask=mask_out,
            uncertainty=masked_unc,
            wcs=None,
            meta=meta_out,
        )
    elif ndim == 2: #For 2D (e.g. slit) spectra
        if spec.uncertainty is not None:
            masked_unc = spec.uncertainty[:, mask]
        else:
            masked_unc = None

        if spec.mask is not None:
            mask_out = spec.mask[:, mask]
        else:
            mask_out = None

        return spec.__class__(
            spectral_axis=spec.wavelength.value[mask] * spec.wavelength.unit,
            flux=spec.flux[:, mask],
            mask=mask_out,
            uncertainty=masked_unc,
            wcs=None,
            meta=meta_out,
        )


def resample_list(spec_to_resample, specList, **kwargs):
    """
    Resample a single EchelleSpectrum or Spectrum1D object into a EchelleSpectrumList object.
    Useful for converting models into echelle spectra with multiple orders.

    Parameters
    ----------
    spec_to_resample: EchelleSpectrum or specutils Spectrum1D object
        Object storing spectrum (typically of a model) to be resampled onto the same grid as specList.
    specList: EchelleSpectrumList object
        Object storing an echelle spectrum (spectrum with multiple orders) with the wavelength grid to
        which spec_to_resample will be resampled.
    **kwargs: optional
        Extra arguments to be passed to specutils.manipulation.resample which is run to resample
        spec_to_resample to each order in specList
    """
    spec_out = copy.deepcopy(specList)
    for i in range(len(specList)):
        meta_out = specList[i].meta
        resampled_spec = spec_to_resample.resample(specList[i], **kwargs)
        if hasattr(resampled_spec, "unc"):
            spec_out[i] = specList[i].__class__(
                spectral_axis=resampled_spec.spectral_axis, flux=resampled_spec.flux, uncertainty=resampled_spec.unc, meta=meta_out, wcs=None)
        else:
            spec_out[i] = specList[i].__class__(
                spectral_axis=resampled_spec.spectral_axis, flux=resampled_spec.flux, meta=meta_out, wcs=None)            
    return spec_out
    

def resample_list_to_list(input_spec, spec_to_match):
    """Linearly resample input_spectra, which is a list of spectra, to match specrum_to_match and return an
     EchelleSpectrumList object with the same spectral axis and naned pixels as specrum_to_match. 
    
    Parameters
    -------
    input_spec :
        A EchelleSpectrumList, or similar to be resampled to match spec_to_match.
    specrum_to_match :
        An EchelleSpectrumLis tspectrum which the input_spec will be resampled to match in both wavelength and naned pixels
    """
    resampled_spec = resample_list(input_spec[0], spec_to_match) #Resample spectra
    for i in range(1, len(input_spec)):
        resampled_spec = resampled_spec + resample_list(input_spec[i], spec_to_match)
    return resampled_spec




def concatenate_orders(spec_list1, spec_list2):
    """
    Combine two EchelleSpectrumList objects into one.
    For example, combine IGRINS H and K bands.

    Parameters
    ----------
    spec_list1: EchelleSpectrumList object
        Echelle spectrum with multiple orders
    spec_list2: EchelleSpectrumList object
        Echelle spectrum with multiple orders to append onto the first list.
    """
    combined_list = copy.deepcopy(spec_list1)
    combined_list.extend(spec_list2)
    return combined_list

def is_list(check_this):
    """
    Checks if a given object is some sort of list or array object.
    For example, checks if a given object is an EchelleSpectrumList or single number for band math.

    Parameters
    check_this: Object to check

    Returns
    -------
    True: Object has more than one element (e.g. is a list or array)
    False: Object has a single element (e.g. a single variable like 10.0)
    """
    return isinstance(check_this, list) or ((type(check_this) is np.ndarray) and (len(np.shape(check_this)) > 1))

class Slit:
    def __init__(self, length=14.8, width=1.0, PA=90.0, guiding_error=1.5, n_axis=5000, name=''):
        """
        A  class to handle information about a spectrometer's slit, used for calculating things like slit losses

        Parameters 
        ----------
        length: float
            Length of the slit on the sky in arcsec.
        width: float
            Width of the slit on the sky in arcsec.
        PA: float
            Position angle of the slit on the sky in degrees.  Measured counterclockwise from North to East.
        guilding_error: float
            Estimate of the guiding error in arcsec.  This smears out the PSF fits in the East-West direction.
            This should be used carefully and only for telescopes on equitorial mounts.
        n_axis: float
            Size of axis for a 2D square array storing estimated profiles along the slit in 2D for later masking
        name: str
            Name of target.  Used in plots.

        """
        self.length = length
        self.width = width
        self.PA = PA
        self.guiding_error = guiding_error
        self.flux_correction = 1.0 #Store flux correction for later reuse

        half_n_axis = n_axis / 2
        dx = 1.2 * (length / n_axis)
        dy = 1.2 * (length / n_axis)
        x2d, y2d = np.meshgrid(np.arange(n_axis), np.arange(n_axis))
        x2d = (x2d - half_n_axis) * dx
        y2d = (y2d - half_n_axis) * dy
        self.x2d = x2d #Store x coordinates of 2D grid
        self.y2d = y2d #Store y coordinates on 2D grid
        self.f2d = np.zeros(np.shape(y2d)) #Store 2D grid of estimated fluxes'
        half_length = 0.5 * self.length
        half_width = 0.5 * self.width        
        self.mask = (x2d <= -half_width) | (x2d >= half_width) | (y2d <= -half_length) | (y2d >= half_length) #Create mask where every pixel inside slit is True and outside is False
        self.name = name #For plot titles to differentiate targets

    def ABBA(self, y, x=None, print_info=True, plot=False, plot_title='', pdfobj=None):
        """
        Given a collapsed spatial profile long slit for a point (stellar) source nodded
        ABBA along the slit, generate an estimate of A and B nods' 2D PSFs.
        The A and B nods are fit with Moffat functions which are then projected from 1D to 2D and then
        a mask is applied representing the slit and the the fraction of light in the PSFs inside the mask
        are integrated to estimate the fraction of light that passes through the slit.

        Parameters 
        ----------
        y: numpy array of floats
            Array representing the spatial profile of the source on the slit.  It should be the PSF for
            a point source nodded ABBA on the slit.
        x: numpy array of floats (optional)
            Array representing the spatial position along the slit in pixel space corrisponding to y.
        print_info: bool
            Print information about the fit.
        plot: bool
            Set to True to plot the 1D profile along the slit, Moffat fits, and residuals
        """
        slit_width_to_length_ratio = self.width / self.length
        if x is None: #Generate equally spaced x array if it is not provided
            ny = len(y)
            x = (np.arange(ny) / ny) * self.length
        #Find maximum and minimum
        i_max = np.where(y == np.nanmax(y))[0][0]
        i_min = np.where(y == np.nanmin(y))[0][0]
        if np.size(i_max) > 1: #Error catch for the rare event when two or more pixels match the max or min y values
            i_max = i_max[0]
        if np.size(i_min) > 1:
            i_min = i_min[0]
        #Fit 2 Moffat distributions to the psfs from A and B positions (see https://docs.astropy.org/en/stable/modeling/compound-models.html)
        g1 = models.Moffat1D(amplitude=y[i_max], x_0=x[i_max], alpha=1.0, gamma=1.0)
        g2 = models.Moffat1D(amplitude=y[i_min], x_0=x[i_min], alpha=1.0, gamma=1.0)


        fine_x = np.arange(0, 20, 0.001)
        integrated_g1 = np.nansum(g1(fine_x))
        integrated_g2 = np.nansum(g2(fine_x))

        gg_init = g1 + g2
        fitter = fitting.TRFLSQFitter()
        try: #Error catch
            gg_fit = fitter(gg_init, x, y, maxiter=10000)

            # #TESTING FLUX CORRECTION, CURRENTLY NOT IMPLEMENTED
            fine_x = np.arange(-20, 20, 0.00001)
            integrated_g1 = np.abs(np.nansum(gg_fit[0](fine_x)))
            integrated_g2 = np.abs(np.nansum(gg_fit[1](fine_x)))
            if integrated_g1 > integrated_g2:
                self.flux_correction = 0.5 + 0.5*(integrated_g1 / integrated_g2)
            else: #integrated_g1 <= integrated_g2
                self.flux_correction = 0.5 + 0.5*(integrated_g2 / integrated_g1)

            if plot:
                plt.figure()
                plt.plot(x, y, '.', label='Star Data')
                plt.plot(x, gg_fit(x), label='Moffat Distribution Fit')
                plt.plot(x, y-gg_fit(x), label='Residuals')
                plt.xlabel('Distance along slit (arcsec)')
                plt.ylabel('Flux')
                plt.legend()
                if plot_title != '':
                    plt.suptitle(plot_title)
                if self.name != '':
                    plt.title(self.name)
                if pdfobj is not None: #Save figure to file if PdfPages object is provided
                    pdfobj.savefig()
            if print_info:
                #log.info('FWHM A beam:', gg_fit[0].fwhm)
                #log.info('FWHM B beam:', gg_fit[1].fwhm)
                print('FWHM A beam:', gg_fit[0].fwhm)
                print('FWHM B beam:', gg_fit[1].fwhm)
            #Numerically estimate light through slit
            g1_fit = models.Moffat2D(amplitude=np.abs(gg_fit[0].amplitude), x_0=gg_fit[0].x_0 - 0.5*self.length, alpha=gg_fit[0].alpha, gamma=gg_fit[0].gamma)
            g2_fit = models.Moffat2D(amplitude=np.abs(gg_fit[1].amplitude), x_0=gg_fit[1].x_0 - 0.5*self.length, alpha=gg_fit[1].alpha, gamma=gg_fit[1].gamma)

            #simulate  guiding error by "smearing out" PSF
            # position_angle_in_radians = self.PA * (np.pi)/180.0 #PA in radians
            # fraction_guiding_error = np.cos(position_angle_in_radians)*self.guiding_error #arcsec, estimated by doubling average fwhm of moffet functions
            # diff_x0 = fraction_guiding_error * np.sin(position_angle_in_radians)
            # diff_y0 = fraction_guiding_error * np.cos(position_angle_in_radians)
            # g1_fit.x_0 += 0.5*diff_x0
            # g2_fit.x_0 += 0.5*diff_x0
            # g1_fit.y_0 += 0.5*diff_y0
            # g2_fit.y_0 += 0.5*diff_y0
            # n = 5
            # for i in range(n):
            #     self.f2d += (1/n)*(g1_fit(self.y2d, self.x2d) + g2_fit(self.y2d, self.x2d))
            #     g1_fit.x_0 -= (1/(n-1))*diff_x0
            #     g2_fit.x_0 -= (1/(n-1))*diff_x0
            #     g1_fit.y_0 -= (1/(n-1))*diff_y0
            #     g2_fit.y_0 -= (1/(n-1))*diff_y0
            self.f2d = np.abs(g1_fit(self.y2d, self.x2d) + g2_fit(self.y2d, self.x2d))
        except: #if bad fit, just return a bunch of nans
            self.f2d = np.zeros(np.shape(self.x2d))
            self.f2d[:] = np.nan

    def ONOFF(self, y, x=None, print_info=True, plot=False, plot_title='', pdfobj=None):
        """
        Given a collapsed spatial profile long slit for a point (stellar) source nodded off slit
        (ONOFF), generate an estimate of the single 2D PSF.
        The ON nod is fit with Moffat functions which is then projected from 1D to 2D and then
        a mask is applied representing the slit and the the fraction of light in the PSF inside the mask
        are integrated to estimate the fraction of light that passes through the slit.

        Parameters 
        ----------
        y: numpy array of floats
            Array representing the spatial profile of the source on the slit.  It should be the PSF for
            a point source nodded ABBA on the slit.
        x: numpy array of floats (optional)
            Array representing the spatial position along the slit in pixel space corrisponding to y.
        print_info: bool
            Print information about the fit.
        plot: bool
            Set to True to plot the 1D profile along the slit, Moffat fits, and residuals
        """
        slit_width_to_length_ratio = self.width / self.length
        if x is None: #Generate equally spaced x array if it is not provided
            ny = len(y)
            x = (np.arange(ny) / ny) * self.length
        #Find maximum 
        i_max = np.where(y == np.nanmax(y))[0][0]
        if np.size(i_max) > 1: #Error catch for the rare event when two or more pixels match the max or min y values
            i_max = i_max[0]
        #Fit Moffat distribution to the psf
        g1 = models.Moffat1D(amplitude=y[i_max], x_0=x[i_max], alpha=1.0, gamma=1.0)
        fitter = fitting.TRFLSQFitter()
        gg_fit = fitter(g1, x, y, maxiter=10000)
        if plot:
            plt.figure()
            plt.plot(x, y, '.', label='Star Data')
            plt.plot(x, gg_fit(x), label='Moffat Distribution Fit')
            plt.plot(x, y-gg_fit(x), label='Residuals')
            plt.xlabel('Distance along slit (arcsec)')
            plt.ylabel('Flux')
            plt.legend()
            if plot_title != '':
                plt.suptitle(plot_title)
            if self.name != '':
                plt.title(self.name)
            if pdfobj is not None: #Save figure to file if PdfPages object is provided
                pdfobj.savefig()
        if print_info:
            print('FWHM A beam:', gg_fit.fwhm)
        #Numerically estimate light through slit
        g1_fit = models.Moffat2D(amplitude=np.abs(gg_fit.amplitude), x_0=gg_fit.x_0 - 0.5*self.length, alpha=gg_fit.alpha, gamma=gg_fit.gamma)
        self.f2d = np.abs(g1_fit(self.y2d, self.x2d))

    def estimate_slit_throughput(self, normalize=True):
        """
        a mask is applied representing the slit and the the fraction of light in the PSFs inside the mask
        are integrated to estimate the fraction of light that passes through the slit.
        """
        if normalize: #You almost always want to normalize
            self.normalize()
        initial_estimated_fraction_through_slit = np.nansum(self.f2d[~self.mask]) / np.nansum(self.f2d) #Get fraction of light inside the slit mask
        #throughput correction calculated from monte carlo simualtions to convert the estimate to actual throughput, NOTE this is IGRINS specific
        throughput_correction_pointing_error_perpendicular_to_slit = models.Chebyshev2D(3, 3, c0_0=0.48615791, c1_0=0.32114591, c2_0=-0.0349109, c3_0=0.01192229, c0_1=-0.14611241, c1_1=-0.16490571, c2_1=-0.01045679, c3_1=0.01671257, c0_2=-0.02158197, c1_2=-0.02213463, c2_2=-0.00031099, c3_2=0.00002206, c0_3=0.01958147, c1_3=0.0302361, c2_3=0.01197411, c3_3=0.00266858, x_domain=(0.16251566201706763, 0.9999351856781427), y_domain=(0.000660434260749021, 1.999411871833546))
        throughput_correction_pointing_error_parallel_to_slit = models.Chebyshev2D(3, 3, c0_0=1.38140979, c1_0=1.54595726, c2_0=0.41060813, c3_0=-0.08619041, c0_1=1.26930674, c1_1=1.63718611, c2_1=0.60197849, c3_1=-0.16875158, c0_2=0.58352811, c1_2=0.65415013, c2_2=0.19616202, c3_2=-0.16319632, c0_3=0.09309019, c1_3=0.09296075, c2_3=-0.04573633, c3_3=-0.05109788, x_domain=(0.11128901778216015, 0.9999999946981601), y_domain=(0.0001453326995801696, 1.999326121378704))
        position_angle_in_radians = self.PA * (np.pi)/180.0 #PA in radians
        fraction_guiding_error_perpendicular = np.cos(position_angle_in_radians)*self.guiding_error #arcsec, estimated by doubling average fwhm of moffet functions
        fraction_guiding_error_parallel = np.sin(position_angle_in_radians)*self.guiding_error
        if np.any(self.f2d > 0.): #Error catch
            if initial_estimated_fraction_through_slit > 0:
                f_through_slit_perpendicular = throughput_correction_pointing_error_perpendicular_to_slit(initial_estimated_fraction_through_slit, fraction_guiding_error_perpendicular*(14.8/self.length)) #Apply a throughput correction to go from estimate to "actual" as determined from a monte carlo simualtion
                f_through_slit_parallel = throughput_correction_pointing_error_parallel_to_slit(initial_estimated_fraction_through_slit, fraction_guiding_error_parallel*(14.8/self.length)) #Apply a throughput correction to go from estimate to "actual" as determined from a monte carlo simualtion
                fraction_through_slit =  np.sqrt((f_through_slit_perpendicular*np.cos(position_angle_in_radians))**2 + (f_through_slit_parallel*np.sin(position_angle_in_radians))**2)
                if fraction_through_slit < 0.:
                    fraction_through_slit = 0.
                elif fraction_through_slit > 1.0:
                    fraction_through_slit = 1.0        
                return fraction_through_slit
            else:
                return np.nan
        else:
            return np.nan

    def clear(self):
        """
        Clear 2D flux array
        """
        self.f2d[:] = 0.0

    def normalize(self):
        """
        #Normalize each pixel by fraction of starlight
        """
        self.f2d = self.f2d / np.nansum(self.f2d)

    def plot2d(self, **kwarg):
        """
        Visualize the 2D distribution with slit overplotted
        """
        plt.figure()
        plt.imshow(self.f2d, origin='lower', aspect='auto', **kwarg)
        plt.colorbar()
        half_width = 0.5*self.width #Pkit slit outline
        half_length = 0.5*self.length
        # slit_ouline_x = np.array([-half_width, half_width, half_width, -half_width, -half_width])
        # slit_ouline_y = np.array([-half_length, -half_length, half_length, half_length, -half_length])
        # plt.plot(slit_ouline_x, slit_ouline_y, color='White', linewidth=3.0)
        numerical_mask = np.ones(np.shape(self.mask))
        plt.contour(self.mask, levels=[0.0,0.5, 1.0], colors='white', linewidths=2)
        if self.name != '':
            plt.title(self.name)

class absoluteFluxCalibration:
    def __init__(self, std_spec, synth_spec):
        """
        A  class to handle absolute flux calibration using a standard star spectrum and synthetic spectrum of the
        standard star.

        Parameters 
        ----------
        std_spec: EchelleSpectrum, EchelleSpectrumList, Spectrum1D, or SpectrumList like object 
            Actual spectrum of the standard star
        synth_spec: Spectrum1D, or SpectrumList like object from gollum
            Synethic spectrum of the standard star from a stellar atmosphere model read in with gollum, or something similar
        """
        self.std_spec = std_spec
        self.synth_spec = synth_spec


class photometry:
    def __init__(self):
        f = FilterGenerator()
        johnson_bands = np.array(['U', 'B','V','R','I']) #2MASS
        twoMass_bands = np.array(['J', 'H', 'Ks']) #Johnson filters
        self.bands =  np.concatenate((johnson_bands, twoMass_bands))
        self.f0_lambda = np.array([3.96526e-9*1e4, 6.13268e-9*1e4, 3.62708e-9*1e4, 2.17037e-9*1e4, 1.12588e-9*1e4, #Source: http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse&gname=Generic&gname2=Bessell&asttype=, with units converted from erg cm^-2 s^-1 ang^-1 to erg cm^-2 s^-1 um^-1 by multiplying by 1e-4
                3.129e-13*1e7, 1.133e-13*1e7, 4.283e-14*1e7]) #2MASS: Convert units to from W cm^-2 um^-1 to erg s^-1 cm^-2 um^-1
        self.x = np.arange(0.0, 10.0, 1e-6)
        self.delta_lambda = np.abs(self.x[1]-self.x[0])
        n = len(self.bands)
        tcurve_interp = []
        tcurve_resampled = []
        for i in range(n):
            if self.bands[i] in twoMass_bands:
                filt = f.reconstruct('2MASS/2MASS.'+self.bands[i])
            elif self.bands[i] in johnson_bands:
                filt = f.reconstruct('Generic/Johnson.'+self.bands[i])
            interp_obj = interp1d(filt.wavelength.to('um'), filt.transmittance, kind='cubic', fill_value=0.0, bounds_error=False)
            tcurve_interp.append(interp_obj)
            tcurve_resampled.append(interp_obj(self.x))
        self.tcurve_interp = tcurve_interp
        self.tcurve_resampled = tcurve_resampled

        # if band == 'K':
        #     band = 'Ks' #Catch to set K band band name to 'Ks'
        # twoMass_bands = np.array(['J', 'H', 'Ks'])
        # johnson_bands = np.array(['U', 'B','V','R','I'])
        # if band in twoMass_bands: #2MASS NIR filters
        #     f0_lambda = (np.array([3.129e-13, 1.133e-13, 4.283e-14]) * 1e7) [band == twoMass_bands][0] #Convert units to from W cm^-2 um^-1 to erg s^-1 cm^-2 um^-1
        #     filt = f.reconstruct('2MASS/2MASS.'+band)
        # elif band in johnson_bands: #Johnson filters
        #     f0_lambda = (np.array([417.5e-11, 632e-11, 363.1e-11, 217.7e-11, 112.6e-11]) * 1e4 )[band == johnson_bands][0] #Source: Table A2 from Bessel (1998), with units converted from erg cm^-2 s^-1 ang^-1 to erg cm^-2 s^-1 um^-1 by multiplying by 1e-4
        #     filt = f.reconstruct('Generic/Johnson.'+band)
        # else:
        #     raise Exception(
        #         "Band"+band+" not recognized. Must be U, B, V, R, I, J, H, or Ks."
        #     )        
        #self.f0_lambda = f0_lambda
        
        # self.tcurve_interp = interp1d(filt.wavelength.to('um'), filt.transmittance, kind='cubic', fill_value=0.0, bounds_error=False) #Create interp obj for the transmission curve
        # self.tcurve_resampled = self.tcurve_interp(self.x)
        #self.vega_V_flambdla_zero_point = 363.1e-7 #Vega flux zero point for V band from Bessell et al. (1998) in erg cm^2 s^-1 um^-1
        self.B = 0. #Store magnitudes, Johnson B and V bands
        self.V = 0.
        self.J = 0. #2MASS J, H, and K bands
        self.H = 0.
        self.K = 0.

    def scale(self, synth_spec, band='V', mag=0.0):
        i = self.grab_band_index(band)
        resampled_synthetic_spectrum =  LinInterpResampler(synth_spec , self.x*u.um).flux.value
        f_lambda = np.nansum(resampled_synthetic_spectrum * self.tcurve_resampled[i] * self.x * self.delta_lambda) / np.nansum(self.tcurve_resampled[i] * self.x * self.delta_lambda)
        magnitude_scale = 10**(0.4*(-mag))
        # print('self.f0_lambda', self.f0_lambda[i])
        # print('f_lambda', f_lambda)
        # print('magnitude_scale', magnitude_scale)
        scaled_synth_spec = synth_spec * (self.f0_lambda[i] / f_lambda) * magnitude_scale
        scaled_synth_spec = synth_spec.__class__(scaled_synth_spec) #Force class after band math to be the same as original class
        return scaled_synth_spec

    def get(self, synth_spec, band='V', resample=True, nan_catch=True):
        i = self.grab_band_index(band)
        if resample:
            resampled_synthetic_spectrum =  LinInterpResampler(synth_spec , self.x*u.um).flux.value
            f_lambda = np.nansum(resampled_synthetic_spectrum * self.tcurve_resampled[i] * self.x * self.delta_lambda) / np.nansum(self.tcurve_resampled[i] * self.x * self.delta_lambda)
            if np.isinf(f_lambda):
                breakpoint()
        else:
            x = synth_spec.wavelength.to('um').value
            delta_lambda = np.concatenate([[x[1]-x[0]], x[1:] - x[:-1]])
            interp_obj = interp1d(self.x, self.tcurve_resampled[i], kind='linear', fill_value=0.0, bounds_error=False)
            resampled_tcurve = interp_obj(x)
            goodpix = (synth_spec.flux.value > 1e-20) & (synth_spec.flux.value < 1e10)
            f_lambda = np.nansum(synth_spec.flux.value[goodpix] * resampled_tcurve[goodpix] * x[goodpix] * delta_lambda[goodpix]) / np.nansum(resampled_tcurve[goodpix] * x[goodpix] * delta_lambda[goodpix])
            print(np.sum(np.isfinite(synth_spec.flux.value)))
            #print(np.nansum(synth_spec.flux.value * resampled_tcurve * x * delta_lambda))
            print(np.nansum(resampled_tcurve * x * delta_lambda))
        magnitude = -2.5 * np.log10(f_lambda / self.f0_lambda[i])
        if nan_catch and (np.isnan(magnitude) or ~np.isfinite(magnitude)): #Catch to prevent nan values from being passed, since FITS headers are incompatible with nans, if a nan is found, return -999 to indicate the result was indefinite
            return -999
        return magnitude

    def grab_band_index(self, band):
        if band == 'K':
            band = 'Ks' #Catch to set K band band name to 'Ks' 
        i = np.where(band == self.bands)[0][0]   
        return i

    def get_simbad_photometry(self, name='', coords=''):
        """
        A function that grabs the B, V, J, H, and K magnitudes from SIMBAD for an object.

        Parameters 
        ----------
        name: str
            The SIMBAD searchable name for the object.
        synth_spec: str
            The the RA and DEC coordinates of the object to search formated as f'{RA} {DEC}'. Decimal degrees or H:M:S/D:A:A format accepted.
        """ 

        #try querying the object by name
        Simbad.add_votable_fields('allfluxes')
        query_result = Simbad.query_object(name)

        #if the name is not simbad searchable search by coordinates instead (if there are coordinates)
        if len(query_result) == 0 and coords != '': 
            #this prints in light pink color to terminal; see: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
            print(f'\n\033[38;5;{196}m{name}'+' IS NOT SIMBAD SEARCHABLE.  SEARCHING USING COORDS: '+coords+'\033[0m')

            #query_region doesnt work well when the string has coords in HMS format, but making the string into a SkyCoord object seems to fix things
            if ':' in coords:
                sky_coord = SkyCoord(coords, unit = (u.hourangle, u.deg), frame = 'icrs')
            else:
                sky_coord = SkyCoord(coords, unit = (u.deg, u.deg), frame = 'icrs')

            #coordinates at McDonald can be far off for IGRINS, this is the same radius we use to query objects for RRISA cross-matching
            query_result = Simbad.query_region(sky_coord, radius='20 arcsec')

            #print the name of the standard found so the users can check to make sure it is the correct one
            #this prints in bright red to terminal; see:https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
            print(f'\n\033[38;5;{63}mSIMBAD SEARCHABLE MAIN ID IS \033[0m'+ f"\033[38;5;{196}m{query_result['main_id'][0]}\033[0m", '\n')

            #set the object's magnitudes attributes with the result of the search.
            self.B = query_result['B'][0] 
            self.V = query_result['V'][0]
            self.J = query_result['J'][0]
            self.H = query_result['H'][0]
            self.K = query_result['K'][0]
            # self.B = query_result['flux'][query_result['flux.filter']=='B'].item()
            # self.V = query_result['flux'][query_result['flux.filter']=='V'].item()
            # self.J = query_result['flux'][query_result['flux.filter']=='J'].item()
            # self.H = query_result['flux'][query_result['flux.filter']=='H'].item()
            # self.K = query_result['flux'][query_result['flux.filter']=='K'].item()

        #if the given object name returns a SIMBAD result
        elif len(query_result) > 0:
            #print the name of the standard found so the users can check to make sure it is the correct one
            #this prints in bright red to terminal; see:https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
            print(f'\n\033[38;5;{63}mSIMBAD SEARCHABLE MAIN ID IS \033[0m'+ f"\033[38;5;{196}m{query_result['main_id'][0]}\033[0m", '\n')

            #set the object's magnitudes attributes with the result of the search.
            self.B = query_result['B'][0] 
            self.V = query_result['V'][0]
            self.J = query_result['J'][0]
            self.H = query_result['H'][0]
            self.K = query_result['K'][0]
            #set the object's magnitudes attributes with the result of the search.
            # self.B = query_result['flux'][query_result['flux.filter']=='B'].item()
            # self.V = query_result['flux'][query_result['flux.filter']=='V'].item()
            # self.J = query_result['flux'][query_result['flux.filter']=='J'].item()
            # self.H = query_result['flux'][query_result['flux.filter']=='H'].item()
            # self.K = query_result['flux'][query_result['flux.filter']=='K'].item()

        #the object name is not SIMBAD searchable and the coords are not given
        else:
            #print an error
            print(f'\n\033[38;5;{196}m{name} IS NOT SIMBAD SEARCHABLE AND NO OBJECT COORDINATES WERE GIVEN.\033[0m')


    def set_photometry(self, synth_spec, nan_catch=True):
        """
        A function that calculates the B, V, J, H, and K magnitudes from a model spectrum. 

        Parameters 
        ----------
        name: str
            The SIMBAD searchable name for the object.
        synth_spec: str
            The the RA and DEC coordinates of the object to search formated as f'{RA} {DEC}'. Decimal degrees or H:M:S/D:A:A format accepted.
        """ 
        #Calculate  B, V, J, H, K mags from properly scaled synethetic spectrum
        self.B = self.get(synth_spec, band='B', nan_catch=nan_catch)
        self.V = self.get(synth_spec, band='V', nan_catch=nan_catch)
        self.J = self.get(synth_spec, band='J', nan_catch=nan_catch)
        self.H = self.get(synth_spec, band='H', nan_catch=nan_catch)
        self.K = self.get(synth_spec, band='K', nan_catch=nan_catch)

    def scale_to_v(self, synth_spec): #Convenience function to scale to stored V band (e.g. from simbad)
        return self.scale(synth_spec, band='V', mag=self.V)

    def scale_to_k(self, synth_spec):  #Convenience function to scale to stored K band (e.g. from simbad)
        return self.scale(synth_spec, band='K', mag=self.K)
        

