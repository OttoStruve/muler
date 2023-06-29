import numpy as np
import copy
from specutils.spectra import Spectrum1D
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy.modeling import models, fitting #import the astropy model fitting package
from scipy.stats import binned_statistic
from specutils.manipulation import LinearInterpolatedResampler
LinInterpResampler = LinearInterpolatedResampler()


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
                resampled_spec = resample_list(input_spec, specrum_to_match) #Resample spectrum
            else:
                resampled_spec = LinInterpResampler(input_spec, spec_to_match.spectral_axis)
            resampled_spec = spec_to_match.__class__( #Ensure resampled_spec is the same object as spec_to_match
                spectral_axis=resampled_spec.spectral_axis, flux=resampled_spec.flux, meta=self.meta, wcs=None)

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


def combine_spectra_misaligned(
    spec_list, pixel_midpoints=None, propagate_uncertainty=False
):
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

    if spec.uncertainty is not None:
        masked_unc = spec.uncertainty[mask]
    else:
        masked_unc = None

    if spec.mask is not None:
        mask_out = spec.mask[mask]
    else:
        mask_out = None

    if spec.meta is not None:
        meta_out = copy.deepcopy(spec.meta)
        if "x_values" in spec.meta.keys():
            meta_out["x_values"] = meta_out["x_values"][mask]
    else:
        meta_out = None

    return spec.__class__(
        spectral_axis=spec.wavelength.value[mask] * spec.wavelength.unit,
        flux=spec.flux[mask],
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
    if np.size(check_this) > 1:
        return True
    else:
        return False

def estimate_slit_throughput_ABBA(y, x=None, slit_length=14.8, slit_width=1.0, PA=90.0, guiding_error=1.5, print_info=True):
    """
    Given a collapsed spatial profile long slit for a point (stellar) source nodded
    ABBA along the slit, returns a numerical estimate of the fraction of light through the slit.
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
    slit_length: float
        Length of the slit on the sky in arcsec.
    slit_width: float
        Width of the slit on the sky in arcsec.
    PA: float
        Position angle of the slit on the sky in degrees.  Measured counterclockwise from North to East.
    guilding_error: float
        Estimate of the guiding error in arcsec.  This smears out the PSF fits in the East-West direction.
        This should be used carefully and only for telescopes on equitorial mounts.
    print_info: bool
        Print information about the fit.

    Returns: 
    ----------
    Float: The fraction of light from the source estimated to pass through the slit.
    """


    slit_width_to_length_ratio = slit_width / slit_length
    if x is None: #Generate equally spaced x array if it is not provided
        ny = len(y)
        x = (np.arange(ny) / ny) * slit_length
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
    gg_init = g1 + g2
    fitter = fitting.TRFLSQFitter()
    gg_fit = fitter(gg_init, x, y)
    if print_info:
        print('FWHM A beam:', gg_fit[0].fwhm)
        print('FWHM B beam:', gg_fit[1].fwhm)
    #Numerically estimate light through slit
    g1_fit = models.Moffat2D(amplitude=np.abs(gg_fit[0].amplitude), x_0=gg_fit[0].x_0 - 0.5*slit_length, alpha=gg_fit[0].alpha, gamma=gg_fit[0].gamma)
    g2_fit = models.Moffat2D(amplitude=np.abs(gg_fit[1].amplitude), x_0=gg_fit[1].x_0 - 0.5*slit_length, alpha=gg_fit[1].alpha, gamma=gg_fit[1].gamma)
    #Generate a 2D grid in x and y for numerically calculating slit loss
    n_axis = 5000
    half_n_axis = n_axis / 2
    max_x = np.nanmax(x)
    dx = 1.2 * (max_x / n_axis)
    dy = 1.2 * (max_x / n_axis)
    y2d, x2d = np.meshgrid(np.arange(n_axis), np.arange(n_axis))
    x2d = (x2d - half_n_axis) * dx
    y2d = (y2d - half_n_axis) * dy
    #simulate  guiding error by "smearing out" PSF
    position_angle_in_radians = PA * (np.pi)/180.0 #PA in radians
    fraction_guiding_error = np.cos(position_angle_in_radians)*guiding_error #arcsec, estimated by doubling average fwhm of moffet functions
    diff_x0 = fraction_guiding_error * np.cos(position_angle_in_radians)
    diff_y0 = fraction_guiding_error * np.sin(position_angle_in_radians)
    g1_fit.x_0 += 0.5*diff_x0
    g2_fit.x_0 += 0.5*diff_x0
    g1_fit.y_0 += 0.5*diff_y0
    g2_fit.y_0 += 0.5*diff_y0
    profiles_2d = np.zeros(np.shape(x2d))
    n = 5
    for i in range(n):
        profiles_2d += (1/n)*(g1_fit(x2d, y2d) + g2_fit(x2d, y2d))
        g1_fit.x_0 -= (1/(n-1))*diff_x0
        g2_fit.x_0 -= (1/(n-1))*diff_x0
        g1_fit.y_0 -= (1/(n-1))*diff_y0
        g2_fit.y_0 -= (1/(n-1))*diff_y0
    #Mask esitmated 2D PSFs to estimate fraction of light through the slit
    profiles_2d = profiles_2d / np.nansum(profiles_2d) #Normalize each pixel by fraction of starlight
    outside_slit = (y2d <= -0.5*slit_width) | (y2d >= 0.5*slit_width) | (x2d <= -0.5*slit_length) | (x2d >= 0.5*slit_length) #Apply mask
    profiles_2d[outside_slit] = np.nan
    fraction_through_slit = np.nansum(profiles_2d)

    return fraction_through_slit
