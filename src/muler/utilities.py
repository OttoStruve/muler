import numpy as np
import copy
from specutils.spectra import Spectrum1D
from astropy.nddata.nduncertainty import StdDevUncertainty
from scipy.stats import binned_statistic


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
        spec_out[i] = spec_to_resample.resample(specList[i], **kwargs)
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
