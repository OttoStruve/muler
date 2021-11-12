import numpy as np
import copy
from specutils.spectra import Spectrum1D
from astropy.nddata.nduncertainty import StdDevUncertainty


def combine_spectra(spec_list):
    """Combines spectra assuming they are aligned pixel-by-pixel"""
    spec_final = spec_list[0]
    for i in range(1, len(spec_list)):
        spec_final = spec_final.add(spec_list[i], propagate_uncertainties=True)
    return spec_final


def combine_spectra_misaligned(spec_list):
    """Combines spectra that might not be aligned pixel-by-pixel

    Misaligned spectra can arise when Radial Velocity shifts have been applied.
    """
    # spec_final = spec_list[0]
    # for i in range(1, len(spec_list)):
    #    spec_final = spec_final.add(spec_list[i], propagate_uncertainties=True)
    # return spec_final


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
        masked_unc = StdDevUncertainty(spec.uncertainty.array[mask])
    else:
        masked_unc = None

    if spec.mask is not None:
        mask_out = spec.mask[mask]
    else:
        mask_out = None

    if spec.meta is not None:
        meta_out = copy.deepcopy(spec.meta)
        if hasattr(spec.meta, "x_values"):
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
