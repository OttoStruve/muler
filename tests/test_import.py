# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
from muler.igrins import IGRINSSpectrum
from specutils import Spectrum1D
from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob

local_files = glob.glob("data/SDCH*.spec_a0v.fits")
file = local_files[0]


def test_basic():
    """Do the basic methods work?"""

    spec = IGRINSSpectrum(file=file, order=10)

    assert spec is not None
    assert isinstance(spec, Spectrum1D)
    assert isinstance(spec.flux, np.ndarray)
    assert len(spec.flux) == len(spec.wavelength)
    assert spec.mask.sum() > 0

    new_spec = spec.remove_nans()

    assert new_spec.shape[0] < spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None

    new_spec = spec.normalize()

    assert new_spec.shape[0] == spec.shape[0]
    assert np.nanmedian(new_spec.flux) == 1


def test_uncertainty():
    """Dooes uncertainty propagation work?"""

    spec = IGRINSSpectrum(file=file, order=10)

    assert spec.uncertainty is not None
    assert hasattr(spec.uncertainty, "array")
    assert len(spec.flux) == len(spec.uncertainty.array)
    assert spec.flux.unit == spec.uncertainty.unit

    new_spec = spec.remove_nans()

    assert len(new_spec.flux) == len(new_spec.uncertainty.array)

    snr_old_vec = spec.flux / spec.uncertainty.array
    snr_old_med = np.nanmedian(snr_old_vec.value)

    new_spec = spec.normalize()

    snr_vec = new_spec.flux / new_spec.uncertainty.array
    snr_med = np.nanmedian(snr_vec.value)
    assert snr_med == snr_old_med
