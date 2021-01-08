from muler.igrins import IGRINSSpectrum
from specutils import Spectrum1D
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
    assert new_spec.mask is None

    new_spec = spec.normalize()

    assert new_spec.shape == spec.shape[0]
    assert np.median(new_spec.flux) == 1

