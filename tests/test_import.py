from muler.igrins import IGRINSSpectrum
from specutils import Spectrum1D
import numpy as np
import glob

local_files = glob.glob("data/SDCH*.spec_a0v.fits")
file = local_files[0]


def test_basic():
    """Does the import work?"""

    spec = IGRINSSpectrum(file=file, order=10)

    assert spec is not None
    assert isinstance(spec, Spectrum1D)
    assert isinstance(spec.flux, np.ndarray)
    assert len(spec.flux) == len(spec.wavelength)
    assert spec.mask.sum() > 0
