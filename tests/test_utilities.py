# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
from muler.hpf import HPFSpectrum
from specutils import Spectrum1D
import glob
from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np

from muler.utilities import combined_spectra

# There should be exactly 3 files in the example data directory
local_files = glob.glob("**/01_A0V_standards/Goldilocks_*.spectra.fits", recursive=True)


def test_combine():
    """Does the combine utility work?"""
    n_files = len(local_files)
    this_spec_list = [
        HPFSpectrum(file=local_files[i], order=10) for i in range(n_files)
    ]

    assert len(this_spec_list) == n_files
    assert n_files == 3
    assert this_spec_list is not None

    coadded_spectrum = combined_spectra(this_spec_list)

    assert isinstance(coadded_spectrum, Spectrum1D)
    assert isinstance(coadded_spectrum, HPFSpectrum)
    assert len(coadded_spectrum.flux) == len(coadded_spectrum.uncertainty.array)
