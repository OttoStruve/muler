# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
import pytest
import time
from muler.hpf import HPFSpectrum, HPFSpectrumList
from specutils import Spectrum1D

# from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob

local_files = glob.glob("data/Goldilocks_*.spectra.fits")
file = local_files[0]


def test_basic():
    """Do the basic methods work?"""

    spec = HPFSpectrum(file=file, order=10)

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

    new_spec = spec.trim_edges()

    assert new_spec.shape[0] < spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None

    ax = new_spec.plot(label="demo", color="r")
    assert ax is not None


def test_smoothing():
    """Does smoothing and outlier removal work?"""
    spec = HPFSpectrum(file=file, order=10)
    new_spec = spec.remove_outliers(threshold=3)

    assert len(new_spec.flux) > 0
    assert new_spec.shape[0] <= spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None


def test_uncertainty():
    """Does uncertainty propagation work?"""

    spec = HPFSpectrum(file=file, order=10)

    assert spec.uncertainty is not None
    assert hasattr(spec.uncertainty, "array")
    assert len(spec.flux) == len(spec.uncertainty.array)
    assert spec.flux.unit == spec.uncertainty.unit

    new_spec = spec.remove_nans()

    assert len(new_spec.flux) == len(new_spec.uncertainty.array)
    assert np.all(new_spec.uncertainty.array > 0)

    snr_old_vec = spec.flux / spec.uncertainty.array
    snr_old_med = np.nanmedian(snr_old_vec.value)

    new_spec = spec.normalize()

    snr_vec = new_spec.flux / new_spec.uncertainty.array
    snr_med = np.nanmedian(snr_vec.value)
    assert np.isclose(snr_med, snr_old_med, atol=0.005)


@pytest.mark.parametrize(
    "precache_hdus", [True, False],
)
def test_spectrumlist_performance(precache_hdus):
    """Does the Spectrum List work?"""
    t0 = time.time()
    spec_list = HPFSpectrumList.read(file, precache_hdus=precache_hdus)
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t Precached HDUs {precache_hdus}: {net_time:0.5f} seconds", end="\t")

    assert spec_list is not None
