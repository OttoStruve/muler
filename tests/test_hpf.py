# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
import pytest
import time
from muler.hpf import HPFSpectrum, HPFSpectrumList
from specutils import Spectrum1D

# from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob
import astropy

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

def test_equivalent_width():
    """Can we measure equivalent widths?"""

    spec = HPFSpectrum(file=file, order=10)
    mu=7
    equivalent_width = spec.measure_ew(mu)

    assert equivalent_width is not None
    assert type(equivalent_width) is not int
    assert equivalent_width > 0.0
    assert type(equivalent_width) is astropy.units.quantity.Quantity


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


def test_sky_and_lfc():
    """Do we track sky and lfc?"""

    spec = HPFSpectrum(file=file, order=10)

    assert spec.sky is not None
    assert isinstance(spec.sky, Spectrum1D)
    assert spec.sky == spec.meta["sky"]

    assert spec.lfc is not None
    assert isinstance(spec.lfc, Spectrum1D)

    assert hasattr(spec.sky, "flux")
    assert isinstance(spec.sky.flux, np.ndarray)
    assert len(spec.sky.flux) == len(spec.flux)
    assert spec.flux.unit == spec.sky.unit

    new_spec = spec.remove_nans()

    assert new_spec.sky is not None
    assert hasattr(new_spec.sky, "flux")

    new_spec2 = new_spec.normalize()

    assert new_spec2.sky is not None
    assert isinstance(new_spec2.sky, Spectrum1D)
    assert hasattr(new_spec2.sky, "flux")

    # Normalize should scale both target and sky flux by the same scalar
    assert np.nanmedian(new_spec2.sky.flux.value) != np.nanmedian(
        new_spec.sky.flux.value
    )
    assert np.median(new_spec2.flux.value) == 1.0

    # The sky/lfc fibers should not have their own sky/lfc fibers: that's redundant
    assert "sky" not in spec.sky.meta.keys()
    assert "lfc" not in spec.sky.meta.keys()
    assert "sky" not in spec.lfc.meta.keys()
    assert "lfc" not in spec.lfc.meta.keys()

    assert spec.lfc.meta["provenance"] == "Laser Frequency Comb"
    assert spec.sky.meta["provenance"] == "Sky fiber"
    assert spec.meta["provenance"] == "Target fiber"


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
