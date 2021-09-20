# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
import pytest
import time
from muler.hpf import HPFSpectrum, HPFSpectrumList
from specutils import Spectrum1D

# from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob
import astropy

local_files = glob.glob("**/Goldilocks_*.spectra.fits", recursive=True)
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

    assert hasattr(new_spec, "provenance")
    assert type(new_spec.provenance) == str
    assert hasattr(new_spec, "pipeline")
    assert new_spec.pipeline in ["Goldilocks", "HPF"]

    ax = new_spec.plot(label="demo", color="r")
    assert ax is not None


def test_bad_inputs():
    """These tests should fail"""
    with pytest.raises(NameError):
        spec = HPFSpectrum(file="junk_file.txt")


def test_equivalent_width():
    """Can we measure equivalent widths?"""

    spec = HPFSpectrum(file=file, order=4)
    mu = 8600  # make sure it is in given order
    equivalent_width = spec.measure_ew(mu)

    assert equivalent_width is not None
    assert type(equivalent_width) is not int
    assert type(equivalent_width) is astropy.units.quantity.Quantity
    assert equivalent_width.unit is spec.wavelength.unit


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

    new_spec = spec.normalize().deblaze()

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

    ## Normalize should scale both target and sky flux by the same scalar
    # assert np.nanmedian(new_spec2.sky.flux.value) != np.nanmedian(
    #    new_spec.sky.flux.value
    # )
    assert np.median(new_spec2.flux.value) == 1.0

    # The sky/lfc fibers should not have their own sky/lfc fibers: that's redundant
    assert "sky" not in spec.sky.meta.keys()
    assert "lfc" not in spec.sky.meta.keys()
    assert "sky" not in spec.lfc.meta.keys()
    assert "lfc" not in spec.lfc.meta.keys()

    assert spec.lfc.meta["provenance"] == "Laser Frequency Comb"
    assert spec.sky.meta["provenance"] == "Sky fiber"
    assert spec.meta["provenance"] == "Target fiber"


def test_RV():
    """Does RV shifting work"""

    spec = HPFSpectrum(file=file)

    assert spec.uncertainty is not None
    assert hasattr(spec, "barycentric_correct")

    correction_velocity = spec.estimate_barycorr()

    assert isinstance(spec.RA, astropy.units.quantity.Quantity)
    assert isinstance(spec.DEC, astropy.units.quantity.Quantity)
    assert correction_velocity is not None
    assert isinstance(correction_velocity, astropy.units.quantity.Quantity)

    new_spec = spec.barycentric_correct()
    assert new_spec is not None
    assert isinstance(new_spec, Spectrum1D)


def test_deblaze():
    """Does the HPF-specific deblazing work?"""
    spec = HPFSpectrum(file=file)

    # There are two blaze templates uploaded, we prefer the Goldilocks one
    blaze_template = spec.get_static_blaze_template()

    assert isinstance(blaze_template, HPFSpectrum)

    blaze_template = spec.get_static_blaze_template(method="2021_median")
    assert isinstance(blaze_template, HPFSpectrum)

    new_spec = spec.deblaze()
    assert isinstance(new_spec, HPFSpectrum)

    # There are two blaze templates uploaded, we prefer the Goldilocks one
    A0V_template = spec.get_static_A0V_template()
    assert isinstance(A0V_template, HPFSpectrum)


def test_sky_subtraction():
    """Does our sky subtraction work in all modes?"""
    spec = HPFSpectrum(file=file)

    # You can get back a wavelength dependent template for scaling the sky fiber
    template = spec.get_static_sky_ratio_template()
    assert isinstance(template, HPFSpectrum)

    for method in ["naive", "scalar", "vector"]:
        new_spec = spec.sky_subtract(method=method)
        assert isinstance(new_spec, HPFSpectrum)

    with pytest.raises(NotImplementedError):
        new_spec = spec.sky_subtract(method="Danny")


def test_HPF_spectrum_list():
    """Does our sky subtraction work in all modes?"""
    spectra = HPFSpectrumList.read(file)

    new_spectra = spectra.sky_subtract(method="scalar")
    assert isinstance(new_spectra, HPFSpectrumList)

    new_spectra = new_spectra.deblaze()
    assert isinstance(new_spectra, HPFSpectrumList)


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
