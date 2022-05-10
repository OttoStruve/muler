# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
import astropy
import pytest
import time
from muler.igrins import IGRINSSpectrum, IGRINSSpectrumList
from specutils import Spectrum1D

# from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob
import astropy.units as u


local_files = glob.glob("**/SDCH*.spec_a0v.fits", recursive=True)
local_files_2 = glob.glob("**/SDCH*.spec.fits", recursive=True)
file = local_files[0]
file_2 = local_files_2[0]


def test_basic():
    """Do the basic methods work?"""

    spec = IGRINSSpectrum(file=file, order=10)

    assert spec is not None
    assert isinstance(spec, Spectrum1D)
    assert isinstance(spec.flux, np.ndarray)
    assert len(spec.flux) == len(spec.wavelength)

    # Check that numpy operations persist the units
    assert np.nanmedian(spec.flux).unit == spec.flux.unit

    assert spec.mask.sum() > 0

    new_spec = spec.normalize()
    assert new_spec.shape[0] == spec.shape[0]
    assert np.nanmedian(new_spec.flux) == 1

    new_spec = spec.remove_nans().normalize()

    assert new_spec.shape[0] < spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None

    new_spec = new_spec.remove_outliers(threshold=6)

    assert len(new_spec.flux) > 0
    assert new_spec.shape[0] <= spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None

    new_spec = spec.trim_edges()

    assert new_spec.shape[0] < spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None

    # Do routine operations commute?
    new_spec1 = spec.trim_edges().remove_nans()
    new_spec2 = spec.remove_nans().trim_edges()

    for attribute in ["flux", "wavelength", "mask"]:
        lhs = new_spec1.__getattribute__(attribute)
        rhs = new_spec2.__getattribute__(attribute)
        assert len(lhs) == len(rhs)
        assert len(new_spec1.uncertainty.array) == len(lhs)
        assert len(new_spec2.uncertainty.array) == len(rhs)
        assert len(new_spec1.flux) == len(rhs)

    assert np.all(new_spec1.meta["x_values"] == new_spec2.meta["x_values"])

    ax = new_spec.plot(label="demo", color="r")
    assert ax is not None


def test_normalize():
    """Do the basic methods work?"""

    spec = IGRINSSpectrum(file=file, order=10)

    new = spec.normalize(normalize_by="median")

    assert isinstance(new, Spectrum1D)
    assert new.flux.unit == u.dimensionless_unscaled
    assert len(spec.flux) == len(spec.wavelength)

    new = spec.normalize(normalize_by="mean")

    assert isinstance(new, Spectrum1D)
    assert new.flux.unit == u.dimensionless_unscaled
    assert len(spec.flux) == len(spec.wavelength)

    new = spec.normalize(normalize_by=42.0)

    assert isinstance(new, Spectrum1D)
    assert new.flux.unit == u.dimensionless_unscaled
    assert len(spec.flux) == len(spec.wavelength)

    # This test should work: we now support (correct) units!
    new = spec.normalize(normalize_by=42.0 * u.ct)
    assert isinstance(new, Spectrum1D)
    assert new.flux.unit == u.dimensionless_unscaled
    assert len(spec.flux) == len(spec.wavelength)

    # This test should not work: the normalizing units must comport
    with pytest.raises(u.UnitConversionError):
        new = spec.normalize(normalize_by=42.0 * u.m)

    # Test normalization on a list of spectra
    spec_list = IGRINSSpectrumList.read(file)

    # We don't (yet) support normalize_by in lists
    with pytest.raises(TypeError):
        new = spec_list.normalize(normalize_by="median")

    new = spec_list.normalize(order_index=5)

    assert isinstance(new, IGRINSSpectrumList)
    assert len(new) == len(spec_list)
    assert new[0].flux.unit == u.dimensionless_unscaled


def test_uncertainty_spec_a0v():
    """Does uncertainty propagation work?"""

    spec = IGRINSSpectrum(file=file, order=10)

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
    assert np.isclose(snr_med, snr_old_med)

    new_spec = spec.remove_nans().deblaze()

    assert len(new_spec.flux) == len(new_spec.uncertainty.array)
    assert np.all(new_spec.uncertainty.array > 0)

    snr_vec = new_spec.flux / new_spec.uncertainty.array
    snr_med = np.nanmedian(snr_vec.value)
    #assert np.isclose(snr_med, snr_old_med, atol=0.005)


def test_uncertainty_spec_fits():
    """Does uncertainty propagation work?"""

    spec = IGRINSSpectrum(file=file_2, order=10)

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
    assert np.isclose(snr_med, snr_old_med)

    new_spec = spec.remove_nans().deblaze()

    assert len(new_spec.flux) == len(new_spec.uncertainty.array)
    assert np.all(new_spec.uncertainty.array > 0)

    snr_vec = new_spec.flux / new_spec.uncertainty.array
    snr_med = np.nanmedian(snr_vec.value)
    #assert np.isclose(snr_med, snr_old_med, atol=0.005)


def test_equivalent_width():
    """Can we measure equivalent widths?"""

    spec = IGRINSSpectrum(file=file)
    mu = np.median(spec.wavelength.value)
    equivalent_width = spec.measure_ew(mu)

    assert equivalent_width is not None
    assert type(equivalent_width) is not int
    assert type(equivalent_width) is astropy.units.quantity.Quantity
    new_unit = equivalent_width.to(spec.wavelength.unit)
    assert new_unit.unit == spec.wavelength.unit


def test_smoothing():
    """Does smoothing and outlier removal work?"""
    spec = IGRINSSpectrum(file=file)
    new_spec = spec.remove_nans().remove_outliers(threshold=3)

    assert len(new_spec.flux) > 0
    assert new_spec.shape[0] <= spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None


def test_sorting():
    """Does Sorting method work?"""
    spec_list = IGRINSSpectrumList.read(file=file)
    full_spec = spec_list.remove_nans().stitch()

    new_spec = full_spec.sort()

    assert new_spec is not None
    assert len(new_spec.flux) == len(full_spec.flux)

    assert np.all(np.diff(new_spec.wavelength.value) > 0)


def test_RV():
    """Does RV shifting work"""

    spec = IGRINSSpectrum(file=file)

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
    """Does uncertainty propagation work?"""

    spec = IGRINSSpectrum(file=file)

    new_spec = spec.remove_nans().deblaze()

    assert new_spec is not None
    assert isinstance(new_spec, Spectrum1D)


@pytest.mark.parametrize(
    "precache_hdus", [True, False],
)
def test_spectrumlist_performance(precache_hdus):
    """Does the Spectrum List work?"""
    t0 = time.time()
    spec_list = IGRINSSpectrumList.read(file, precache_hdus=precache_hdus)
    t1 = time.time()
    net_time = t1 - t0
    print(f"\n\t Precached HDUs {precache_hdus}: {net_time:0.5f} seconds", end="\t")

    assert spec_list is not None
