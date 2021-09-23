# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
from astropy.nddata.nduncertainty import StdDevUncertainty
import astropy.units as u
import pytest
import time
from muler.hpf import HPFSpectrum, HPFSpectrumList
from specutils import Spectrum1D, spectra

# from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob
import astropy
from specutils.spectra import spectral_axis

local_files = glob.glob("**/Goldilocks_*.spectra.fits", recursive=True)
A0V_files = [file for file in local_files if "01_A0V_standards" in file]
file = local_files[0]


@pytest.mark.parametrize(
    "file", local_files,
)
def test_basic(file):
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


@pytest.mark.parametrize(
    "file", local_files,
)
def test_bad_inputs(file):
    """These tests should fail"""
    with pytest.raises(NameError):
        spec = HPFSpectrum(file="junk_file.txt")


@pytest.mark.parametrize(
    "file", local_files,
)
def test_equivalent_width(file):
    """Can we measure equivalent widths?"""

    spec = HPFSpectrum(file=file, order=4)
    mu = 8600  # make sure it is in given order
    equivalent_width = spec.measure_ew(mu)

    assert equivalent_width is not None
    assert type(equivalent_width) is not int
    assert type(equivalent_width) is astropy.units.quantity.Quantity
    assert equivalent_width.unit is spec.wavelength.unit


@pytest.mark.parametrize(
    "file", A0V_files,
)
def test_smoothing(file):
    """Does smoothing and outlier removal work?"""
    spec = HPFSpectrum(file=file, order=10)
    new_spec = spec.remove_outliers(threshold=3)

    assert len(new_spec.flux) > 0
    assert new_spec.shape[0] <= spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None


@pytest.mark.parametrize(
    "file", A0V_files,
)
def test_uncertainty(file):
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

    new_spec = spec.normalize().remove_nans()

    snr_vec = new_spec.flux / new_spec.uncertainty.quantity
    snr_med = np.nanmedian(snr_vec.value)
    assert np.isclose(snr_med, snr_old_med, atol=0.005)

    # Test SNR attribute
    assert np.all(new_spec.snr == snr_vec)

    new_spec = spec.normalize().deblaze()

    snr_vec = new_spec.flux / new_spec.uncertainty.array
    snr_med = np.nanmedian(snr_vec.value)
    assert np.isclose(snr_med, snr_old_med, atol=0.005)


def test_snr():
    """Test the Signal-to-noise ratio calculation"""
    n_pix = 10
    flux = np.ones(n_pix) * u.ct
    wave = np.linspace(1.0, 1.5, n_pix) * u.micron
    snr_per_pixel = 100.0
    unc_per_pixel = 1 / snr_per_pixel
    unc_vector = np.repeat(unc_per_pixel, n_pix) * u.ct
    unc = StdDevUncertainty(unc_vector)
    spec = HPFSpectrum(flux=flux, spectral_axis=wave, uncertainty=unc)

    snr_vec = spec.flux / spec.uncertainty.quantity

    assert isinstance(spec, HPFSpectrum)
    assert np.all(snr_vec == snr_per_pixel)
    assert np.all(spec.snr == snr_vec)
    assert hasattr(spec.snr, "unit")
    assert spec.snr.unit == u.dimensionless_unscaled


@pytest.mark.parametrize(
    "file", local_files,
)
def test_sky_and_lfc(file):
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


@pytest.mark.parametrize(
    "file", local_files,
)
def test_RV(file):
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


@pytest.mark.parametrize(
    "file", local_files,
)
def test_deblaze(file):
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


@pytest.mark.parametrize(
    "file", local_files,
)
def test_sky_subtraction(file):
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


@pytest.mark.parametrize(
    "file", local_files,
)
def test_HPF_spectrum_list(file):
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
