# from astropy.nddata.ccddata import _uncertainty_unit_equivalent_to_parent
import pytest
import time
from muler.nirspec import KeckNIRSPECSpectrum, KeckNIRSPECSpectrumList
from specutils import Spectrum1D

# from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np
import glob
import astropy
from specutils.spectra.spectrum_list import SpectrumList

local_files = glob.glob("**/NS.*_flux_tbl.fits*", recursive=True)
file = local_files[0]


def test_basic():
    """Do the basic methods work?"""

    spec = KeckNIRSPECSpectrum(file=file)

    assert spec is not None
    assert isinstance(spec, Spectrum1D)
    assert isinstance(spec.flux, np.ndarray)
    assert len(spec.flux) == len(spec.wavelength)
    assert spec.mask.sum() >= 0

    new_spec = spec.remove_nans()

    assert new_spec.shape[0] <= spec.shape[0]
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

    spec = KeckNIRSPECSpectrum(file=file).remove_nans().trim_edges()
    mu = np.nanmedian(spec.wavelength.value)

    assert spec.mask.dtype == bool

    equivalent_width = spec.measure_ew(mu)

    assert equivalent_width is not None
    assert type(equivalent_width) is not int
    assert type(equivalent_width) is astropy.units.quantity.Quantity
    assert equivalent_width.unit is spec.wavelength.unit


def test_smoothing():
    """Does smoothing and outlier removal work?"""
    spec = KeckNIRSPECSpectrum(file=file)
    new_spec = spec.remove_outliers(threshold=3)

    assert len(new_spec.flux) > 0
    assert new_spec.shape[0] <= spec.shape[0]
    assert new_spec.shape[0] > 0
    assert new_spec.mask is not None


def test_uncertainty():
    """Does uncertainty propagation work?"""

    spec = KeckNIRSPECSpectrum(file=file)

    assert spec.flux is not None

    new_spec = spec.remove_nans().deblaze()

    assert len(new_spec.flux) == len(new_spec.uncertainty.array)
    assert np.all(new_spec.uncertainty.array > 0)

    snr_old_vec = spec.flux / spec.uncertainty.array
    snr_old_med = np.nanmedian(snr_old_vec.value)

    new_spec = spec.normalize()

    snr_vec = new_spec.flux / new_spec.uncertainty.array
    snr_med = np.nanmedian(snr_vec.value)
    assert np.isclose(snr_med, snr_old_med, atol=0.005)


def test_RV():
    """Does RV shifting work"""

    spec = KeckNIRSPECSpectrum(file=file)

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

    spec = KeckNIRSPECSpectrum(file=file)

    new_spec = spec.remove_nans().deblaze()

    assert new_spec is not None
    assert isinstance(new_spec, Spectrum1D)


def test_lists():
    """Does uncertainty propagation work?"""

    speclist = KeckNIRSPECSpectrumList.read(files=local_files)

    new_speclist = speclist.remove_nans().normalize()

    assert new_speclist is not None
    assert isinstance(new_speclist, SpectrumList)
