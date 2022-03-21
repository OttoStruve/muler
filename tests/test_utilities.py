import pytest
import astropy
import time
from muler.hpf import HPFSpectrum
from muler.igrins import IGRINSSpectrumList
from specutils import Spectrum1D, SpectrumList
import glob
from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np


from muler.utilities import combine_spectra, apply_numpy_mask, concatenate_orders

# There should be exactly 3 files in the example data directory
local_files = glob.glob("**/01_A0V_standards/Goldilocks_*.spectra.fits", recursive=True)
igrins_H_file = glob.glob(
    "**/IGRINS/01_IGRINS_test_data/SDCH*.spec_a0v.fits", recursive=True
)[0]
igrins_K_file = glob.glob(
    "**/IGRINS/01_IGRINS_test_data/SDCK*.spec_a0v.fits", recursive=True
)[0]


def test_concatenate_orderss():
    """Does the combine utility work with a list of spectra?"""
    n_files = len(local_files)
    spec_list1 = IGRINSSpectrumList.read(file=igrins_H_file)
    spec_list2 = IGRINSSpectrumList.read(file=igrins_K_file)

    ## Currently the source code as written will break with this kwarg
    median_flux = np.nanmedian(spec_list1[23].flux.value)
    with pytest.raises(TypeError):
        output = spec_list1.normalize(median_flux=median_flux)

    ## spec_list normalization will work with no kwarg though:
    output = spec_list1.normalize()

    assert output is not None
    assert type(output) == type(spec_list1)

    with pytest.raises(astropy.units.core.UnitConversionError):
        full_H_and_K_list = spec_list1 + spec_list2

    full_H_and_K_list = concatenate_orders(spec_list1, spec_list2)

    assert isinstance(full_H_and_K_list, SpectrumList)
    assert isinstance(full_H_and_K_list, IGRINSSpectrumList)
    assert len(full_H_and_K_list) == (len(spec_list1) + len(spec_list2))


def test_apply_mask():
    """Does applying a boolean mask work?"""
    spec = HPFSpectrum(file=local_files[0], order=10)

    assert spec is not None

    # A mask with all ones should keep the same shape
    mask = np.ones_like(spec.flux.value, dtype=bool)
    spec_out = apply_numpy_mask(spec, mask)
    assert len(spec_out.flux) == len(spec.flux)

    # A mask with all zeros should raise an error
    mask = np.zeros_like(spec.flux.value, dtype=bool)
    with pytest.raises(AssertionError):
        spec_out = apply_numpy_mask(spec, mask)

    # A regular mask should work
    mask[0:5] = True
    spec_out = apply_numpy_mask(spec, mask)
    assert len(spec_out.flux) < len(spec.flux)
    assert len(spec_out.flux) == 5

    assert hasattr(spec_out, "meta")
    assert hasattr(spec_out, "mask")


def test_combine():
    """Does the combine utility work?"""
    n_files = len(local_files)
    this_spec_list = [
        HPFSpectrum(file=local_files[i], order=10) for i in range(n_files)
    ]

    assert len(this_spec_list) == n_files
    assert n_files == 3
    assert this_spec_list is not None

    coadded_spectrum = combine_spectra(this_spec_list)

    assert isinstance(coadded_spectrum, Spectrum1D)
    assert isinstance(coadded_spectrum, HPFSpectrum)
    assert len(coadded_spectrum.flux) == len(coadded_spectrum.uncertainty.array)

    ## Does the signal-to-noise ratio scale the way we'd expect?
    # Make a single fake HPF spectrum
    reference_spectrum = this_spec_list[0]
    n_pixels = len(reference_spectrum.wavelength)

    # SNR = 100
    input_sigma = 0.01
    input_mean = 1.0
    input_snr = input_mean / input_sigma
    uncertainty_vector = np.repeat(input_sigma, repeats=n_pixels)
    flux_vector = np.ones_like(uncertainty_vector)

    fake_spectrum = HPFSpectrum(
        spectral_axis=reference_spectrum.wavelength,
        flux=flux_vector * reference_spectrum.flux.unit,
        uncertainty=StdDevUncertainty(uncertainty_vector),
    )

    assert isinstance(fake_spectrum, HPFSpectrum)

    # Combining 4 spectra should double the SNR: sqrt(4) = 2
    fake_spectrum_list = [fake_spectrum] * 4
    net_spectrum = combine_spectra(fake_spectrum_list)

    net_snr = np.mean(net_spectrum.flux.value / net_spectrum.uncertainty.array)
    assert net_snr == (input_snr * 2)

    # Test combining several copies of that spectrum
    n_spectra_tests = [9, 16, 25, 36, 100, 1000, 2, 1]

    for n_spectra in n_spectra_tests:
        fake_spectrum_list = [fake_spectrum] * n_spectra

        t0 = time.time()
        net_spectrum = combine_spectra(fake_spectrum_list)
        t1 = time.time()
        net_time = t1 - t0
        print(
            f"\n\t Net time for combining {n_spectra} spectra: {net_time:0.5f} seconds",
            end="\t",
        )

        # The new SNR should be twice as low as before:
        assert np.allclose(
            np.mean(fake_spectrum.uncertainty.array) * np.sqrt(n_spectra),
            np.mean(net_spectrum.uncertainty.array),
        ), "Coadding {} spectra should root-N down the uncertainty to {}".format(
            n_spectra, np.sqrt(n_spectra)
        )
