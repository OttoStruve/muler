import pytest
import time
from muler.hpf import HPFSpectrum
from specutils import Spectrum1D
import glob
from astropy.nddata.nduncertainty import StdDevUncertainty
import numpy as np

from muler.utilities import combine_spectra, apply_numpy_mask

# There should be exactly 3 files in the example data directory
local_files = glob.glob("**/01_A0V_standards/Goldilocks_*.spectra.fits", recursive=True)


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
