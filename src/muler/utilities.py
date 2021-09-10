import numpy as np
from astropy.nddata.nduncertainty import StdDevUncertainty


def combined_spectra(spec_list):
    spec_final = spec_list[0]
    for i in range(len(spec_list) - 1):
        if i != len(spec_list) - 1:
            new_sigma = np.sqrt(1 / spec_list[i + 1].uncertainty.array)
            spec_list[i + 1].uncertainty = StdDevUncertainty(new_sigma)
            spec_final = spec_final.add(spec_list[i + 1], propagate_uncertainties=True)
    return spec_final

