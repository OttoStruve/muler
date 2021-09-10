import numpy as np
from astropy.nddata.nduncertainty import StdDevUncertainty


def combine_spectra(spec_list):
    spec_final = spec_list[0]
    for i in range(1, len(spec_list)):
        spec_final = spec_final.add(spec_list[i], propagate_uncertainties=True)
    return spec_final

