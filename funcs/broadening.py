from astropy.constants import k_B, m_p, c
import numpy as np
from astropy import units as u

def thermal_broadening_sigma(T, wavelength=6562.8):
    """
    Compute the thermal broadening of a line at a given temperature T.
    
    Parameters
    ----------
    T : float
        Temperature in K.
    wavelength : float
        Wavelength of the line in Angstroms.

    Returns
    -------
    float
        The Gaussian sigma of the thermal broadening in km/s.

    """


    sigmal = 2 * np.sqrt(2 * np.log(2) * k_B * T * u.K / m_p / (c**2)) * (wavelength * u.AA)
    sigmal = (sigmal).to("AA")
    # print(sigmal)

    # convert broadening to km/s
    broaden_v = c * sigmal / (6562.8 * u.AA)
    broaden_v = broaden_v.to("km/s")
    # print(broaden_v)

    # convert to gaussian sigma
    broaden_sigma = broaden_v / 2.355

    return broaden_sigma.value
