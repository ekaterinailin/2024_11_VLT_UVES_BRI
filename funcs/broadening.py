from astropy.constants import k_B, m_p, c
import numpy as np
from astropy import units as u


from scipy.ndimage import convolve1d
from scipy.special import voigt_profile

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





def voigt_kernel_kms(dv, sigma_kms, gamma_kms, kernel_width_kms=None):
    """
    Build a normalized Voigt kernel in velocity space.

    Parameters
    ----------
    dv : float
        Velocity pixel spacing in km/s.
    sigma_kms : float
        Gaussian standard deviation in km/s.
        Relate to FWHM via: sigma = fwhm / (2 * sqrt(2 * ln2))
    gamma_kms : float
        Lorentzian HWHM in km/s.
    kernel_width_kms : float, optional
        Full width of the kernel in km/s. Defaults to a width that
        covers the broader component generously.

    Returns
    -------
    kernel : ndarray
        Normalized Voigt kernel.
    """
    if kernel_width_kms is None:
        half_extent = max(10 * gamma_kms, 5 * sigma_kms)
        kernel_width_kms = 2 * half_extent

    # Round to odd number of pixels
    n_pixels = int(np.round(kernel_width_kms / dv))
    if n_pixels % 2 == 0:
        n_pixels += 1
    n_pixels = max(n_pixels, 3)

    center = n_pixels // 2
    x = (np.arange(n_pixels) - center) * dv   # velocity axis in km/s

    kernel = voigt_profile(x, sigma_kms, gamma_kms)
    return kernel / kernel.sum()


def wav_to_vel(wave, wave0):
    """
    Convert a wavelength array to velocity in km/s relative to a rest wavelength.

    Uses the non-relativistic Doppler approximation (valid for v << c).
    For M dwarf chromospheric work this is fine; for >~0.01c use the
    relativistic form.

    Parameters
    ----------
    wave : array_like
        Wavelength array in Angstroms (or any consistent unit).
    wave0 : float
        Rest wavelength of the line in the same units.
        e.g. H-alpha: 6562.8 Ang

    Returns
    -------
    vel : ndarray
        Velocity array in km/s.
    """
    C_KMS = 2.99792458e5   # speed of light in km/s
    return ((np.asarray(wave) - wave0) / wave0) * C_KMS


def vel_to_wav(vel, wave0):
    """Inverse of wav_to_vel."""
    C_KMS = 2.99792458e5
    return wave0 * (1.0 + np.asarray(vel) / C_KMS)


def thermal_sigma_kms(T_K, mass_amu=1.008):
    """
    Gaussian sigma (std dev) of thermal broadening in km/s.

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.
    mass_amu : float
        Atomic mass of the absorber in amu. Default is hydrogen (1.008).

    Returns
    -------
    sigma_kms : float
        Thermal velocity dispersion (sigma, not FWHM) in km/s.
    """
    k_B  = 1.380649e-23   # J/K
    m_H  = 1.6735575e-27  # kg per amu
    sigma_ms = np.sqrt(k_B * T_K / (mass_amu * m_H))
    return sigma_ms / 1e3  # convert m/s -> km/s


def convolve_voigt_kms(vel, flux, sigma_kms, gamma_kms,
                       kernel_width_kms=None):
    """
    Convolve a spectrum (in velocity space) with a Voigt profile.

    Parameters
    ----------
    vel : array_like
        Velocity axis in km/s, uniform spacing required.
    flux : array_like
        1D flux array corresponding to vel.
    sigma_kms : float
        Gaussian sigma (std dev) in km/s.
        To include microturbulence xi (km/s), combine in quadrature:
            sigma_total = sqrt(sigma_thermal**2 + xi**2)
    gamma_kms : float
        Lorentzian HWHM in km/s (pressure broadening component).
        Convert from wavelength units via:
            gamma_kms = (gamma_aa / wave0_aa) * C_KMS
    kernel_width_kms : float, optional
        Full kernel extent in km/s. Default is automatic.

    Returns
    -------
    flux_conv : ndarray
        Convolved flux, same shape as input.
    """
    vel  = np.asarray(vel,  dtype=float)
    flux = np.asarray(flux, dtype=float)

    dv = vel[1] - vel[0]
    if not np.allclose(np.diff(vel), dv, rtol=1e-4):
        raise ValueError("vel axis must be uniformly spaced. "
                         "Resample before calling this function.")

    kernel = voigt_kernel_kms(dv, sigma_kms, gamma_kms, kernel_width_kms)
    return convolve1d(flux, kernel, mode='nearest')