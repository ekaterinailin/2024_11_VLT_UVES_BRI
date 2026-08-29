#!/usr/bin/env python3
"""
Posterior on the stellar inclination i* of LSR J1835, from v*sin(i*).

Combines each literature v*sin(i*) measurement with the equatorial velocity
(from the adopted radius and rotation period, each with its own uncertainty)
into a posterior on i*:

    P(i*|d) ~ sin(i*) * exp(-(v*sin(i*) - v_eq*sin(i*))^2 / (2*var)) / sqrt(var)
    var = sigma_v_sin_i^2 + sigma_v_eq^2 * sin^2(i*)

Plots one curve per v*sin(i*) source plus the combined estimate, saves the
figure to results_lsr/, and prints the peak-posterior inclination per source.

Usage
-----
    python 6_posterior_inclination.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

RESULTS_DIR = "results_lsr"


def posterior_inclination(i_star, v_sin_i, R_star, P_rot, sigma_v_sin_i, sigma_v_eq):
    """
    Computes the unnormalized posterior probability of stellar inclination i*.

    P(i* | d) ~ sin(i*) * exp(-(...**2) / (2*(sig2_v_sin_i + sig2_v_eq*sin^2(i*))))
                          / sqrt(sig2_v_sin_i + sig2_v_eq*sin^2(i*))

    Parameters
    ----------
    i_star        : float or array  Inclination angle [radians]
    v_sin_i       : float           Observed projected velocity v*sin(i*) [km/s]
    R_star        : float           Stellar radius [km or consistent units]
    P_rot         : float           Rotation period [s or consistent units]
    sigma_v_sin_i : float           Uncertainty on v*sin(i*) [km/s]
    sigma_v_eq    : float           Uncertainty on equatorial velocity v_eq [km/s]

    Returns
    -------
    float or array : Unnormalized posterior probability
    """
    sin_i = np.sin(i_star)
    v_eq = (2 * np.pi * R_star) / P_rot
    variance = sigma_v_sin_i**2 + sigma_v_eq**2 * sin_i**2
    delta = v_sin_i - v_eq * sin_i
    exponent = -delta**2 / (2 * variance)
    return sin_i * np.exp(exponent) / np.sqrt(variance)


def main():
    i_star = np.linspace(0, np.pi / 2, 100)
    R_star = 1.07 * u.R_jup
    P_rot = 2.845 * u.h
    sigma_R_star = 0.05 * u.R_jup
    sigma_veq = (2 * np.pi * sigma_R_star / P_rot).to(u.km / u.s)

    literature = [
        (43.9, 2.2, "Crossfield+2014", "steelblue"),
        (50.0, 5.0, "Berger+2008", "navy"),
        (49.2, 4.9, "Reiners+2018", "peru"),
        (43.7, 3.5, "Varas+2026", "orange"),
        (46.7, 2.03, "Combined", "k"),
    ]

    plt.figure()
    peak_i = {}
    for v_sin_i, sigma_v_sin_i, label, color in literature:
        posterior = posterior_inclination(
            i_star, v_sin_i, R_star.to(u.km).value, P_rot.to(u.s).value,
            sigma_v_sin_i, sigma_veq.value,
        )
        posterior /= np.trapezoid(posterior, i_star)
        peak_i[label] = np.degrees(i_star[np.argmax(posterior)])
        lw = 3 if label == "Combined" else 1.5
        plt.plot(np.degrees(i_star), posterior, label=label, color=color, lw=lw)

    plt.xlabel('Inclination [deg]')
    plt.xlim(0, 90)
    plt.ylim(0, None)
    plt.ylabel('Posterior density')
    plt.legend(frameon=False)

    out_path = os.path.join(RESULTS_DIR, "inclination_posterior.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")

    print("Peak-posterior inclination per v*sin(i*) source:")
    for label, i_peak in peak_i.items():
        print(f"  {label:16} i_peak = {i_peak:5.1f} deg")


if __name__ == "__main__":
    main()
