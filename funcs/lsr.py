import glob
import os
import re
import inspect
import numpy as np
from .modelfactory import SpectralModelFactory
from scipy.interpolate import interp1d
import astropy.units as u

PROT_LSR = 2.864 / 24.  # days
NAME_LSR = "LSR J1835"

# Grid resolution for the spherical model surface. Used both to run the
# UltraNest fits (run_forward_model_lsr.py) and to reproduce/visualise them
# (read_forward_model_results.ipynb) -- keep both pointed at this constant so
# they can never silently drift apart.
GRIDSIZE_LSR = 30000

# Historical default configuration -- every completed fit in results_lsr/
# was run at this (broaden, inclination), hardcoded inside setup_lsr_factory
# until this refactor. variant_name() special-cases exactly this pair so
# every existing results_lsr/*_ultranest folder and every already-registered
# model name keep resolving unchanged; only a non-default choice gets an
# explicit suffix.
DEFAULT_BROADEN = 9
DEFAULT_INCLINATION_DEG = 90


def setup_lsr_factory(gridsize=GRIDSIZE_LSR, broaden=DEFAULT_BROADEN,
                       inclination_deg=DEFAULT_INCLINATION_DEG, **kwargs):
    """Build a SpectralModelFactory for LSR J1835.

    Parameters
    ----------
    gridsize : int
        Number of points on the spherical model grid.
    broaden : float
        Gaussian broadening (km/s) applied to every spot/ring line profile.
    inclination_deg : float
        Stellar rotation-axis inclination in degrees (90 = equator-on).
    """

    vbins = np.linspace(-90, 90, 51)
    vmids = 0.5 * (vbins[1:] + vbins[:-1])
    gamma_kms = 10.5
    i_rot = np.pi/2 - inclination_deg / 180 * np.pi  # stellar inclination in radians

    # rotation period in days
    P_rot = PROT_LSR
    omega = 2 * np.pi / P_rot

    # stellar radius in solar radii
    R_star = (1.07 * u.R_jup).to(u.R_sun).value

    # maximum rot velocity of the star in km/s
    vmax = omega * R_star * 695700. / 86400. # km/s

    # velocity step size
    ddv = vmids[1] - vmids[0]

    alphas = np.linspace(0, 2 * np.pi, 21)[:-1]

    radius = 15 / 180 * np.pi
    ringwidth = 30 / 180 * np.pi


    spectral_model = SpectralModelFactory(
        vbins, vmids, broaden, i_rot, omega, vmax, R_star, ddv, alphas, radius, ringwidth, gamma_kms,
        registry_file='my_models.json', gridsize=gridsize, **kwargs)

    # Initialize once:
    return spectral_model, vmids, alphas


def get_lsr_data(path, vmids):
    pth = os.path.join(path, "lsr_0p*")

    file_list = glob.glob(pth)
    print(f"Found {len(file_list)} files in {path}")
    file_list.sort()

    data_list = []
    wavs = []
    for file in file_list:
        data = np.loadtxt(file)
        wav = data[:,0]
        flux = data[:,1]
        wavs.append(wav)
        data_list.append(np.array(flux))

    print(f"Number of files: {len(file_list)}")

    # wav to velocity
    c = 299792.458  # speed of light in km/s
    lambda_0 = 6562.8  # reference wavelength in Angstroms
    velocities = (wavs[0] - lambda_0) / lambda_0 * c

    # interpolate the observed fluxes to the model vmids
    interp_fluxes = []
    for flux, wav in zip(data_list, wavs):
        velocities = (wav - lambda_0) / lambda_0 * c
        f = interp1d(velocities, flux, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_fluxes.append(f(vmids))

    data = np.array(interp_fluxes)
    data = data[::-1]

    data_err = np.full_like(data, 0.)  # assuming constant error for simplicity

    errpath = "results_lsr/subtracted_spectra_errvals.txt"
    print(f"Reading error values from: {errpath}")
    # read the error values from the text file of this form
    with open(errpath, "r") as f:
        lines = f.readlines()
        errvals = []
        for line in lines:
            errval = float(line.split(":")[1].strip())
            errvals.append(errval)

    # propagate to data_err
    for i in range(len(data)):
        data_err[i,:] = errvals[i]

    # reverse order here as well to match the data
    data_err = data_err[::-1]

    return data, data_err


def ring_only(model, amplring, ringlat, ringwidth, i_mag, alpha0):
    return model.ring(i_mag, ringlat, ringwidth, alpha0, amplring)


def quiescent_background_one_spot(model, lon3, amplon1, lat1, amplback, width1):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon3, width1, amplon1),
        model.equatorial_ring(amplback)
    )


def quiescent_background_two_spots(model, lon1, lon2, amplon1, amplon2, amplback, lat1, lat2, width1, width2):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.equatorial_ring(amplback),
    )


def loose_ring_one_spot(model, lon3, amplon1, lat1, amplring, ringlat, i_mag, alpha0, width1, ringwidth2):
    return model.combine(
        model.spot(lat1, lon3, width1, amplon1),
        model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
    )


def loose_ring_two_spots(model, lon1, lon2, amplon1, amplon2, lat1, lat2, amplring, 
                         ringlat, i_mag, alpha0, ringwidth2, width1, width2):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
    )


def loose_ring_quiescent_background(model, amplring, ringlat, i_mag, alpha0, ringwidth2, amplback):
    model.ringwidth = np.pi
    return model.combine(model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
                         model.equatorial_ring(amplback))


def loose_ring_quiescent_background_one_spot(model, amplring, ringlat, i_mag, alpha0, ringwidth2, amplback,
                                             lon1, amplon1, lat1, width1):
    model.ringwidth = np.pi
    return model.combine(model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
                        model.spot(lat1, lon1, width1, amplon1),
                         model.equatorial_ring(amplback))


def three_spots(model, lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3):
    # model.width1 = np.pi/4
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
    )


def quiescent_background_three_spots(model, lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3, amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.equatorial_ring(amplback)
    )


def two_rings(model, amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth2, ringwidth1):
    return model.combine(
        model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth1, alpha0, amplring2)
    )


def two_rings_one_spot(model, amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                     lon1, amplon1, lat1, width1):
    return model.combine(
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
        model.spot(lat1, lon1, width1, amplon1)
    )


def two_rings_one_spot_quiescent_background(model, amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                     lon1, amplon1, lat1, width1, amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
        model.spot(lat1, lon1, width1, amplon1),
        model.equatorial_ring(amplback)
    )


def two_rings_quiescent_background(model, amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2, amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
        model.equatorial_ring(amplback)
    )


def loose_ring_quiescent_background_two_spots(model, amplring, ringlat, i_mag, alpha0, ringwidth2, amplback,
                                             lon1, amplon1, lat1, width1,
                                             lon2, amplon2, lat2, width2):
    model.ringwidth = np.pi
    return model.combine(model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
                        model.spot(lat1, lon1, width1, amplon1),
                        model.spot(lat2, lon2, width2, amplon2),
                         model.equatorial_ring(amplback))


def two_rings_two_spots(model, amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                     lon1, amplon1, lat1, width1,
                     lon2, amplon2, lat2, width2):
    return model.combine(
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2)
    )


def four_spots(model, lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4)
    )


def three_spots_one_ring(model, lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
    )


def four_spots_one_ring(model, lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                   amplring, ringlat, i_mag, alpha0, ringwidth2):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
    )


def three_spots_two_rings(model, lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                       amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
    )


def four_spots_two_rings(model, lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
                                          lat1, lat2, lat3, lat4,
                                          width1, width2, width3, width4,
                                          amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
    )


def five_spots_two_rings(model, lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5, lat1, lat2, lat3, lat4, lat5, width1, width2, width3, width4, width5,
                       amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
    )


def three_spots_one_ring_quiescent_background(model, lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                       amplring, ringlat, i_mag, alpha0, ringwidth2, amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
        model.equatorial_ring(amplback)
    )


def two_spots_two_rings_quiescent_background(model, lon1, lon2, amplon1, amplon2, lat1, lat2, width1, width2,
                                             amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                                             amplback):
    
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
        model.equatorial_ring(amplback)
    )


def four_spots_quiescent_background(model, lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                   amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.equatorial_ring(amplback)
    )


def five_spots(model, lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5, lat1, lat2, lat3, lat4, lat5, width1, width2, width3, width4, width5):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5)
    )


def six_spots(model, lon1, lon2, lon3, lon4, lon5, lon6, amplon1, amplon2, amplon3, amplon4, amplon5, amplon6,
           lat1, lat2, lat3, lat4, lat5, lat6,
           width1, width2, width3, width4, width5, width6):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5),
        model.spot(lat6, lon6, width6, amplon6)
         )


def five_spots_quiescent_background(model, lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
                     lat1, lat2, lat3, lat4, lat5,
                     width1, width2, width3, width4, width5,
                     amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5),
        model.equatorial_ring(amplback)
    )


def six_spots_ring(model, lon1, lon2, lon3, lon4, lon5, lon6, amplon1, amplon2, amplon3, amplon4, amplon5, amplon6,
           lat1, lat2, lat3, lat4, lat5, lat6,
           width1, width2, width3, width4, width5, width6,
           amplring1, ringlat, i_mag, alpha0, ringwidth1):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5),
        model.spot(lat6, lon6, width6, amplon6),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1)
    )


def five_spots_ring(model, lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
                     lat1, lat2, lat3, lat4, lat5,
                     width1, width2, width3, width4, width5,
                     amplring1, ringlat, i_mag, alpha0, ringwidth1):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1)
    )


def four_spots_one_ring_quiescent_background(model, lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
                                          lat1, lat2, lat3, lat4,
                                          width1, width2, width3, width4,
                                          amplring1, ringlat, i_mag, alpha0, ringwidth1,
                                          amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.equatorial_ring(amplback)
    )


def three_spots_two_rings_quiescent_background(model, lon1, lon2, lon3, amplon1, amplon2, amplon3,
                                              lat1, lat2, lat3,
                                              width1, width2, width3,
                                              amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                                              amplback):
    model.ringwidth = np.pi
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
        model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
        model.equatorial_ring(amplback)
    )


def seven_spots(model, lon1, lon2, lon3, lon4, lon5, lon6, lon7,
               amplon1, amplon2, amplon3, amplon4, amplon5, amplon6, amplon7,
               lat1, lat2, lat3, lat4, lat5, lat6, lat7,
               width1, width2, width3, width4, width5, width6, width7):
    return model.combine(
        model.spot(lat1, lon1, width1, amplon1),
        model.spot(lat2, lon2, width2, amplon2),
        model.spot(lat3, lon3, width3, amplon3),
        model.spot(lat4, lon4, width4, amplon4),
        model.spot(lat5, lon5, width5, amplon5),
        model.spot(lat6, lon6, width6, amplon6),
        model.spot(lat7, lon7, width7, amplon7)
    )


MODEL_RECIPES = {
    'ring_only': ring_only,
    'quiescent_background_one_spot': quiescent_background_one_spot,
    'quiescent_background_two_spots': quiescent_background_two_spots,
    'loose_ring_one_spot': loose_ring_one_spot,
    'loose_ring_two_spots': loose_ring_two_spots,
    'loose_ring_quiescent_background': loose_ring_quiescent_background,
    'loose_ring_quiescent_background_one_spot': loose_ring_quiescent_background_one_spot,
    'three_spots': three_spots,
    'quiescent_background_three_spots': quiescent_background_three_spots,
    'two_rings': two_rings,
    'two_rings_one_spot': two_rings_one_spot,
    'two_rings_one_spot_quiescent_background': two_rings_one_spot_quiescent_background,
    'two_rings_quiescent_background': two_rings_quiescent_background,
    'loose_ring_quiescent_background_two_spots': loose_ring_quiescent_background_two_spots,
    'two_rings_two_spots': two_rings_two_spots,
    'four_spots': four_spots,
    'three_spots_one_ring': three_spots_one_ring,
    'four_spots_one_ring': four_spots_one_ring,
    'three_spots_two_rings': three_spots_two_rings,
    'four_spots_two_rings': four_spots_two_rings,
    'five_spots_two_rings': five_spots_two_rings,
    'three_spots_one_ring_quiescent_background': three_spots_one_ring_quiescent_background,
    'two_spots_two_rings_quiescent_background': two_spots_two_rings_quiescent_background,
    'four_spots_quiescent_background': four_spots_quiescent_background,
    'five_spots': five_spots,
    'six_spots': six_spots,
    'five_spots_quiescent_background': five_spots_quiescent_background,
    'six_spots_ring': six_spots_ring,
    'five_spots_ring': five_spots_ring,
    'four_spots_one_ring_quiescent_background': four_spots_one_ring_quiescent_background,
    'three_spots_two_rings_quiescent_background': three_spots_two_rings_quiescent_background,
    'seven_spots': seven_spots,
}


MODEL_DISPLAY_NAMES = {
    'ring_only': 'Ring',
    'quiescent_background_one_spot': '1 Spot + Q. bkg.',
    'quiescent_background_two_spots': '2 Spots + Q. bkg.',
    'loose_ring_one_spot': 'Ring + 1 Spot',
    'loose_ring_two_spots': 'Ring + 2 Spots',
    'loose_ring_quiescent_background': 'Ring + Q. bkg.',
    'loose_ring_quiescent_background_one_spot': 'Ring + 1 Spot +Q. bkg.',
    'three_spots': '3 Spots',
    'quiescent_background_three_spots': '3 Spots + Q. bkg.',
    'two_rings': '2 Rings',
    'two_rings_one_spot': '2 Rings + 1 Spot',
    'two_rings_one_spot_quiescent_background': '2 Rings + 1 Spot + Q. bkg.',
    'two_rings_quiescent_background': '2 Rings + Q. bkg.',
    'loose_ring_quiescent_background_two_spots': 'Ring + 2 Spots + Q. bkg.',
    'two_rings_two_spots': '2 Rings + 2 Spots',
    'four_spots': '4 Spots',
    'three_spots_one_ring': 'Ring + 3 Spots',
    'four_spots_one_ring': 'Ring + 4 Spots',
    'three_spots_two_rings': '2 Rings + 3 Spots',
    'four_spots_two_rings': '2 Rings + 4 Spots',
    'five_spots_two_rings': '2 Rings + 5 Spots',
    'three_spots_one_ring_quiescent_background': 'Ring + 3 Spots + Q. bkg.',
    'two_spots_two_rings_quiescent_background': '2 Rings + 2 Spots + Q. bkg.',
    'four_spots_quiescent_background': '4 Spots + Q. bkg.',
    'five_spots': '5 Spots',
    'six_spots': '6 Spots',
    'five_spots_quiescent_background': '5 Spots + Q. bkg.',
    'six_spots_ring': 'Ring + 6 Spots',
    'five_spots_ring': 'Ring + 5 Spots',
    'four_spots_one_ring_quiescent_background': 'Ring + 4 Spots + Q. bkg.',
    'three_spots_two_rings_quiescent_background': '2 Rings + 3 Spots + Q. bkg.',
    'seven_spots': '7 Spots',
}


def variant_name(base_name, broaden=DEFAULT_BROADEN, inclination_deg=DEFAULT_INCLINATION_DEG):
    """Name for one (base_name, broaden, inclination) combination.

    Matches the bare `base_name` exactly at the historical default
    (broaden=9, inclination=90 deg), so every existing results_lsr/*_ultranest
    folder and every name read_forward_model_results.ipynb already expects
    keep resolving unchanged. Any other (broaden, inclination_deg) gets an
    explicit suffix, both for the registered model's own name and for the
    results folder run_forward_model_lsr.py saves it under.
    """
    if broaden == DEFAULT_BROADEN and inclination_deg == DEFAULT_INCLINATION_DEG:
        return base_name
    return f"{base_name}_broaden{broaden:g}_inc{inclination_deg:g}"


def _bind_recipe(recipe, model, name):
    """Turn a `def recipe(model, <physical params>)` function into a plain
    `def name(<physical params>)` callable bound to this specific `model`
    (i.e. this specific broaden/inclination configuration), suitable for
    `model.register(...)`.

    Sets `__name__` and `__signature__` explicitly (dropping the leading
    `model` parameter) so that everything downstream that introspects the
    registered function -- SpectralModelFactory.get_param_names/get_bounds/
    create_ultranest_prior/create_ultranest_likelihood, and
    run_forward_model_lsr.py's own `mmodel.__name__` -- sees exactly what it
    would have for a hand-written, single-purpose model function.
    """
    def _variant(*args, **kwargs):
        return recipe(model, *args, **kwargs)

    _variant.__name__ = name
    _variant.__qualname__ = name
    _variant.__doc__ = recipe.__doc__
    sig = inspect.signature(recipe)
    physical_params = list(sig.parameters.values())[1:]  # drop `model`
    _variant.__signature__ = sig.replace(parameters=physical_params)
    return _variant


# Before broaden/inclination became explicit parameters (see variant_name
# above), they were baked into the factory, and distinct broaden/inclination
# choices got their own hand-duplicated model function with the value
# spelled out in the name -- e.g. "four_spots_one_ring_4broaden",
# "five_spots_medium_low_inc". Those names are gone from MODEL_RECIPES (the
# duplicates were identical in structure to their base model, see the
# module docstring/history), but they're still the folder names of already
# -completed fits under results_lsr/*_ultranest, so lookups by that name
# need to keep resolving -- and resolve to a model that actually runs at
# the (broaden, inclination) that fit used, not just one with matching
# parameter names.
#
# "_<N>broaden" spells its value out literally (parsed below). The
# inclination words don't: this is the actual degree each one meant,
# recovered from the display-name strings the pre-refactor lsr.py carried
# alongside each duplicate function (see git history, e.g. "Ring + 4 Spots
# at 75 deg inc." for what was then "four_spots_one_ring_medium_low_inc").
_LEGACY_INCLINATION_DEG = {
    "low_inc": 70,
    "medium_low_inc": 75,
    "medium_low": 75,  # four_spots_medium_low used this spelling for the same 75 deg
    "medium_inc": 80,
}

_LEGACY_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)_(?:(?P<inc_word>medium_low_inc|medium_inc|low_inc|medium_low)"
    r"|(?P<broaden_num>\d+)broaden)$"
)

# The current naming scheme (variant_name, above): base_name +
# "_broaden{b}_inc{i}", both numeric and both spelled with `:g` formatting.
_VARIANT_NAME_RE = re.compile(
    r"^(?P<base>.+?)_broaden(?P<broaden>[0-9.]+)_inc(?P<inc>[0-9.]+)$"
)


def _variant_config_from_name(name):
    """If `name` names one (base model, broaden, inclination) combination --
    either the current numeric scheme (variant_name) or a historical
    hand-duplicated name -- return (base_name, broaden, inclination_deg) for
    the configuration it actually refers to; otherwise return None.

    This is what lets `register_variants_from_results` register *any*
    results_lsr/*_ultranest folder name on demand: the folder name alone is
    enough to recover the exact configuration to rebuild, current or
    historical.
    """
    match = _VARIANT_NAME_RE.match(name)
    if match and match.group("base") in MODEL_RECIPES:
        return match.group("base"), float(match.group("broaden")), float(match.group("inc"))

    match = _LEGACY_SUFFIX_RE.match(name)
    if not match or match.group("base") not in MODEL_RECIPES:
        return None
    base_name = match.group("base")
    if match.group("broaden_num") is not None:
        return base_name, float(match.group("broaden_num")), DEFAULT_INCLINATION_DEG
    return base_name, DEFAULT_BROADEN, _LEGACY_INCLINATION_DEG[match.group("inc_word")]


def variant_display_name(name):
    """Pretty label for `name` -- a base model structure or any (current or
    historical) broaden/inclination variant of one -- or None if `name`
    isn't recognized.

    Pure string lookup: does not need a model registered or built, so it's
    cheap to call for every results_lsr/*_ultranest folder found on disk
    (see e.g. read_forward_model_results.ipynb's map_names).
    """
    if name in MODEL_RECIPES:
        return MODEL_DISPLAY_NAMES[name]
    config = _variant_config_from_name(name)
    if config is None:
        return None
    base_name, broaden, inclination_deg = config
    pretty_name = MODEL_DISPLAY_NAMES[base_name]
    return f"{pretty_name} (broaden={broaden:g} km/s, inc={inclination_deg:g} deg)"


def register_variants_from_results(model, results_dirs=("results_lsr",)):
    """Scan `results_dirs` for `<name>_ultranest` folders and register every
    `<name>` not already on `model` (see `_variant_config_from_name`),
    running each at the exact (broaden, inclination) it actually names --
    not `model`'s own configuration -- whether that name is the current
    numeric scheme or a historical hand-duplicated one.

    This is what makes `model.get_model(<folder name>)` "just work" for any
    completed fit without having to separately know or pass the broaden/
    inclination it used: read it straight off the folder name.

    `model` must already have the base model structures registered at the
    default broaden/inclination (see register_lsr_models). Every variant
    factory this builds mirrors `model`'s own `obj_only`/`foreshortening`
    (read off `model._common_kwargs`), so e.g. a caller using
    `obj_only=True` (for sphere plotting, .ring()/.spot()/.background()
    returning AuroralRing objects rather than raw arrays) gets that
    consistently for every variant too, not silently back to
    SpectralModelFactory's own default.
    """
    # The two behaviour toggles SpectralModelFactory accepts as **kwargs
    # (as opposed to i_rot/omega/vmax/R_star/ddv/alphas/THETA/PHI, always
    # set from setup_lsr_factory's own positional args -- forwarding those
    # here as **kwargs too would collide with them).
    extra_kwargs = {
        k: model._common_kwargs[k]
        for k in ("obj_only", "foreshortening")
        if k in model._common_kwargs
    }

    to_register = {}
    for results_dir in results_dirs:
        if not os.path.isdir(results_dir):
            continue
        for entry in os.listdir(results_dir):
            if not entry.endswith("_ultranest"):
                continue
            name = entry[: -len("_ultranest")]
            if model.is_registered(name):
                continue
            config = _variant_config_from_name(name)
            if config is not None:
                to_register[name] = config

    # Group by (broaden, inclination_deg) so each distinct configuration
    # only costs one extra factory, however many names/model structures
    # share it.
    by_config = {}
    for name, (base_name, broaden, inclination_deg) in to_register.items():
        by_config.setdefault((broaden, inclination_deg), []).append((name, base_name))

    registered = []
    for (broaden, inclination_deg), entries in by_config.items():
        variant_model, _, _ = setup_lsr_factory(
            gridsize=model.gridsize, broaden=broaden, inclination_deg=inclination_deg,
            **extra_kwargs)
        for name, base_name in entries:
            func = _bind_recipe(MODEL_RECIPES[base_name], variant_model, name)
            variant_model.register(func, name=name)
            model.alias(name, name, source=variant_model)
            registered.append(name)
    return registered


def register_lsr_models(model, broaden=DEFAULT_BROADEN, inclination_deg=DEFAULT_INCLINATION_DEG):
    """Register every LSR model structure on `model`, named/labelled for the
    given (broaden, inclination_deg).

    `model` itself must already be configured with this exact (broaden,
    inclination_deg) -- e.g. via
    `setup_lsr_factory(broaden=broaden, inclination_deg=inclination_deg)` --
    this only needs to know that configuration to name and label things
    consistently with it (see `variant_name`); it does not itself change
    `model`'s broadening or inclination.

    Also registers every other configuration named by an existing
    results_lsr/*_ultranest folder (see `register_variants_from_results`),
    so afterwards `model.get_model(<any completed fit's folder name>)`
    resolves regardless of what (broaden, inclination_deg) was passed here
    -- the folder name alone carries what it needs.

    Returns
    -------
    (functions, pretty_names) : two lists, positionally aligned, matching
        the original register_lsr_models(model) -> (funcs, names) contract.
    """
    funcs, names = [], []
    for base_name, recipe in MODEL_RECIPES.items():
        name = variant_name(base_name, broaden, inclination_deg)
        func = _bind_recipe(recipe, model, name)
        model.register(func, name=name)
        funcs.append(func)

        pretty_name = MODEL_DISPLAY_NAMES[base_name]
        if name != base_name:
            pretty_name = f"{pretty_name} (broaden={broaden:g} km/s, inc={inclination_deg:g} deg)"
        names.append(pretty_name)

    register_variants_from_results(model)

    return funcs, names
