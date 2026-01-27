import glob
import os
import numpy as np
from .modelfactory import SpectralModelFactory
from scipy.interpolate import interp1d
import astropy.units as u


def setup_lsr_factory(**kwargs):

    vbins = np.linspace(-90, 90, 101)
    vmids = 0.5 * (vbins[1:] + vbins[:-1])
    broaden = 18
    i_rot = np.pi/2 - 90 / 180 * np.pi  # stellar inclination in radians

    # rotation period in days
    P_rot = 2.864 / 24.
    omega = 2 * np.pi / P_rot

    # stellar radius in solar radii
    R_star = (1.07 * u.R_jup).to(u.R_sun).value

    # maximum rot velocity of the star in km/s
    vmax = omega * R_star * 695700. / 86400. # km/s

    # velocity step size
    ddv = vmids[1] - vmids[0]

    alphas = np.linspace(0, 2 * np.pi, 20)

    radius = 15 / 180 * np.pi
    ringwidth = 30 / 180 * np.pi


    # Initialize once:
    return SpectralModelFactory(
        vbins, vmids, broaden, i_rot, omega, vmax, R_star, ddv, alphas, radius, ringwidth,
        registry_file='my_models.json', **kwargs), vmids, alphas


def get_lsr_data(path, vmids):
    file_list = glob.glob(os.path.join(path, "lsr_0p*"))
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

    data_err = np.full_like(data, 0.1)  # assuming constant error for simplicity
    return data, data_err


def register_lsr_models(model):

    @model.register
    def ring_only(amplring, ringlat, ringwidth, i_mag, alpha0):
        return model.ring(i_mag, ringlat, ringwidth, alpha0, amplring)

    @model.register
    def ring_one_spot(lon1, amplon1, lat1, amplring):
        model.ringwidth = np.pi/6
        return model.combine(
            model.spot(lat1, lon1, model.width1, amplon1),
            model.equatorial_ring(amplring)
        )

    @model.register
    def quiescent_background_one_spot(lon1, amplon1, lat1, amplring):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, model.width1, amplon1),
            model.equatorial_ring(amplring)
        )

    @model.register
    def ring_two_spots(lon1, lon3, amplon1, amplon3, amplring, lat1, lat2):
        model.ringwidth = np.pi/6
        return model.combine(
            model.spot(lat1, lon1, model.width1, amplon1),
            model.spot(lat2, lon3, model.width1, amplon3),
            model.equatorial_ring(amplring)
        )

    @model.register
    def quiescent_background_two_spots(lon1, lon2, amplon1, amplon2, amplback, lat1, lat2):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, model.width1, amplon1),
            model.spot(lat2, lon2, model.width1, amplon2),
            model.equatorial_ring(amplback),
        )

    names = ['Ring Only', 'Ring + 1 Spot', 'Ring + 2 Spots',
                       'Quiescent bkg. + 1 Spot', 'Quiescent bkg. + 2 Spots']
    return [ring_only, ring_one_spot, ring_two_spots, 
            quiescent_background_one_spot, quiescent_background_two_spots], names
