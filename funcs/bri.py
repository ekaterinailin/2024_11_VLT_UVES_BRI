from funcs.modelfactory import SpectralModelFactory
from scipy.interpolate import interp1d
import astropy.units as u
import numpy as np

import glob
import os

PROT_BRI = 3.052 / 24.  # days
NAME_BRI = "BRI 0021"

def setup_bri_factory(**kwargs):

    vbins = np.linspace(-90, 90, 101)
    vmids = 0.5 * (vbins[1:] + vbins[:-1])
    broaden = 18
    i_rot = np.pi/2 - 51.7 / 180 * np.pi  # stellar inclination in radians

    # rotation period in days
    P_rot = PROT_BRI
    omega = 2 * np.pi / P_rot

    # stellar radius in solar radii
    R_star = (0.11 * u.R_sun).value

    # maximum rot velocity of the star in km/s
    vmax = omega * R_star * 695700. / 86400. # km/s

    # velocity step size
    ddv = vmids[1] - vmids[0]

    alphas = np.linspace(0, 2 * np.pi, 4)

    radius = 15 / 180 * np.pi
    ringwidth = 30 / 180 * np.pi

    model = SpectralModelFactory(
        vbins, vmids, broaden, i_rot, omega, vmax, R_star, ddv, alphas, radius, ringwidth,
        registry_file='bri_models.json', **kwargs)

    # set i_mag range to 0, pi
    new_bounds = {'truei_mag': (np.pi/2, np.pi), "ringwidth": (5 / 180 * np.pi, 90 / 180 * np.pi),
                  "lon1": (0,2*np.pi), 'lat1': (0, np.pi), "trueringlat": (0, np.pi/2), "trueringlat2": (-np.pi/2, 0)}
    model.parameter_bounds.update(new_bounds)       ,

    # Initialize once:
    return model, vmids, alphas

def get_bri_data(path, vmids):

    # get data from inside the folder, i.e. four spectra with names lsr_0pxxxxxx 
    file_list = glob.glob(os.path.join(path, "lsr_0p*"))
    file_list.sort()

    print(f"Found {len(file_list)} spectra files in {path}.")

    fluxes = []
    wavs = []
    alphas = []
    for file in file_list:
        alphas.append(float(file.split("lsr_0p")[1]) / 1000000)
        data = np.loadtxt(file)
        wav = data[:,0]
        flux = data[:,1]
        wavs.append(wav)
        fluxes.append(np.array(flux))

    alphas = np.array(alphas) * np.pi * 2


    assert len(file_list) == 4 # we expect four files

    c = 299792.458  # speed of light in km/s
    lambda_0 = 6562.8  # reference wavelength in Angstroms
    velocities = (wavs[0] - lambda_0) / lambda_0 * c

    # interpolate the observed fluxes to the model vmids
    interp_fluxes = []
    for flux, wav in zip(fluxes, wavs):
        velocities = (wav - lambda_0) / lambda_0 * c
        f = interp1d(velocities, flux, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_fluxes.append(f(vmids))

    data = np.array(interp_fluxes)
    data_err = np.zeros_like(data)

    errpath = "results_bri/subtracted_spectra_errvals.txt"
    # read the error values from the text file of this form
    with open(errpath, "r") as f:
        lines = f.readlines()
        errvals = []
        for line in lines:
            parts = line.strip().split(":")
            if len(parts) == 2:
                errvals.append(float(parts[1].strip()))

    # we have 8 error values, but only 4 spectra, take 1 and 4, 2 and 5, 3 and 6, 4 and 7 and quadratically add
    selected_errvals = []
    for i in range(4):
        err1 = errvals[i]
        err2 = errvals[i + 4]
        combined_err = np.sqrt(err1**2 + err2**2) / 2 
        selected_errvals.append(combined_err)

    # propagate to data_err
    for i in range(4):
        data_err[i,:] = selected_errvals[i]
    
    assert (data_err != 0).all(), "Data errors contain zeros!"


    dalpha = (0.75 *u.hr / (3.052 * u.day) * 2 * np.pi).value 
    print(f"BRI data dalpha (radians): {dalpha}")

    return data, data_err, dalpha

def register_bri_models(model, dalpha=0):

    @model.register
    def ring_only(amplring, trueringlat, ringwidth, truei_mag, alpha0):
        m1 = (model.ring(truei_mag, trueringlat, ringwidth, alpha0 - dalpha/2, amplring) - 1) / 3 + 1
        m2 = (model.ring(truei_mag, trueringlat, ringwidth, alpha0, amplring) - 1) / 3 +1
        m3 = (model.ring(truei_mag, trueringlat, ringwidth, alpha0 + dalpha/2, amplring)-1) / 3 +1
        return model.combine(m1, m2, m3) 
    
    @model.register
    def ring_only_show(amplring, trueringlat, ringwidth, truei_mag, alpha0):
        return model.ring(truei_mag, trueringlat, ringwidth, alpha0, amplring)
    
    @model.register
    def spot_only(lon1, amplon1, truelat, width1):
        m1 = (model.spot(truelat, lon1, width1, amplon1) - 1) / 3 + 1
        m2 = (model.spot(truelat, lon1 + dalpha/2, width1, amplon1) - 1) / 3 + 1
        m3 = (model.spot(truelat, lon1 - dalpha/2, width1, amplon1) - 1) / 3 + 1
        return model.combine(m1, m2, m3)
    
    @model.register
    def spot_only_show(lon1, amplon1, truelat, width1):
        return model.spot(truelat, lon1, width1, amplon1)
        
    @model.register
    def two_spots(lon1, amplon1, truelat, lon2, amplon2, truelat2, width1, width2):
        m1 = (model.spot(truelat, lon1, width1, amplon1) - 1) / 3 + 1
        m2 = (model.spot(truelat, lon1 + dalpha/2, width1, amplon1) - 1) / 3 + 1
        m3 = (model.spot(truelat, lon1 - dalpha/2, width1, amplon1) - 1) / 3 + 1
        n1 = (model.spot(truelat2, lon2, width2, amplon2) - 1) / 3 + 1
        n2 = (model.spot(truelat2, lon2 + dalpha/2, width2, amplon2) - 1) / 3 + 1
        n3 = (model.spot(truelat2, lon2 - dalpha/2, width2, amplon2) - 1) / 3 + 1
        return model.combine(m1, m2, m3, n1, n2, n3)
    
    @model.register
    def two_spots_show(lon1, amplon1, truelat, lon2, amplon2, truelat2, width1, width2):
        return model.combine(
            model.spot(truelat, lon1, width1, amplon1),
            model.spot(truelat2, lon2, width2, amplon2)
        )
    
    @model.register
    def loose_ring_one_spot(truelat, lon1, amplon1, amplring, trueringlat, truei_mag, alpha0, width1, ringwidth):
        m1 = (model.ring(truei_mag, trueringlat, ringwidth, alpha0 - dalpha/2, amplring) - 1) / 3 + 1
        m2 = (model.ring(truei_mag, trueringlat, ringwidth, alpha0, amplring) - 1) / 3 +1
        m3 = (model.ring(truei_mag, trueringlat, ringwidth, alpha0 + dalpha/2, amplring)-1) / 3 +1
        n1 = (model.spot(truelat, lon1, width1, amplon1) - 1) / 3 + 1
        n2 = (model.spot(truelat, lon1 + dalpha/2, width1, amplon1) - 1) / 3 + 1
        n3 = (model.spot(truelat, lon1 - dalpha/2, width1, amplon1) - 1) / 3 + 1
        return model.combine(m1, m2, m3, n1, n2, n3)
    
    @model.register
    def loose_ring_one_spot_show(truelat, lon1, amplon1, amplring, trueringlat, truei_mag, alpha0, width1, ringwidth):
        return model.combine(
            model.ring(truei_mag, trueringlat, ringwidth, alpha0, amplring),
            model.spot(truelat, lon1, width1, amplon1)
        )


    names = ['Ring', "Spot", "2 Spots", "Ring + 1 Spot"]#, 'Ring + 1 Spot', 'Ring + 2 Spots',
            #  'Quiescent bkg. + 1 Spot', 'Quiescent bkg. + 2 Spots',
            #  '2 Spots','Loose Ring + 1 Spot']
    return [ring_only, spot_only, two_spots, loose_ring_one_spot], names#, ring_one_spot, ring_two_spots, 
            # quiescent_background_one_spot, quiescent_background_two_spots,
            # two_spots, loose_ring_one_spot], names