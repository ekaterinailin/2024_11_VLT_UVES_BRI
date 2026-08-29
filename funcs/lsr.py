import glob
import os
import numpy as np
from .modelfactory import SpectralModelFactory
from .auroralring import AuroralRing
from scipy.interpolate import interp1d
import astropy.units as u

PROT_LSR = 2.864 / 24.  # days
NAME_LSR = "LSR J1835"

# Grid resolution for the spherical model surface. Used both to run the
# UltraNest fits (run_forward_model_lsr.py) and to reproduce/visualise them
# (read_forward_model_results.ipynb) -- keep both pointed at this constant so
# they can never silently drift apart.
GRIDSIZE_LSR = 30000


def setup_lsr_factory(gridsize=GRIDSIZE_LSR, **kwargs):

    vbins = np.linspace(-90, 90, 51)
    vmids = 0.5 * (vbins[1:] + vbins[:-1])
    broaden = 9
    gamma_kms = 10.5  
    i_rot = np.pi/2 - 90 / 180 * np.pi  # stellar inclination in radians

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


def register_lsr_models(model):

    @model.register
    def ring_only(amplring, ringlat, ringwidth, i_mag, alpha0):
        return model.ring(i_mag, ringlat, ringwidth, alpha0, amplring)


    @model.register
    def quiescent_background_one_spot(lon3, amplon1, lat1, amplback, width1):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon3, width1, amplon1),
            model.equatorial_ring(amplback)
        )


    @model.register
    def quiescent_background_two_spots(lon1, lon2, amplon1, amplon2, amplback, lat1, lat2, width1, width2):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.equatorial_ring(amplback),
        )
    

    @model.register
    def loose_ring_one_spot(lon3, amplon1, lat1, amplring, ringlat, i_mag, alpha0, width1, ringwidth2):
        return model.combine(
            model.spot(lat1, lon3, width1, amplon1),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    
    @model.register
    def loose_ring_two_spots(lon1, lon2, amplon1, amplon2, lat1, lat2, amplring, 
                             ringlat, i_mag, alpha0, ringwidth2, width1, width2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    
    @model.register
    def loose_ring_quiescent_background(amplring, ringlat, i_mag, alpha0, ringwidth2, amplback):
        model.ringwidth = np.pi
        return model.combine(model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
                             model.equatorial_ring(amplback))
    
    @model.register
    def loose_ring_quiescent_background_one_spot(amplring, ringlat, i_mag, alpha0, ringwidth2, amplback,
                                                 lon1, amplon1, lat1, width1):
        model.ringwidth = np.pi
        return model.combine(model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
                            model.spot(lat1, lon1, width1, amplon1),
                             model.equatorial_ring(amplback))
    
    @model.register
    def three_spots(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3):
        # model.width1 = np.pi/4
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
        )
    
    @model.register
    def quiescent_background_three_spots(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3, amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.equatorial_ring(amplback)
        )
    
    @model.register
    def two_rings(amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth2, ringwidth3):
        return model.combine(
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth3, alpha0, amplring2)
        )
    
    @model.register
    def two_rings_one_spot(amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                         lon1, amplon1, lat1, width1):
        return model.combine(
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.spot(lat1, lon1, width1, amplon1)
        )

    @model.register
    def two_rings_one_spot_quiescent_background(amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                         lon1, amplon1, lat1, width1, amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.spot(lat1, lon1, width1, amplon1),
            model.equatorial_ring(amplback)
        )

    @model.register
    def two_rings_quiescent_background(amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2, amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.equatorial_ring(amplback)
        )
    
    @model.register
    def loose_ring_quiescent_background_two_spots(amplring, ringlat, i_mag, alpha0, ringwidth2, amplback,
                                                 lon1, amplon1, lat1, width1,
                                                 lon2, amplon2, lat2, width2):
        model.ringwidth = np.pi
        return model.combine(model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
                            model.spot(lat1, lon1, width1, amplon1),
                            model.spot(lat2, lon2, width2, amplon2),
                             model.equatorial_ring(amplback))

    @model.register
    def two_rings_two_spots(amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                         lon1, amplon1, lat1, width1,
                         lon2, amplon2, lat2, width2):
        return model.combine(
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2)
        )

    @model.register
    def four_spots(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4)
        )
    
    @model.register
    def four_spots_medium_low(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4)
        )

    
    @model.register
    def three_spots_one_ring(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )

    @model.register
    def four_spots_one_ring(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    

    @model.register
    def four_spots_one_ring_18broaden(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    
    @model.register
    def four_spots_one_ring_4broaden(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    
    @model.register
    def four_spots_one_ring_low_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )


        
    @model.register
    def four_spots_one_ring_medium_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    
    @model.register
    def four_spots_one_ring_medium_low_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplring, ringlat, i_mag, alpha0, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring)
        )
    
    @model.register
    def three_spots_two_rings(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
        )

    @model.register
    def three_spots_two_rings_medium_low_inc(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
        )

    @model.register
    def three_spots_two_rings_medium_inc(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
        )
    
    @model.register
    def four_spots_two_rings_medium_low_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                           amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2)
        )
    
    @model.register
    def five_spots_two_rings_medium_low_inc(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5, lat1, lat2, lat3, lat4, lat5, width1, width2, width3, width4, width5,
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

    @model.register
    def three_spots_one_ring_quiescent_background(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring, ringlat, i_mag, alpha0, ringwidth2, amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
            model.equatorial_ring(amplback)
        )
    

    @model.register
    def three_spots_one_ring_quiescent_background_medium_low_inc(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring, ringlat, i_mag, alpha0, ringwidth2, amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
            model.equatorial_ring(amplback)
        )

    @model.register
    def three_spots_one_ring_quiescent_background_medium_inc(lon1, lon2, lon3, amplon1, amplon2, amplon3, lat1, lat2, lat3, width1, width2, width3,
                           amplring, ringlat, i_mag, alpha0, ringwidth2, amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.ring(i_mag, ringlat, ringwidth2, alpha0, amplring),
            model.equatorial_ring(amplback)
        )
    


    @model.register
    def two_spots_two_rings_quiescent_background(lon1, lon2, amplon1, amplon2, lat1, lat2, width1, width2,
                                                 amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                                                 amplback):
        
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.equatorial_ring(amplback)
        )

    @model.register
    def two_spots_two_rings_quiescent_background_medium_low_inc(lon1, lon2, amplon1, amplon2, lat1, lat2, width1, width2,
                                                 amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                                                 amplback):
        
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.equatorial_ring(amplback)
        )
    
    @model.register
    def two_spots_two_rings_quiescent_background_medium_inc(lon1, lon2, amplon1, amplon2, lat1, lat2, width1, width2,
                                                 amplring1, ringlat, amplring2, i_mag, alpha0, ringwidth1, ringwidth2,
                                                 amplback):
        
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.ring(i_mag, ringlat, ringwidth1, alpha0, amplring1),
            model.ring(i_mag, -ringlat, ringwidth2, alpha0, amplring2),
            model.equatorial_ring(amplback)
        )

    @model.register
    def four_spots_quiescent_background(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.equatorial_ring(amplback)
        )
    
    @model.register
    def four_spots_quiescent_background_medium_low_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.equatorial_ring(amplback)
        )
    
    @model.register
    def four_spots_quiescent_background_medium_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4, lat1, lat2, lat3, lat4, width1, width2, width3, width4,
                       amplback):
        model.ringwidth = np.pi
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.equatorial_ring(amplback)
        )
    
    @model.register
    def five_spots(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5, lat1, lat2, lat3, lat4, lat5, width1, width2, width3, width4, width5):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.spot(lat5, lon5, width5, amplon5)
        )

    @model.register
    def five_spots_medium_low_inc(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5, lat1, lat2, lat3, lat4, lat5, width1, width2, width3, width4, width5):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.spot(lat5, lon5, width5, amplon5)
        )
    
    @model.register
    def five_spots_medium_inc(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5, lat1, lat2, lat3, lat4, lat5, width1, width2, width3, width4, width5):
        return model.combine(
            model.spot(lat1, lon1, width1, amplon1),
            model.spot(lat2, lon2, width2, amplon2),
            model.spot(lat3, lon3, width3, amplon3),
            model.spot(lat4, lon4, width4, amplon4),
            model.spot(lat5, lon5, width5, amplon5)
        )
    

    # now we need to add a set of 6-component modelS

    @model.register
    def six_spots(lon1, lon2, lon3, lon4, lon5, lon6, amplon1, amplon2, amplon3, amplon4, amplon5, amplon6,
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
    
    @model.register
    def six_spots_medium_low_inc(lon1, lon2, lon3, lon4, lon5, lon6, amplon1, amplon2, amplon3, amplon4, amplon5, amplon6,
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

    @model.register
    def five_spots_quiescent_background(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
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

    @model.register
    def six_spots_ring(lon1, lon2, lon3, lon4, lon5, lon6, amplon1, amplon2, amplon3, amplon4, amplon5, amplon6,
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


    @model.register
    def five_spots_quiescent_background_medium_low_inc(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
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
    
    @model.register
    def five_spots_ring(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
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


       
    @model.register
    def five_spots_ring_medium_low_inc(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
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
    
    @model.register
    def four_spots_one_ring_quiescent_background(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
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
    
    @model.register
    def four_spots_one_ring_quiescent_background_medium_low_inc(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
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
    
    @model.register
    def four_spots_two_rings(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
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

    @model.register
    def four_spots_two_rings_18broaden(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
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
    
    @model.register
    def four_spots_two_rings_4broaden(lon1, lon2, lon3, lon4, amplon1, amplon2, amplon3, amplon4,
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
    
    @model.register
    def five_spots_ring_4broaden(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
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

    @model.register
    def five_spots_ring_18broaden(lon1, lon2, lon3, lon4, lon5, amplon1, amplon2, amplon3, amplon4, amplon5,
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

    @model.register
    def three_spots_two_rings_quiescent_background(lon1, lon2, lon3, amplon1, amplon2, amplon3,
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
    
    @model.register
    def three_spots_two_rings_quiescent_background_medium_low_inc(lon1, lon2, lon3, amplon1, amplon2, amplon3,
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
    
    @model.register
    def seven_spots(lon1, lon2, lon3, lon4, lon5, lon6, lon7,
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
# 'Ring', 'Eq. ring + 1 Spot', 'Eq. ring + 2 Spots', '2 Spots',
#              'Q. bkg. + 1 Spot', 'Q. bkg. + 2 Spots',
#             'Ring + 1 Spot', #'2 Loose Spots','2 Rings',
            #  'Q. bkg. + Ring',

    names = ["Ring", "Ring + Q. bkg.", 'Ring + 1 Spot', '2 Rings', "1 Spot + Q. bkg.",
             "Ring + 2 Spots",  'Ring + 1 Spot +Q. bkg.', '3 Spots', '3 Spots + Q. bkg.','2 Rings + Q. bkg.',
             'Ring + 2 Spots + Q. bkg.', '2 Rings + 1 Spot', '2 Rings + 1 Spot + Q. bkg.', '2 Rings + 2 Spots',
             "4 Spots", 'Ring + 3 Spots', 'Ring + 4 Spots','2 Rings + 3 Spots','Ring + 3 Spots + Q. bkg.', 
             '2 Rings + 2 Spots + Q. bkg.', '4 Spots + Q. bkg.', '5 Spots', '2 Spots + Q. bkg.',
             '6 Spots', 'Ring + 5 Spots', '5 Spots + Q. bkg.', 'Ring + 4 Spots + Q. bkg.', '2 Rings + 4 Spots',
             '2 Rings + 3 Spots + Q. bkg.', '7 Spots','Ring + 4 Spots at 70 deg inc.','Ring + 4 Spots at 80 deg inc.', 
             'Ring + 4 Spots at 75 deg inc.', '4 Spots at 75 deg inc.', '4 Spots + Q. bkg. at 75 deg inc.',
             '2 Rings + 3 Spots at 75 deg inc.','2 Rings + 2 Spots + Q. bkg. at 75 deg inc.',
             '5 Spots at 75 deg inc.', 'Ring + 3 Spots + Q. bkg. at 75 deg inc.', '2 Rings + 4 Spots at 75 deg inc.',
             '6 Spots at 75 deg inc.','Ring + 5 Spots at 75 deg inc.', '5 Spots + Q. bkg. at 75 deg inc.',
             '2 Rings + 3 Spots + Q. bkg. at 75 deg inc.','Ring + 4 Spots + Q. bkg. at 75 deg inc.',
              '2 Rings + 3 Spots at 80 deg inc.','Ring + 3 Spots + Q. bkg. at 80 deg inc.',
             '2 Rings + 2 Spots + Q. bkg. at 80 deg inc.', '5 Spots at 80 deg inc.','4 Spots + Q. bkg. at 80 deg inc.',
             '2 Rings + 5 Spots at 75 deg inc.', '2 Rings + 4 Spots at 18 kms broadening', '4 Spots + 1 Ring at 18 kms broadening',
             '2 Rings + 4 Spots at 4 kms broadening','4 Spots + 1 Ring at 4 kms broadening','Ring + 6 Spots',
             'Ring + 5 Spots at 4 kms broadening', 'Ring + 5 Spots at 18 kms broadening']
             
    return [ring_only, loose_ring_quiescent_background, loose_ring_one_spot, two_rings,quiescent_background_one_spot,
            loose_ring_two_spots, loose_ring_quiescent_background_one_spot, three_spots,
            quiescent_background_three_spots, two_rings_quiescent_background,
            loose_ring_quiescent_background_two_spots, two_rings_one_spot,
            two_rings_one_spot_quiescent_background, two_rings_two_spots,
            four_spots, three_spots_one_ring, four_spots_one_ring, three_spots_two_rings,
            three_spots_one_ring_quiescent_background, two_spots_two_rings_quiescent_background, 
            four_spots_quiescent_background, five_spots, quiescent_background_two_spots,
            six_spots, five_spots_ring, five_spots_quiescent_background,four_spots_one_ring_quiescent_background,
            four_spots_two_rings,three_spots_two_rings_quiescent_background, seven_spots, four_spots_one_ring_low_inc,
            four_spots_one_ring_medium_inc, four_spots_one_ring_medium_low_inc, four_spots_medium_low,
            four_spots_quiescent_background_medium_low_inc, three_spots_two_rings_medium_low_inc,
            two_spots_two_rings_quiescent_background_medium_low_inc,five_spots_medium_low_inc,
            three_spots_one_ring_quiescent_background_medium_low_inc,four_spots_two_rings_medium_low_inc,
            six_spots_medium_low_inc, five_spots_ring_medium_low_inc, five_spots_quiescent_background_medium_low_inc,
            three_spots_two_rings_quiescent_background_medium_low_inc,four_spots_one_ring_quiescent_background_medium_low_inc,
            three_spots_two_rings_medium_inc,three_spots_one_ring_quiescent_background_medium_inc,
            two_spots_two_rings_quiescent_background_medium_inc, five_spots_medium_inc,
            four_spots_quiescent_background_medium_inc,five_spots_two_rings_medium_low_inc,
            four_spots_two_rings_18broaden,four_spots_one_ring_18broaden,
            four_spots_two_rings_4broaden, four_spots_one_ring_4broaden, six_spots_ring,
            five_spots_ring_4broaden, five_spots_ring_18broaden], names


