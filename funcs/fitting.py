
"""
UTF-8, Python 3.11.7

------------
Auroral Oval
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


"""
import numpy as np 

from scipy.ndimage import gaussian_filter1d
from funcs.auroralring import AuroralRing


def model_ring(vbins, vmids, imag, phimin, dphi, alpha_0,broaden, ampl, 
               alphas=None, foreshortening=False, i_rot=0, omega=0, vmax=0, 
               R_star=0, ddv=0.1, obj_only=False, typ="ring"): 

    # mid latitude of ring around magnetic axis in radians
    mid_lat = phimin + dphi / 2

    # define the auroral ring
    ring = AuroralRing(i_rot=i_rot, i_mag=imag, latitude=mid_lat,
                        width=dphi, Rstar=R_star,  
                    v_bins=vbins, v_mids=vmids, omega=omega, vmax=vmax, typ=typ)

    alphas_ = alphas + alpha_0 

    spectra = ring.get_flux_numerically(alphas_, normalize=False, 
                                        foreshortening=foreshortening,)

    if obj_only:
        return ring
    else:    
        # wav = ring.v_mids / 2.9979246e5 * 6562.8 + 6562.8
        dv = broaden / ddv
        spectra = np.array([gaussian_filter1d(spectrum, dv) for spectrum in spectra])

        # print(spectra.shape)    
        maxval = np.max(spectra)

        if maxval == 0:
            return np.ones_like(spectra[0]) 
        else:
            return spectra / maxval * ampl + 1
    

def model_spot(vbins, vmids, lat1, lon1, width1, ampl, broaden, 
               i_rot=0, omega=0, vmax=0, R_star=0, ddv=0.1,alphas=0,
             foreshortening=True, obj_only=False, 
             typ="spot"): 

  
    # define the auroral ring
    ring = AuroralRing(i_rot=i_rot, latitude=lat1,
                        width=width1, Rstar=R_star,  
                         longitude=lon1, 
                    v_bins=vbins, v_mids=vmids, omega=omega, vmax=vmax,  typ=typ)

    dv = broaden / ddv

    spectra = ring.get_spot_flux_numerically(alphas, normalize=False, foreshortening=foreshortening)

    if obj_only:
        return ring
    else:

        spectra = np.array([gaussian_filter1d(spectrum, dv) for spectrum in spectra])
        maxval = np.max(spectra)

        if maxval == 0:
            return np.ones_like(spectra[0]) 
        else:
            return spectra / maxval * ampl + 1
    
