
"""
UTF-8, Python 3.11.7

------------
Auroral Oval
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


"""
import numpy as np 
import ultranest

from scipy.ndimage import gaussian_filter1d
from funcs.auroralring import AuroralRing


def model_ring(vbins, vmids, imag, phimin, dphi, alpha_0,broaden, ampl, 
               alphas=None, foreshortening=False, i_rot=0, omega=0, vmax=0, 
               R_star=0, ddv=0.1, obj_only=False, background=0, typ="ring"): 

    # mid latitude of ring around magnetic axis in radians
    mid_lat = phimin + dphi / 2

    # define the auroral ring
    ring = AuroralRing(i_rot=i_rot, i_mag=imag, latitude=mid_lat,
                        width=dphi, Rstar=R_star,  
                    v_bins=vbins, v_mids=vmids, omega=omega, vmax=vmax, typ=typ)

    alphas_ = alphas + alpha_0 

    spectra = ring.get_flux_numerically(alphas_, 0, 0, np.pi*2, normalize=False, 
                                        foreshortening=foreshortening, background=background)

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
             foreshortening=True, obj_only=False, croissant=False,
             background=0, typ="spot"): 

  
    # define the auroral ring
    ring = AuroralRing(i_rot=i_rot, latitude=lat1,
                        width=width1, Rstar=R_star,  
                         longitude=lon1, 
                    v_bins=vbins, v_mids=vmids, omega=omega, vmax=vmax, croissant=croissant, typ=typ)

    dv = broaden / ddv

    spectra = ring.get_spot_flux_numerically(alphas, normalize=False, foreshortening=foreshortening, nspots=1,
                                             background=background)

    if obj_only:
        return ring
    else:

        spectra = np.array([gaussian_filter1d(spectrum, dv) for spectrum in spectra])
        maxval = np.max(spectra)

        if maxval == 0:
            return np.ones_like(spectra[0]) 
        else:
            return spectra / maxval * ampl + 1
    
def model_two_spots(vbins, vmids, lat1, lon1, width1, lat2, lon2, width2, ampl, broaden, 
                    alphas = None,ddv=0.1,
                    i_rot=0, omega=0, vmax=0, R_star=0,
                    foreshortening=True, obj_only=False): 

  
    # define the auroral ring
    ring = AuroralRing(i_rot=i_rot, latitude=lat1,
                        width=width1, Rstar=R_star,  
                         longitude=lon1, latitude2=lat2, width2=width2, longitude2=lon2, 
                    v_bins=vbins, v_mids=vmids, omega=omega, vmax=vmax, )

    # get flux for both spots
    spectra = ring.get_spot_flux_numerically(alphas, normalize=False, foreshortening=foreshortening, nspots=2)

    if obj_only:
        return ring
    else:

        # apply thermal broadening
        spectra = np.array([gaussian_filter1d(spectrum, broaden / ddv) for spectrum in spectra])

        # normalize the spectra and return
        maxval = np.max(spectra)
        if maxval == 0:
            return np.ones_like(spectra[0]) 
        else:
            return spectra / maxval * ampl + 1


def model_points(vbins, vmids, lat1, lon1,lat2, lon2, relamp, ampl, broaden,
                    i_rot=0, omega=0, vmax=0, R_star=0,ddv=0.1,alphas=None,
                 foreshortening=True, obj_only=False):   

  
    # define the auroral ring
    ring = AuroralRing(i_rot=i_rot, latitude=np.array([lat1,lat2,]), 
                       longitude=np.array([lon1,lon2,]), width=0, amps=np.array([1,relamp]),
                         Rstar=R_star, v_bins=vbins, v_mids=vmids, omega=omega, vmax=vmax, )

    # get flux for both point spots
    spectra = ring.get_spot_flux_numerically(alphas, normalize=False, foreshortening=foreshortening, nspots=999)

    if obj_only:
        return ring
    else:

        # apply thermal broadening
        spectra = np.array([gaussian_filter1d(spectrum, broaden / ddv) for spectrum in spectra])

        # normalize the spectra and return
        maxval = np.max(spectra)
        if maxval == 0:
            return np.ones_like(spectra[0]) 
        else:
            return spectra / maxval * ampl + 1



def prior_transform_ring(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    
    # magnetic obliquity from 0 to 90
    params[0] = cube[0] * np.pi/2 
    
    # minimum latitude from -90 to 90
    params[1] = cube[1] * np.pi - np.pi/2

    # width of the ring from 0 to 180
    params[2] = cube[2] * (np.pi/2 -0.05 - params[1]) + 0.05 

    # start rotation phase from 0 to 1
    params[3] = cube[3] * 2 * np.pi

    # thermal broadening
    params[4] = cube[4] * 30 + 5

    # amplitude
    params[5] = cube[5] *  5

    return params


def prior_transform_spot(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    
    # lat
    params[0] = cube[0] * np.pi/2
    
    # lon
    params[1] = cube[1] * 2 * np.pi

    # width
    params[2] = cube[2] * np.pi

    # ampl
    params[-2] = cube[-2] * 3 

    # broad
    params[-1] = cube[-1] *  20  + 10
    return params


def prior_transform_twospots(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    
    # lat
    params[0] = cube[0] * np.pi/2
    
    # lon
    params[1] = cube[1] * 2 * np.pi

    # width
    params[2] = cube[2] * np.pi

        # lat
    params[3] = cube[3] * np.pi/2
    
    # lon
    params[4] = cube[4] * 2 * np.pi

    # width
    params[5] = cube[5] * np.pi


    # ampl
    params[-2] = cube[-2] * 3 

    # broad
    params[-1] = cube[-1] *  20  + 10
    return params


def prior_transform_points(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    
    # lat
    params[0] = np.pi/2 - cube[0] * np.pi/2
    
    # lon
    params[1] = cube[1] * np.pi * 2


    # lat
    params[2] = np.pi/2 - cube[2] * np.pi/2
    
    # lon
    params[3] = cube[3] * np.pi * 2

    # relamp
    params[4] = cube[4] 
    
    # ampl
    params[-2] = cube[-2] * 3 

    # broad
    params[-1] = cube[-1] *  20  + 10

    return params

def setup_model_sampler(modelname, i_rot, omega, vmax, R_star, ddv, 
                  vbins, vmids, yerr2, ys,alphas, foreshortening):
     
    if modelname == "ring":
        model = lambda *args: model_ring(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, 
                                         R_star=R_star, ddv=ddv, foreshortening=foreshortening)   
        obj = lambda *args: model_ring(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, 
                                         R_star=R_star, ddv=ddv, foreshortening=foreshortening, obj_only=True)
        p1 = ["imag", "phimin", "dphi", "alpha_0", "broaden","ampl"]
        def loglike(theta):
            imag, phimin, dphi, alpha_0, broaden, ampl = theta
            model_ = model(vbins, vmids, imag, phimin, dphi, alpha_0, broaden, ampl)
            return  -0.5 * np.sum((ys - model_) ** 2 / yerr2) 
        logprior = prior_transform_ring
        wrapped_params =[False, False, False, True, False, False]

    elif modelname == "spot":
        model = lambda *args: model_spot(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, R_star=R_star,
                                          ddv=ddv, foreshortening=foreshortening)
        obj = lambda *args: model_spot(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, R_star=R_star,
                                          ddv=ddv, foreshortening=foreshortening, obj_only=True)
        p1 = ["lat1", "lon1", "width1", "ampl", "broaden"]
        def loglike(theta):
            lat1, lon1, width1, ampl, broaden = theta
            model_ = model(vbins, vmids, lat1, lon1, width1, ampl, broaden)
            return  -0.5 * np.sum((ys - model_) ** 2 / yerr2) 
        logprior = prior_transform_spot
        wrapped_params =[False, True, False, False, False]

    elif modelname == "twospots":
        model = lambda *args: model_two_spots(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, R_star=R_star, 
                                              ddv=ddv, foreshortening=foreshortening)
        obj = lambda *args: model_two_spots(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, R_star=R_star, 
                                              ddv=ddv, foreshortening=foreshortening, obj_only=True)
        p1 = ["lat1", "lon1", "width1", "lat2", "lon2", "width2", "ampl", "broaden"]
        def loglike(theta):
            lat1, lon1, width1, lat2, lon2, width2, ampl, broaden = theta
            model_ = model(vbins, vmids, lat1, lon1, width1, lat2, lon2, width2, ampl, broaden)
            return  -0.5 * np.sum((ys - model_) ** 2 / yerr2) 
        logprior = prior_transform_twospots
        wrapped_params =[False, True, False, False, True, False, False, False]

    elif modelname == "points":
        model = lambda *args: model_points(*args, alphas=alphas, i_rot=i_rot, omega=omega, vmax=vmax, R_star=R_star, 
                                           ddv=ddv, foreshortening=foreshortening)
        p1 = ["lat1", "lon1", "lat2", "lon2", "relamp", "ampl", "broaden"]
        def loglike(theta):
            lat1, lon1, lat2, lon2, relamp, ampl, broaden = theta
            model_ = model(vbins, vmids, lat1, lon1, lat2, lon2, relamp, ampl, broaden)
            return  -0.5 * np.sum((ys - model_) ** 2 / yerr2) 
        logprior = prior_transform_points
        wrapped_params =[False, True, False, True, False, False, False]

    return ultranest.ReactiveNestedSampler(p1, loglike, logprior, wrapped_params=wrapped_params), p1, model, loglike, obj
        