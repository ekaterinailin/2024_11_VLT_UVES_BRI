
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
from funcs.analytical_ring import get_equivr_lines, compute_curve_length


def model_ring(rot0, obliquity, colat_min, width, ampl, broaden, ampl_base, vrs=None, rots=None, foreshortening=False, base=None ): 

    spectra = []
    for rot in rots:
        xs, zs, masks = get_equivr_lines(vrs, colat_min, colat_min+width, obliquity, rot+rot0, N=40)
        spectra.append(compute_curve_length(xs, zs, masks, foreshortening=foreshortening) + base * ampl_base)

    spectra = np.array([gaussian_filter1d(spectrum, broaden) for spectrum in spectra])

    # print(spectra.shape)    
    maxval = np.max(spectra)

    if maxval == 0:
        return np.ones_like(spectra[0]) 
    else:
        return spectra / maxval * ampl + 1



def prior_transform_ring(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()

    # start rotation phase from 0 to 2pi
    params[0] = cube[0] * 2 * np.pi

    # magnetic obliquity from 0 to 90
    params[1] = cube[1] * np.pi/2 
    
    # minimum co-latitude from -0 to 90
    params[2] = cube[2] * np.pi / 2 

    # width of the ring from small to 90
    params[3] = cube[3] * np.pi/2 + 0.03

    # amplitude
    params[4] = cube[4] *  5

    # thermal broadening
    params[5] = cube[5] * 200 + 20

    # amplitude base
    params[6] = cube[6] * 2

    return params



def setup_model_sampler(modelname, vrs, yerr2, ys, rots, foreshortening, base):
     
    if modelname == "ring":
        model = lambda *args: model_ring(*args, vrs=vrs, rots=rots, foreshortening=foreshortening, base=base)   
        
        p1 = ["rot0", "obliquity", "colat_min", "width", "ampl","broaden","ampl_base"]
        def loglike(theta):
            rot0, obliquity, colat_min, width, ampl, broaden, ampl_base = theta
            model_ = model(rot0, obliquity, colat_min, width, ampl, broaden, ampl_base)
            return  -0.5 * np.sum((ys - model_) ** 2 / yerr2) 
        logprior = prior_transform_ring
        wrapped_params =[True, False, False, False, False, False, False]

  

    return ultranest.ReactiveNestedSampler(p1, loglike, logprior, wrapped_params=wrapped_params), p1, model, loglike
        