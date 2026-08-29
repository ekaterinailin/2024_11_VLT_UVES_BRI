import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import ultranest.stepsampler
from funcs.lsr import setup_lsr_factory, get_lsr_data, register_lsr_models, GRIDSIZE_LSR

import os
import sys

if __name__ == "__main__":

    #read modelname from command line
    modelname = sys.argv[1]

    model, vmids, alphas = setup_lsr_factory(gridsize=GRIDSIZE_LSR)

    model_funcs, model_funcs_pretty_names = register_lsr_models(model)

    path = "data/lsr1835bins20/"
    # path = "/home/ilin/Documents/2025_10_pydoppler/scripts/lsr1835bins20/"

    data, data_err = get_lsr_data(path, vmids)


    mmodel = model.get_model(modelname)
    # Create ultranest functions
    param_names = model.get_param_names(mmodel)
    prior_transform = model.create_ultranest_prior(mmodel)
    log_likelihood = model.create_ultranest_likelihood(mmodel, data, data_err)
    wrapped_params = model.get_wrapped_params(mmodel)

    # Create and run sampler
    sampler = ultranest.ReactiveNestedSampler(
        param_names=param_names,
        loglike=log_likelihood,
        transform=prior_transform,
        wrapped_params=wrapped_params,
        log_dir=f"results_lsr/{mmodel.__name__}_ultranest",
        resume="resume")

    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=2*len(param_names)*2,
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction,)


    result = sampler.run(
        min_num_live_points=400,
        dlogz=0.5,
        max_ncalls=7500000000,
        Lepsilon=0.001
    )
