#!/usr/bin/env python3
"""
Run an UltraNest fit for one LSR J1835 model structure at a given
broadening and inclination.

`broaden` and `inclination` are no longer baked into hand-duplicated model
functions (four_spots_one_ring_medium_inc, five_spots_ring_4broaden, ...) --
they're parameters. Given a base model structure (e.g. "four_spots_one_ring")
plus --broaden/--inclination, this builds a factory configured for exactly
that combination, registers every model structure on it under a name that
reflects all three (base structure, broaden, inclination), and saves the
UltraNest results to a matching results_lsr/<name>_ultranest/ folder --
with a small metadata JSON recording the same three values in that folder's
extra/ subdirectory, so the configuration is never left to be inferred from
the folder name alone.

At the historical default (broaden=9, inclination=90 deg), the name is just
the bare base model name -- e.g. results_lsr/four_spots_one_ring_ultranest/
-- unchanged from every existing completed fit.

Usage
-----
    # default configuration (broaden=9, inclination=90), same as before
    python 3b_run_forward_model_lsr.py four_spots_one_ring

    # a specific broaden/inclination -- saved to
    # results_lsr/four_spots_one_ring_broaden18_inc75_ultranest/
    python 3b_run_forward_model_lsr.py four_spots_one_ring --broaden 18 --inclination 75
"""

import argparse
import json
import os

import ultranest.stepsampler
from funcs.lsr import (
    setup_lsr_factory, get_lsr_data, register_lsr_models,
    variant_name, GRIDSIZE_LSR, DEFAULT_BROADEN, DEFAULT_INCLINATION_DEG,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_model", help='base model structure, e.g. "four_spots_one_ring" '
                                        "(see funcs.lsr.MODEL_RECIPES for the full list)")
    ap.add_argument("--broaden", type=float, default=DEFAULT_BROADEN,
                     help=f"Gaussian broadening in km/s (default {DEFAULT_BROADEN})")
    ap.add_argument("--inclination", type=float, default=DEFAULT_INCLINATION_DEG,
                     help=f"stellar inclination in degrees (default {DEFAULT_INCLINATION_DEG})")
    ap.add_argument("--data-path", default="data/lsr1835bins20/")
    args = ap.parse_args()

    model, vmids, alphas = setup_lsr_factory(
        gridsize=GRIDSIZE_LSR, broaden=args.broaden, inclination_deg=args.inclination)

    register_lsr_models(model, broaden=args.broaden, inclination_deg=args.inclination)

    name = variant_name(args.base_model, args.broaden, args.inclination)
    mmodel = model.get_model(name)

    data, data_err = get_lsr_data(args.data_path, vmids)

    # Create ultranest functions
    param_names = model.get_param_names(mmodel)
    prior_transform = model.create_ultranest_prior(mmodel)
    log_likelihood = model.create_ultranest_likelihood(mmodel, data, data_err)
    wrapped_params = model.get_wrapped_params(mmodel)

    log_dir = f"results_lsr/{mmodel.__name__}_ultranest"

    # Create and run sampler
    sampler = ultranest.ReactiveNestedSampler(
        param_names=param_names,
        loglike=log_likelihood,
        transform=prior_transform,
        wrapped_params=wrapped_params,
        log_dir=log_dir,
        resume="resume")

    # Record the (base model, broaden, inclination) this run used, so it's
    # never left to be inferred from the folder name alone. ReactiveNested
    # Sampler already created log_dir/extra/ as part of its own layout.
    extra_dir = os.path.join(log_dir, "extra")
    os.makedirs(extra_dir, exist_ok=True)
    with open(os.path.join(extra_dir, "model_variant.json"), "w") as f:
        json.dump({
            "base_model": args.base_model,
            "broaden_kms": args.broaden,
            "inclination_deg": args.inclination,
            "variant_name": name,
        }, f, indent=2)

    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=2*len(param_names)*2,
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction,)

    result = sampler.run(
        min_num_live_points=400,
        dlogz=0.5,
        max_ncalls=7500000000,
        Lepsilon=0.001
    )


if __name__ == "__main__":
    main()
