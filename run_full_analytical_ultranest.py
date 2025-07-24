"""
UTF-8, Python 3.11.7

------------
Auroral Oval
------------

Ekaterina Ilin, 2025, MIT License, ilin@astron.nl


"""
import sys

import pandas as pd
import numpy as np

import astropy.units as u
from astropy.constants import c
from funcs.fitting_analytical import setup_model_sampler
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # med_min = "min"
    # full_sub = "full"
    # foreshortening = False
    # model = "ring"
    # ncalls = 10000

    # read from command line instead
    foreshortening = bool(int(sys.argv[1]))
    modelname = sys.argv[2]
    # ncalls = int(sys.argv[5])

    description = (f"Foreshortening: {foreshortening}\nmodel: {modelname}")
    
    extension = f"{modelname}_{foreshortening}"

    # rotation period in days
    P_rot = 2.845 / 24.
    omega = 2 * np.pi / P_rot

    # stellar radius in solar radii
    R_star = (1.07 * u.R_jup).to(u.R_sun).value

    # maximum rot velocity of the star in km/s
    vmax = omega * R_star * 695700. / 86400. # km/s

    # data
    df = pd.read_csv(f'data/lsr_norm_spectra.csv', index_col=0)

    # take wavelength above 6570 AA to calculate the std
    dferrs = df[df.index.astype(float) > 6570]
    ystds = dferrs.std(axis=0).values

    # select only index values between 6559 and 6566 
    df = df.loc[(df.index > 6559.5) & (df.index < 6566)]

    # lambda value convert to velocity
    x = df.index.values
    vs = ((x - 6562.8) / 6562.8 * c).to(u.km/u.s).value
    vrs = vs / vmax

    # take all 49 spectra
    ys_ = df.values

    # invert the order of ys and ystds
    ys_ = ys_[::-1]
    ystds = ystds[::-1]

    # convert to numpy arrays (and take square of the stds to get variance)
    ys = np.array(ys_).T
    yerr2 = (np.array(ystds).astype(float)**2)
    # reshape yerr2 from (49,) to (49, 1) for broadcasting
    yerr2 = yerr2.reshape(-1, 1)

    print(np.isfinite(yerr2).all())
    print(np.isfinite(ys).all())

    # get the rotational phase angles from the column names
    rots = (df.columns.values.astype(float) % 1) * 2 * np.pi

    # diagnostics figure
    off=0
    plt.figure(figsize=(6,15))
    for i, y in enumerate(ys):
        plt.scatter(vrs, y+off, s=2, c="steelblue")
        plt.axhline(1+off, c="k", ls="--")
        plt.text(vrs[-1]+.5, 1+off, f"{rots[i]/np.pi/2:.1f}")
        off += 0.8
    plt.xlim(vrs[0], vrs[-1])
    plt.ylim(1-0.3,2+off)
    plt.xlabel("Velocity [km/s]")
    plt.ylabel("Normalized flux")
    plt.text(vrs[-1] +1.5, 0.5+off, description)
    plt.tight_layout()
    plt.savefig(f"figures/lsr_norm_spectra_{extension}.png")

    # ---------------------------------------------------------------------------------------------------------------------
    # get the base curve for quiescent state
    base = pd.read_csv("curve_length_f.txt", dtype=float, sep=",")

    # interpolate base.curve_length column from base.vr to vrs
    curve_lengths_f = np.interp(vrs, base.vr.values, base.curve_length_f.values)

    # ---------------------------------------------------------------------------------------------------------------------
    # set up the sampler
    sampler, p1, model, loglike = setup_model_sampler(modelname, vrs, yerr2, ys, rots, foreshortening=foreshortening,
                                                      base=curve_lengths_f)
    
    # ---------------------------------------------------------------------------------------------------------------------
    # run sampler
    result = sampler.run(min_num_live_points=50, dlogz=0.5, viz_callback=False)#, max_ncalls=10000)
    sampler.print_results() 

    # ---------------------------------------------------------------------------------------------------------------------
    # show posterior distribution of live points
    samples = result["weighted_samples"]["points"]

    fig, axes = plt.subplots(1, len(p1), figsize=(15, 3))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.hist(samples[:,i], bins=40, histtype="step", color="k")
        ax.set_title(p1[i])
        ax.set_xlim(np.min(samples[:,i]), np.max(samples[:,i]))
    plt.tight_layout()
    plt.savefig(f"figures/posterior_{extension}.png")

    # ---------------------------------------------------------------------------------------------------------------------
    # plot best fit resultsT

    res = result["posterior"]["median"].copy()

    best_fit = model(*res)

    plt.figure(figsize=(5, 12))
    offset = 0
    for i in range(len(ys)):
        plt.plot(x, best_fit[i] + offset, label="best fit", c="k" ,zorder=10)
        plt.errorbar(x, ys[i] + offset, yerr = ystds[i], label="data", alpha=0.5, fmt="o") 
        plt.axhline(offset+1, color="black", alpha=0.5)
        plt.text(6566.5, offset+0.5, f"{rots[i]/np.pi/2:.1f}", fontsize=8)
        offset += 1

    plt.xlabel("Wavelength [A]")
    plt.ylabel("Normalized flux")
    plt.xlim(x[0], x[-1])
    plt.tight_layout()
    plt.savefig(f"figures/best_fit_{extension}.png")

    # ---------------------------------------------------------------------------------------------------------------------
    # print best fit parameters
    res = np.array(result["posterior"]["median"])
    res = np.round(res,2)

    for i, p in enumerate(p1):
        if (p[:3] == "lat") | (p[:3] == "lon")   | (p[:3] == "wid") | (p[:3] == "ima")  | (p[:3] == "phi") | (p[:4] == "dphi") | (p[:3] == "alp"):
            res[i] = res[i] * 180/np.pi
            res[i] = np.round(res[i],2)
            
        print(f"{p} = {res[i]}")

    # ---------------------------------------------------------------------------------------------------------------------
    # calculate AIC
    
    # number of parameters
    n = len(p1)        

    # lnL
    lnL = loglike(res)
    
    # AIC
    AIC = 2 * n - 2 * lnL


    # ---------------------------------------------------------------------------------------------------------------------
    # save results to a file

    # save res
    np.save(f"results/res_{extension}.npy", result)

    # now also save the median and std best_fit values to a table
    # init the table with the column names if it doesn't exist
    try:
        df = pd.read_csv(f"results/allspec_results_ultranest_{modelname}.csv")
    except:
        df = pd.DataFrame(columns=["model", "foreshortening", "AIC"] + p1)

    # add the new row
    df = pd.concat([df, pd.DataFrame([[modelname, foreshortening, AIC] + list(res)], columns=df.columns)], ignore_index=True)

    # save the table
    df.to_csv(f"results/allspec_results_ultranest_{modelname}.csv", index=False)

    print("\nDone!\n")





