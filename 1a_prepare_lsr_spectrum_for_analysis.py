
from astropy.io import fits

from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

import celerite2
from celerite2 import terms

import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from funcs.lsr import PROT_LSR


wavlabel = r"Wavelength [$\AA$]"
adulabel = "Flux [ADU]"

def fit_gp(x, maskedx, maskedy, maskedyerr):
    """GP fit to the data. 

    Parameters
    ----------
    x : array
        Wavelength array (including the region we want to predict)
    maskedx : array
        Wavelength array (excluding the region we want to predict)
    maskedy : array
        Flux array (excluding the region we want to predict)
    maskedyerr : array
        Flux error array (excluding the region we want to predict)

    Returns
    -------
    mu : array
        Mean of the GP prediction
    variance : array
        Variance of the GP prediction
    """

    # setup the kernel:

    # Quasi-periodic term
    term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)

    # Non-periodic component
    term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
    kernel = term1 + term2

    # Setup the GP
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    gp.compute(maskedx, yerr=maskedyerr)

    print("Initial log likelihood: {0}".format(gp.log_likelihood(maskedy)))

    # Plot the initial prediction
    def plot_prediction(gp):
        plt.errorbar(maskedx, maskedy, yerr=maskedyerr, fmt=".k", capsize=0, label="truth")

        if gp:
            mu, variance = gp.predict(maskedy, t=x, return_var=True)
            sigma = np.sqrt(variance)
            plt.plot(x, mu, label="prediction")
            plt.fill_between(x, mu - sigma, mu + sigma, color="C0", alpha=0.2)

            return mu, variance

    plt.title("Initial prediction")
    plot_prediction(gp)

    # log likelihood fit 

    def set_params(params, gp):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = terms.SHOTerm(
            sigma=theta[0], rho=theta[1], tau=theta[2]
        ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
        gp.compute(maskedx, diag=maskedyerr**2 + theta[5], quiet=True)
        return gp


    def neg_log_like(params, gp):
        gp = set_params(params, gp)
        return -gp.log_likelihood(maskedy)


    initial_params = [0.0, 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0), np.log(0.01)]
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
    opt_gp = set_params(soln.x, gp)
    
    plt.figure()
    plt.title("maximum likelihood prediction")
    return plot_prediction(opt_gp)



if __name__ == "__main__":

    # load data
    files = glob.glob('spectra/spectra_lsr/spec1d_HI.*.fits')

    data, error, wavs, timestamps = [], [], [], []
    for file in files:
        hdu = fits.open(file)
        timestamps.append(((hdu[0].header["MJD"] - 57578) / PROT_LSR)  - 2.07)
        
        wavs.append(hdu[8].data["OPT_WAVE"])
        data.append(hdu[8].data["OPT_COUNTS"])
        error.append(hdu[8].data["OPT_COUNTS_SIG"])

    wavs = np.array(wavs)
    data = np.array(data)
    error = np.array(error)

    # sort everything by timestamp
    timestamps = np.array(timestamps)
    sortidx = np.argsort(timestamps)[::-1]
    timestamps = timestamps[sortidx]
    wavs = wavs[sortidx]
    data = data[sortidx]
    error = error[sortidx]

    # take max and min of wavs[0] and wavs[1] as ranges for interpolation
    wmin = np.min(wavs.T[0])
    wmax = np.max(wavs.T[-1])

    wmin, wmax, len(wavs[0])

    wavs_interp = np.linspace(wmin, wmax, len(wavs[0]))
    data_interp = np.zeros((len(wavs), len(wavs_interp)))
    error_interp = np.zeros((len(wavs), len(wavs_interp)))

    for i in range(len(wavs)):
        data_interp[i] = np.interp(wavs_interp, wavs[i], data[i])
        error_interp[i] = np.interp(wavs_interp, wavs[i], error[i])

    data_interp = np.array(data_interp)
    error_interp = np.array(error_interp)

    coadd_err = np.sqrt(np.sum(error_interp**2, axis=0)) / len(data)

    plt.figure(figsize=(10, 6))
    for i in range(len(data_interp)):
        plt.plot(wavs_interp, data_interp[i])

    coadd = np.median(data_interp, axis=0)

    plt.plot(wavs_interp, coadd, label="coadd", lw=2, c="k")   
    plt.xlim(wavs_interp[0], wavs_interp[-1])
    plt.xlabel(wavlabel)
    plt.ylabel(adulabel)
    plt.savefig("figures/lsr_spectra_and_coadd.png", dpi=300)

    # DEFINE THE MASKS and THRESHOLDS -------------------------------------------------

    minrange, maxrange = 6562.8 - 10, 6562.8 + 20
    minhalpharange, maxhalpharange = 6562.8 - 2.2, 6562.8 + 2.1
    mincairange, maxcairange = 6566.2, 6589
    sigma_thresh = 3
    wavcai = 6572.78 # wavelength of the Ca I line

    # DIAGNOSTIC PLOT -- COADDED SPECTRUM ----------------------------------------------
    # plot the coadded spectrum
    plt.figure(figsize=(10, 5))
    plt.errorbar(wavs_interp, coadd, yerr=coadd_err, fmt='.-', alpha=0.5, c="grey")
    plt.xlim(minrange, maxrange)
    # plt.ylim(20, 150)
    plt.xlabel(wavlabel)
    plt.ylabel(adulabel)
    plt.tight_layout()
    plt.savefig("figures/lsr_raw_coadded_spectrum.png", dpi=300)

    # mask the region around the H alpha line and interpolate the continuum\

    mask = (wavs_interp > mincairange) & (wavs_interp < maxcairange)
    x, y, yerr = wavs_interp[mask], coadd[mask] / np.median(coadd[mask]), coadd_err[mask] /  np.median(coadd[mask])

    # FIT GP TO THE SPECTRUM TO FIND THE CA I LINE CENTER ----------------------------
    plt.figure()
    hires_x = np.linspace(mincairange, maxcairange, 2000)
    mucai, varcai = fit_gp(hires_x, x, y, yerr)
    plt.savefig("figures/lsr_gp_fit_cai_region.png", dpi=300)

    offset = hires_x[np.argmin(mucai)] - wavcai

    print(f"Ca I line center is offset by: {offset:.3f} AA")


    # SHIFT THE SPECTRUM --------------------------------------------------------------
    # shift the spectrum by the offset

    nwavs = wavs_interp - offset

    # now use only the region around the H alpha line 6562.8
    mask = (nwavs > minrange) & (nwavs < maxrange)

    x = nwavs[mask]
    y = coadd[mask] / np.median(coadd[mask])
    yerr = coadd_err[mask] / np.median(coadd[mask])

    # diagnostic plot   
    plt.figure(figsize=(8, 4))
    plt.errorbar(x, y, yerr=yerr, fmt='.-', alpha=0.5, c="grey")
    plt.axvline(6562.8, color='r', ls='--', label='rest frame')
    plt.xlabel(wavlabel)
    plt.ylabel("Normalized flux")
    plt.xlim(minrange, maxrange)
    plt.tight_layout()
    plt.savefig("figures/lsr_shifted_coadded_spectrum.png", dpi=300)

    # GP FIT THE CONTINUUM UNDER THE H ALPHA LINE --------------------------------

    # mask the region around the H alpha line 
    mask = (x > minhalpharange) & (x < maxhalpharange)
    maskedx, maskedy, maskedyerr = x[~mask], y[~mask], yerr[~mask]

    # GP fit the continuum
    plt.figure()
    mu, var = fit_gp(x, maskedx, maskedy, maskedyerr)

    # SUBTRACT THE CONTINUUM ------------------------------------------------------

    # init the subtracted spectra
    nspecs, newerrs = [], []

    # the final spectrum is in the defined range
    mask = (nwavs > minrange) & (nwavs < maxrange)
    nnwavs = nwavs[mask]

    # mask out the H alpha line where you want to subtract the continuum from
    hamask =  ~((nnwavs > minhalpharange) & (nnwavs < maxhalpharange))

    # loop over all spectra
    for spectrum, err in list(zip(data_interp, error_interp)):

        # mask the region around the H alpha line 
        ms = spectrum[mask]
        mms = spectrum[mask][hamask]
        mus = mu[hamask]

        # scale mu to the spectrum by using a constant factor
        def func(factor):
            return np.sum((mms - factor * mus)**2)
        
        # minimize the difference between the spectrum and the mu
        res = minimize(func, 1.0)

        # scale the continuum
        scaledmu = res.x[0] * mu

        # subtract the continuum
        subtracted = (ms - scaledmu) / res.x[0] + 1

        # propagate the errors
        err = np.sqrt(err[mask]**2 + var) / res.x[0]

        # store the subtracted spectrum
        nspecs.append(subtracted)
        newerrs.append(err)


    # PLOT THE SUBTRACTED SPECTRA -------------------------------------------------
    plt.figure(figsize=(10, 8))
    i = 0
    for nspec, nerr, tstamp in list(zip(nspecs, newerrs, timestamps)):
        plt.errorbar(nnwavs, nspec + i*0.8, yerr=nerr, alpha=0.3, c="grey", fmt=".-")
        plt.text(maxrange+0.1, 1. + i*0.8, f"{tstamp:.2f}", fontsize=10)
        
        i += 1

    plt.xlim(minrange, maxrange)
    plt.xlabel(wavlabel)
    plt.ylabel("Normalized flux")
    plt.tight_layout()
    plt.savefig("figures/lsr_subtracted_spectra.png", dpi=300)

    # mask outliers in the nspecs as positive 3 sigma outliers outside the halpha line using hamask
    outspecs = []

    # loop over all spectra
    for spec, err in list(zip(nspecs, newerrs)):

        # replace 3 sigma outliers with 1
        mask = np.abs(spec[hamask] - 1.) > sigma_thresh * err[hamask]

        # apply the mask to the whole spectrum
        outspec = np.copy(spec)
        outspec[hamask] = np.where(mask, 1., outspec[hamask])
        outspecs.append(outspec)

    # plot the outlier-free spectra
    plt.figure(figsize=(10, 10))
    i = 0
    for nspec, nerr, tstamp in list(zip(outspecs, newerrs, timestamps)):
        plt.errorbar(nnwavs, nspec + i*0.8, yerr=nerr, alpha=0.3, c="grey", fmt=".-")
        plt.text(maxrange+0.1, 1. + i*0.8, f"{tstamp:.2f}", fontsize=10)
        i += 1

    plt.axvline(6562.8, color='r', ls='--')

    plt.xlim(minrange, maxrange)
    plt.xlabel(wavlabel)
    plt.ylabel("Normalized flux (+ offset)")
    plt.tight_layout()
    plt.savefig("figures/lsr_subtracted_spectra_no_outliers.png", dpi=300)


    # EXPORT THE RESULTS -----------------------------------------------------------

    # write the spectra to a pandas dataframe with the wavelengths as index and the timestamps as columns
    df = pd.DataFrame(np.array(outspecs).T, index=nnwavs, columns=timestamps)

    # write the results
    df.to_csv('results_lsr/lsr_norm_spectra.csv')

    # MAKE A SMOOTHED GREYSCALE PLOT ------------------------------------------------

    smoothgrid = gaussian_filter1d(df.values[200:500].T[::-1], 0.1, axis=1)

    indices = df.index[200:500]

    plt.imshow(smoothgrid, aspect='auto', cmap='gray', extent=(indices[0],indices[-1],
                                                            max(timestamps) * PROT_LSR * 24,min(timestamps)*PROT_LSR*24,))
    plt.axvline(6562.8, color='w', linestyle='--')
    plt.xlabel(wavlabel)
    plt.ylabel("Time [h]")
    plt.savefig("figures/lsr_spectra_smoothed_greyscale.png", dpi=300)

