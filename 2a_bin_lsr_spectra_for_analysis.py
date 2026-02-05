import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VARIANCE_THRESHOLD = 0.2
NBINS = 20

# path to doppler imaging location
dopplerpath = "/home/ilin/Documents/2025_10_pydoppler/scripts"

# read spectra from csv file
df = pd.read_csv('results_lsr/lsr_norm_spectra.csv')

# rename unnamed: 0 to lambda
df = df.rename(columns={'Unnamed: 0': 'lambda'})

# retype the column names to float except lambda
df.columns = ['lambda'] + [float(x)%1 for x in df.columns[1:].values]


# DEFINE THE BINS ----------------------------------------------------------------

# check that there are at least two spectra in each bin of width 0.05 in phase
phases = np.array([float(x) for x in df.columns[1:].values])
hist, bins = np.histogram(phases, bins=np.linspace(0, 1, NBINS+1))
binmids = (bins[:-1] + bins[1:]) / 2
assert (hist >= 2).all(), "There are bins with less than 2 spectra!"

# make a histogram of the number of spectra in each bin for diagnostic purposes
plt.plot(binmids, hist, drawstyle='steps-post')
plt.ylim(0,4)
plt.xlim(0,1)
plt.xlabel('Phase')
plt.ylabel('Number of spectra in bin')
plt.savefig('results_lsr/lsr_histogram_of_spectra_per_bin.png')

# ----------------------------------------------------------------------------------


# MEDIAN SPECTRA IN BINS ------------------------------------------------------------
# create a new metadata file lsr183520bins.fas
# make a diagnostic plot of the median spectra in each bin, offset by a constant for visibility
with open(f"{dopplerpath}/lsr1835bins20/lsr1835bins20.fas", "w") as f:

    plt.figure(figsize=(10, 6))

    # cycle through the bins
    for i in range(len(bins)-1):

        # select the phases covered in the current bin
        bin_phases = phases[(phases >= bins[i]) & (phases < bins[i+1])]

        # if there are spectra in the bin, calculate the median spectrum and plot it
        if len(bin_phases) > 1:

            # select the spectra in the current bin
            bin_spectra = df.iloc[:, 1:].loc[:, bin_phases]

            # get the median
            median_spectrum = bin_spectra.median(axis=1)

            # plot the median spectrum
            plt.plot(df['lambda'], median_spectrum+i, label=f'Phase {bins[i]:.2f}-{bins[i+1]:.2f}')

            # new df with lambda and median_spectrum
            median_spectrum = pd.DataFrame({'lambda': df['lambda'], 'flux': median_spectrum})
           
            # get bin center
            col = (bins[i]+bins[i+1])/2

            # format the column name as lsr_0p123456 to conform with pydoppler
            colname = f"{float(col)%1:.6f}".replace(".","p")

            # no header, separate is two spaces
            median_spectrum.to_csv(f"{dopplerpath}/lsr1835bins20/lsr_{colname}", 
                                   header=False, index=False, sep=" ")

            # convert to float
            colphase = float(col)

            # write the column name and phase to the metadata file
            f.write(f"lsr_{colname} {colphase}\n")
        else:
            raise ValueError(f"Bin {bins[i]:.2f}-{bins[i+1]:.2f} has less than 2 spectra!")
# ---------------------------------------------------------------------------------------


# DIAGNOSTIC PLOT: MEDIAN SPECTRA IN BINS WITH GREY OVERPLOT OF ALL SPECTRA IN BIN --

plt.figure(figsize=(15, 10))
for i in range(len(bins)-1):
    bin_phases = phases[(phases >= bins[i]) & (phases < bins[i+1])]
    if len(bin_phases) > 0:
        bin_spectra = df.iloc[:, 1:].loc[:, bin_phases]
        median_spectrum = bin_spectra.median(axis=1)

        plt.subplot(5, 5, i+1)
        for col in bin_phases:
            plt.plot(df['lambda'], df[col], color='grey', alpha=0.3)
        plt.plot(df['lambda'], median_spectrum, color='red')
        plt.title(f'Phase {bins[i]:.2f}-{bins[i+1]:.2f}')
        plt.ylim(0.8, 4)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------


# DIAGNOSTIC PLOT: VARIANCE SPECTRA IN BINS ------------------------------------------------

plt.figure(figsize=(15, 10))
for i in range(len(bins)-1):
    bin_phases = phases[(phases >= bins[i]) & (phases < bins[i+1])]
    if len(bin_phases) > 0:
        bin_spectra = df.iloc[:, 1:].loc[:, bin_phases]
        variance_spectrum = bin_spectra.var(axis=1)

        plt.subplot(4, 5, i+1)
        plt.plot(df['lambda'], variance_spectrum, color='blue')
        plt.title(f'Phase {bins[i]:.2f}-{bins[i+1]:.2f}')
        plt.ylim(-0.1, 0.5)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------


# MASK HIGHLY VARIABLE REGIONS IN SPECTRA and CALCULATE NOISE -------------------------

# for the spectra in each bin, calculate the variance as a function of lambda 
# and mask the regions with variance > 0.2 with NaN
median_spectra = []
errvals_all = []

# loop through each bin to get variance measure
for i in range(len(bins)-1):

    # select the phases covered in the current bin
    bin_phases = phases[(phases >= bins[i]) & (phases < bins[i+1])]

    # if there are at least two spectra in the bin, calculate the variance spectrum
    if len(bin_phases) > 1:

        # select the spectra in the current bin
        bin_spectra = df.iloc[:, 1:].loc[:, bin_phases]

        # get the variance spectrum
        variance_spectrum = bin_spectra.var(axis=1)

        # mask the regions with variance > 0.2 with NaN
        mask = variance_spectrum > VARIANCE_THRESHOLD

        # remove the regions with variance > 0.2 from the bin spectra
        bin_spectra.loc[mask, :] = np.nan

        # recalculate the median spectrum
        median_spectrum = bin_spectra.median(axis=1)

        # append the median spectrum to the list of median spectra
        median_spectra.append(median_spectrum)

        # save each median spectrum to a new csv and write to metadata file lsr183520bins.fas
        median_spectrum = pd.DataFrame({'lambda': df['lambda'], 'flux': median_spectrum})

        # delete the rows with NaN in median_spectrum
        median_spectrum = median_spectrum.dropna()

        # get bin center
        col = (bins[i]+bins[i+1])/2

        # format the column name as lsr_0p123456 to conform with pydoppler
        colname = f"{float(col)%1:.6f}".replace(".","p")

        # no header, separate is two spaces to conform with pydoppler
        median_spectrum.to_csv(f"{dopplerpath}/lsr1835bins20/lsr_{colname}", header=False, index=False, sep=" ")


        # now we take the bin spectra and measure the std in the regions
        # outside of the H alpha line and write it to a text file
        minhalpharange, maxhalpharange = 6562.8 - 1.2, 6562.8 + 1.2
        ha_mask = (df['lambda'] > minhalpharange) & (df['lambda'] < maxhalpharange)

        # for each spectrum in the bin, measure the std in the regions outside of 
        # the H alpha line and then add the values quadratically
        errvals = []
        for col in bin_phases:
            spec = df[col]
            std = np.std(spec[~ha_mask])
            errvals.append(std)
            
        # add the values quadratically
        total_err = np.sqrt(np.sum(np.array(errvals)**2))/len(errvals)
        print(f"Bin {bins[i]:.2f}-{bins[i+1]:.2f}: total error = {total_err:.3f}")
        errvals_all.append(total_err)
    else:
        raise ValueError(f"Bin {bins[i]:.2f}-{bins[i+1]:.2f} has less than 2 spectra!")

# write the total error to a text file
with open("results_lsr/subtracted_spectra_errvals.txt", "w") as f:
    for i in range(len(bins)-1):

        f.write(f"{bins[i]}: {errvals_all[i]:.3f}\n")

# -----------------------------------------------------------------------------------------