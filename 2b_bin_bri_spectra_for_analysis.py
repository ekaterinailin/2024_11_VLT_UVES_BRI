import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from funcs.bri import PROT_BRI

# path to doppler imaging location
dopplerpath = "/home/ilin/Documents/2025_10_pydoppler/scripts"

# get delta lambda for vsini
lambda0 = 6562.8 * u.AA  # H-alpha line
c = 299792.458 * u.km / u.s
vsini = 34.2 * u.km / u.s
dlambda_vsini = lambda0 * (vsini / c)
print(f"Delta lambda for vsini={vsini}: {dlambda_vsini:.2f}")

# read in the normalized spectra
df = pd.read_csv('results_bri/bri_norm_spectra.csv')

# CONVERT TIME TO PHASE -----------------------------------------------------------------

def convert_time_to_phase(time_str,prot):
    """Convert time string in format 'HH:MM:SS' to phase in degrees."""
    h, m = map(int, time_str.split('_'))
    total_min = h * 60 + m 
    phase = ((total_min / 60) / prot) % 1 
    return phase

prot = PROT_BRI * 24

# rename unnamed: 0 to lambda
df = df.rename(columns={'Unnamed: 0': 'lambda'})

# retype the column names to float except lambda
df.columns = ['lambda'] + [convert_time_to_phase(x, prot) for x in df.columns[1:].values]

df.head()

phases = df.columns[1:].values
phases = np.sort(phases)

# -----------------------------------------------------------------------------------------


# DIAGNOSTIC PLOT: plot all spectra with offset ----------------------------------------
plt.figure(figsize=(7, 16))
for i, phase in enumerate(phases):
    plt.plot(df['lambda'], df[phase] + (i%4)*1.5, label=f'Phase: {phase:.2f}')
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Normalized Flux + offset')
plt.xlim(6560,6565)
plt.savefig('results_bri/all_spectra_offset.png', dpi=300)

# -------------------------------------------------------------------------------------

# now rebin the specta by a factor of 5 in wavelength to improve noise properties
def rebin_spectrum(wavelengths, fluxes, bin_size):
    """Rebin the spectrum by averaging over bin_size."""
    num_bins = len(wavelengths) // bin_size
    rebinned_wavelengths = np.mean(wavelengths[:num_bins*bin_size].reshape(-1, bin_size), axis=1)
    rebinned_fluxes = np.mean(fluxes[:num_bins*bin_size].reshape(-1, bin_size), axis=1)
    rebinned_fluxes_std = np.std(fluxes[:num_bins*bin_size].reshape(-1, bin_size), axis=1)
    return rebinned_wavelengths, rebinned_fluxes, rebinned_fluxes_std    

bin_size = 5
rebinned_data = {'lambda': []}
for phase in phases:
    wavelengths = df['lambda'].values
    fluxes = df[phase].values
    rebinned_wavelengths, rebinned_fluxes, rebinned_fluxes_std = rebin_spectrum(wavelengths, fluxes, bin_size)
    rebinned_data['lambda'] = rebinned_wavelengths
    rebinned_data[phase] = rebinned_fluxes

    rebinned_data[f"{phase:.6f}_std"] = rebinned_fluxes_std 

rebinned_df = pd.DataFrame(rebinned_data)

# ------------------------------------------------------------------------------------------

# DIAGNOSTIC PLOT: plot the rebinned spectra with error bars for one phase  
plt.figure(figsize=(7, 16))
for i, phase in enumerate(phases):
    plt.plot(rebinned_df['lambda'], rebinned_df[phase] + (i%4)*1.5,label=f'Phase: {phase:.2f}')
    plt.axhline((i%4)*1.5+1, color='gray', linestyle='--', alpha=0.5,)
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Normalized Flux + offset')
plt.xlim(6560,6565)
plt.axvline(6562.8, color='gray', linestyle='--', alpha=0.5)
plt.axvline(6562.8 - dlambda_vsini.to(u.AA).value, color='red', linestyle='--', alpha=0.5)
plt.axvline(6562.8 + dlambda_vsini.to(u.AA).value, color='red', linestyle='--', alpha=0.5,
            label=f"vsini = {vsini:.1f}")
plt.legend(loc='upper right')
plt.savefig('results_bri/rebinned_spectra_offset.png', dpi=300)

# -----------------------------------------------------------------------------------------


# BIN THE SPECTRA IN PAIRS OF TWO TO IMPROVE SNR FURTHER --------------------------------

# sort columns by phase
sorted_phases = np.sort(phases)
df = df[['lambda'] + list(sorted_phases)]

# take the mean of every two consqutive spectra
df_mean = pd.DataFrame()
df_mean['lambda'] = rebinned_df['lambda']
mean_phases = []
for i in range(0, len(sorted_phases)-1, 2):
    phase1 = sorted_phases[i]
    phase2 = sorted_phases[i+1]
    mean_phase = (phase1 + phase2) / 2
    df_mean[mean_phase] = (rebinned_df[phase1] + rebinned_df[phase2]) / 2 
    df_mean[f"{mean_phase:.6f}_std"] = np.sqrt(rebinned_df[f"{phase1:.6f}_std"]**2 + rebinned_df[f"{phase2:.6f}_std"]**2) / 2 
    mask = (rebinned_df['lambda'] < 6562.8 - 1.) | (rebinned_df['lambda'] > 6562.8 + 1.)
    df_mean.loc[mask, f"{mean_phase:.6f}_std"] = df_mean.loc[mask, f"{mean_phase:.6f}_std"] 
    df_mean[f"{mean_phase:.6f}_var"] = rebinned_df[[phase1,phase2]].var(axis=1) 
    mean_phases.append(mean_phase)

    
# sort columns by phase
mean_phases = np.sort(mean_phases)

# ------------------------------------------------------------------------------------------

# PREPARE THE FILES FOR DOPPLER IMAGING ----------------------------------------------------

# setup meta file for each out put file
with open(f"{dopplerpath}/bri_4/bri_4.fas", "w") as f:
    
    for phase in mean_phases:
        colname = f"{float(phase)%1:.6f}".replace(".","p")
        print(colname)
        error_col = f"{float(phase):.6f}_std"

        # no header, separate is two spaces
        df_mean[["lambda", phase, error_col]].to_csv(f"{dopplerpath}/bri_4/lsr_{colname}", 
                                                     header=False, index=False, sep=" ")

        colphase = float(phase)%1
        f.write(f"{dopplerpath}/lsr_{colname} {colphase}\n")

# ------------------------------------------------------------------------------------------

# DIAGNOSTIC PLOT: plot the mean spectra with error bars for one phase --------------------

plt.figure(figsize=(7, 16))
for i, phase in enumerate(mean_phases):
    error_col = f"{float(phase):.6f}_std"
    plt.errorbar(df_mean['lambda'], df_mean[phase] + (i%4)*1.5, yerr=df_mean[error_col], 
                 label=f'Phase: {phase:.2f}', fmt='-o', markersize=2, elinewidth=1, capsize=2)
    plt.plot(df_mean['lambda'], df_mean[f"{phase:.6f}_var"]+ (i%4)*1.5 +1)
    mask = df_mean[f"{phase:.6f}_var"]>.15
    plt.plot(df_mean['lambda'][mask], df_mean[f"{phase:.6f}_var"][mask]+ (i%4)*1.5 +1, 'ro')
plt.axvline(6562.8, color='gray', linestyle='--', alpha=0.5)
plt.axvline(6562.8 - dlambda_vsini.to(u.AA).value, color='red', linestyle='--', alpha=0.5)
plt.axvline(6562.8 + dlambda_vsini.to(u.AA).value, color='red', linestyle='--', alpha=0.5,
            label=f"vsini = {vsini:.1f}")
plt.legend(loc='upper right')
plt.xlabel('Wavelength (Angstrom)')
plt.ylabel('Normalized Flux + offset')
plt.xlim(6560,6565)
plt.savefig('results_bri/mean_spectra_offset.png', dpi=300)

# --------------------------------------------------------------------------
