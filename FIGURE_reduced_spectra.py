import pandas as pd
import matplotlib.pyplot as plt

from funcs.lsr import PROT_LSR
from funcs.bri import PROT_BRI

def plot_spectra(target):
    # get the spectra and rotation phases
    spectra = pd.read_csv(f"results_{target}/{target}_norm_spectra.csv", index_col=0)
    if target == "lsr": 
        prot = PROT_LSR * 24 # hours
        title = "LSR J1835"
        hour_time = (spectra.columns.values.astype(float) * prot) 
        spectra.columns = hour_time
    elif target == "bri":
        prot = PROT_BRI * 24  # hours
        title = "BRI 0021"
        time_list = spectra.columns.values
        hour_time = [float(t.split('_')[0]) + float(t.split('_')[1]) / 60 for t in time_list]
        spectra.columns = hour_time

    # plot the spectra in sequence
    plt.figure(figsize=(5, spectra.values.shape[1]*0.2 +2))
    off = 0
    for colname in spectra:
        plt.scatter(spectra.index, spectra[colname]+off, c="steelblue", s=1)
        plt.text(spectra.index[-1]+0.5, off+1, f"{colname:.2f}", fontsize=8)
        off += 1.2
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel("Normalized flux")   
    plt.xlim(spectra.index[0], spectra.index[-1])
    plt.axvline(6562.8, c="navy", ls="-", zorder=-10, linewidth=0.8)
    plt.text(spectra.index[-1]+0.4, off+.8, r"$\Delta t$ [h]", fontsize=10)
    plt.ylim(0.5, off+1.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"figures/{target}_spectra.png", dpi=300)

for target in ["lsr", "bri"]:
    plot_spectra(target)