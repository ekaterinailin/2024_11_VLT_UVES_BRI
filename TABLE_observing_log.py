"""
UTF-8, Python 3

------------
Auroral Rings
------------

Ekaterina Ilin, 2024, MIT License, ilin@astron.nl


Compile the observing log of the UVES data into a latex table.
"""


from astropy.io import fits
import glob
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # get all the files
    files = glob.glob('spectra/*.fits')

    # sort files by date
    files.sort()

    # extract these lines from each file and make a pandas dataframe with the columns MJD-OBS, AIRM, FWHM
    data = []
    for file in files[::-1]:
        hdu = fits.open(file)
        data.append([hdu[0].header["MJD-OBS"], hdu[0].header["HIERARCH ESO TEL AIRM START"], hdu[0].header["HIERARCH ESO TEL AMBI FWHM START"]])
        hdu.close()

    # make a pandas dataframe
    df = pd.DataFrame(data, columns=["MJD-OBS", "AIRM", "FWHM"])

    # rename to start of observation [MJD], airmass, seeing [arcsec]
    df.rename(columns={"MJD-OBS": "start of observation [MJD]", "AIRM": "airmass", "FWHM": "seeing [arcsec]"}, inplace=True)

    # round to 2 decimal places and convert to string airmass and seeing
    df["airmass"] = df["airmass"].apply(lambda x: f"{np.round(x, 2):.2f}")
    df["seeing [arcsec]"] = df["seeing [arcsec]"].apply(lambda x: f"{np.round(x, 2):.2f}")



    # convert to latex
    string = df.to_latex(index=False, escape=False)

    # replace rll with lll
    string = string.replace("rll}", "lll}\\hline")

    # replace midrule, top and bottom rule with hline
    string = string.replace("\\toprule", "\\hline")
    string = string.replace("\\bottomrule", "\\hline")

    # remove midrule
    string = string.replace("\\midrule", "\\hline")

    # put hline after [h] & \\
    string = string.replace("[h] &  \\\\", "[h] &  \\\\ \\hline")


    # write to file
    with open("uves_observing_log.tex", "w") as f:
        f.write(string)
        