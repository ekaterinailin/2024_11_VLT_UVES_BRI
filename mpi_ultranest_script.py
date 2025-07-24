"""
MPI-parallelized version of the UltraNest script
Run with: mpirun -np <N_CORES> python run_full_analytical_ultranest_mpi.py <foreshortening> <modelname>
"""
import sys
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.constants import c
from funcs.fitting_analytical import setup_model_sampler
import matplotlib.pyplot as plt

# MPI imports
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    use_mpi = True
except ImportError:
    rank = 0
    size = 1
    use_mpi = False
    print("Warning: mpi4py not available, running without MPI")

if __name__ == "__main__":
    
    # Only rank 0 handles arguments and setup
    if rank == 0:
        foreshortening = bool(int(sys.argv[1]))
        modelname = sys.argv[2]
        
        description = f"Foreshortening: {foreshortening}\nmodel: {modelname}"
        extension = f"{modelname}_{foreshortening}"
        
        print(f"Running with MPI on {size} processes")
        print(description)
    else:
        foreshortening = None
        modelname = None
        extension = None
    
    # Broadcast parameters to all processes
    if use_mpi:
        foreshortening = comm.bcast(foreshortening, root=0)
        modelname = comm.bcast(modelname, root=0)
        extension = comm.bcast(extension, root=0)
    
    # Data loading and preprocessing (only rank 0 needs to do this initially)
    if rank == 0:
        # rotation period in days
        P_rot = 2.845 / 24.
        omega = 2 * np.pi / P_rot
        
        # stellar radius in solar radii
        R_star = (1.07 * u.R_jup).to(u.R_sun).value
        
        # maximum rot velocity of the star in km/s
        vmax = omega * R_star * 695700. / 86400.  # km/s
        
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
        yerr2 = yerr2.reshape(-1, 1)
        
        # get the rotational phase angles from the column names
        rots = (df.columns.values.astype(float) % 1) * 2 * np.pi
        
        # get the base curve for quiescent state
        base = pd.read_csv("curve_length_f.txt", dtype=float, sep=",")
        curve_lengths_f = np.interp(vrs, base.vr.values, base.curve_length_f.values)
        
        print("Data loaded and preprocessed")
    else:
        vrs = ys = yerr2 = rots = curve_lengths_f = x = ystds = None
    
    # Broadcast data to all processes
    if use_mpi:
        vrs = comm.bcast(vrs, root=0)
        ys = comm.bcast(ys, root=0)
        yerr2 = comm.bcast(yerr2, root=0)
        rots = comm.bcast(rots, root=0)
        curve_lengths_f = comm.bcast(curve_lengths_f, root=0)
        x = comm.bcast(x, root=0)
        ystds = comm.bcast(ystds, root=0)
    
    # Set up the sampler (all processes need this)
    sampler, p1, model, loglike = setup_model_sampler(
        modelname, vrs, yerr2, ys, rots, 
        foreshortening=foreshortening, base=curve_lengths_f
    )
    
    # Configure UltraNest for MPI
    if use_mpi:
        sampler.mpi_size = size
        sampler.mpi_rank = rank
    
    if rank == 0:
        print("Starting UltraNest sampling...")
    
    # Run sampler (MPI-parallelized)
    result = sampler.run(min_num_live_points=50, dlogz=0.5, viz_callback=False)
    
    # Only rank 0 handles output and plotting
    if rank == 0:
        sampler.print_results()
        
        # Rest of your plotting and analysis code...
        samples = result["weighted_samples"]["points"]
        
        # Diagnostics figure
        off = 0
        plt.figure(figsize=(6, 15))
        for i, y in enumerate(ys):
            plt.scatter(vrs, y+off, s=2, c="steelblue")
            plt.axhline(1+off, c="k", ls="--")
            plt.text(vrs[-1]+.5, 1+off, f"{rots[i]/np.pi/2:.1f}")
            off += 0.8
        plt.xlim(vrs[0], vrs[-1])
        plt.ylim(1-0.3, 2+off)
        plt.xlabel("Velocity [km/s]")
        plt.ylabel("Normalized flux")
        plt.text(vrs[-1] +1.5, 0.5+off, f"{description}\nMPI: {size} processes")
        plt.tight_layout()
        plt.savefig(f"figures/lsr_norm_spectra_{extension}.png")
        
        # Posterior plots
        fig, axes = plt.subplots(1, len(p1), figsize=(15, 3))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            ax.hist(samples[:,i], bins=40, histtype="step", color="k")
            ax.set_title(p1[i])
            ax.set_xlim(np.min(samples[:,i]), np.max(samples[:,i]))
        plt.tight_layout()
        plt.savefig(f"figures/posterior_{extension}.png")
        
        # Best fit plots
        res = result["posterior"]["median"].copy()
        best_fit = model(*res)
        
        plt.figure(figsize=(5, 12))
        offset = 0
        for i in range(len(ys)):
            plt.plot(x, best_fit[i] + offset, label="best fit", c="k", zorder=10)
            plt.errorbar(x, ys[i] + offset, yerr=ystds[i], label="data", alpha=0.5, fmt="o") 
            plt.axhline(offset+1, color="black", alpha=0.5)
            plt.text(6566.5, offset+0.5, f"{rots[i]/np.pi/2:.1f}", fontsize=8)
            offset += 1
        
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Normalized flux")
        plt.xlim(x[0], x[-1])
        plt.tight_layout()
        plt.savefig(f"figures/best_fit_{extension}.png")
        
        # Print best fit parameters
        res = np.array(result["posterior"]["median"])
        res = np.round(res, 2)
        
        for i, p in enumerate(p1):
            if (p[:3] == "lat") | (p[:3] == "lon") | (p[:3] == "wid") | (p[:3] == "ima") | (p[:3] == "phi") | (p[:4] == "dphi") | (p[:3] == "alp"):
                res[i] = res[i] * 180/np.pi
                res[i] = np.round(res[i], 2)
            print(f"{p} = {res[i]}")
        
        # Calculate AIC
        n = len(p1)
        lnL = loglike(res)
        AIC = 2 * n - 2 * lnL
        
        # Save results
        np.save(f"results/res_{extension}.npy", result)
        
        # Save to CSV table
        try:
            df_results = pd.read_csv(f"results/allspec_results_ultranest_{modelname}.csv")
        except:
            df_results = pd.DataFrame(columns=["model", "foreshortening", "AIC"] + p1)
        
        df_results = pd.concat([df_results, pd.DataFrame([[modelname, foreshortening, AIC] + list(res)], columns=df_results.columns)], ignore_index=True)
        df_results.to_csv(f"results/allspec_results_ultranest_{modelname}.csv", index=False)
        
        print(f"\nDone! Used {size} MPI processes\n")
