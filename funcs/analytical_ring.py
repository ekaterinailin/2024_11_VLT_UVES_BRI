"""
UTF-8 
Python 3.10

Ekaterina Ilin 2025 -- MIT Licencse

Functions for the analytical ring model.
"""

import numpy as np


def get_equivr_lines(vrs, colat_min, colat_max, obliquity, rot, N=40):
    """
    Function to calculate the spectrum based on the given parameters.
    
    Parameters:
    -----------
    vrs : numpy array
        Doppler velocity range
    colat_min : float
        Minimum colatitude in radians
    colat_max : float
        Maximum colatitude in radians
    obliquity : float
        Obliquity in radians
    rot : float
        Rotation in radians
    N : int, optional
        Number of points in the colatitude range (default is 40) -- take x4 for the final length
    
    Returns:
    --------
    xs : np.ndarray
        A 2D array where each row corresponds to a different vr and contains the x coordinates of the ring at a given vr.
    zs : np.ndarray
        A 2D array where each row corresponds to a different vr and contains the z coordinates of the ring at a given vr.
    masks : np.ndarray
        A 2D boolean array where each row corresponds to a different vr and indicates whether the corresponding point is 
        visible on the hemisphere and lies on the sphere.
    -----------
    """
    #pre-compute the cosines and sines of the rotation and obliquity angles
    crot, srot = np.cos(rot), np.sin(rot)
    cobl, sobl = np.cos(obliquity), np.sin(obliquity)

    # create the colatitude range
    colats = np.linspace(colat_min, colat_max, N)

    # pre-compute the cosines and sines of the colatitudes    
    ccolat = np.cos(colats)
    scolat = np.sin(colats)

    # Tile values
    ccolat = np.tile(ccolat, 4)
    scolat = np.tile(scolat, 4)
    colats = np.tile(colats, 4)

    # pluspi structure: [0]*N, [π]*N, [0]*N, [π]*N
    pluspi = np.concatenate([
        np.zeros(N),
        np.ones(N) * np.pi,
        np.zeros(N),
        np.ones(N) * np.pi
    ])

    # sigs: [1]*2N, [-1]*2N
    sigs = np.concatenate([
        np.ones(2 * N),
        -np.ones(2 * N)
    ])

    # pre-compute the F, G, D values to save computation time
    F = scolat * crot 
    F2 = F**2
    G = scolat * srot * cobl
    D = ccolat * sobl * srot

    # fill a len(vrs) * 4*N array with zeros -- this will be the output dimensions
    xs = np.zeros((len(vrs), 4 * N), dtype=float)
    zs = np.zeros((len(vrs), 4 * N), dtype=float)
    masks = np.zeros((len(vrs), 4 * N), dtype=bool)
    
    # loop over the vrs
    for i, vr in enumerate(vrs):

        # calculate the vr^2
        vr2 = vr**2

        # calculate the D2 value
        D2 = (-vr - D)**2

        # tan(phi)
        tphi = (sigs*np.emath.sqrt(-D2**2 + D2*F2 + D2*G**2) + F*G) / (D2 - F2)

        # revert to phi, get real values, and apply pluspi
        phi = np.arctan(tphi).real + pluspi       
        
        # calculate the cosines and sines of phi
        cphi, sphi = np.cos(phi).real, np.sin(phi).real   

        # equations (10) and (11) get you the x and z coordinates of the ring at a given vr
        scolat_cphi = scolat * cphi
        x_phi = crot * (cobl * scolat_cphi - sobl * ccolat) - srot * scolat * sphi
        z_phi = sobl * scolat_cphi + cobl * ccolat

        # mask points outside the colatitude limits, and non-finite values   
        mask = ((np.abs(x_phi**2 + vr2 + z_phi**2 - 1) < 1e-4)& 
                (x_phi > 0))
        
        # store the points
        xs[i, :] = x_phi
        zs[i, :] = z_phi
        masks[i, :] = mask

    return xs, zs, masks



def compute_curve_length(x, z, masks, foreshortening=False):
    """Function to compute the length of a curve defined by x and z coordinates.

    Parameters:
    -----------
    x : numpy array
        x coordinates of the curve
    z : numpy array
        z coordinates of the curve

    Returns:
    --------
    length : float
        Length of the curve
    """
    # mask all the values that are not visible or don't lie on the sphere
    x[~masks] = np.nan
    z[~masks] = np.nan

    # if no valid points are present, return 0
    if np.isnan(x).all():
        return 0
    
    # otherwise, compute the length of the curves for all vrs
    else:
        # the dimension is (len(vrs), 4*N)
        sorted_indices = np.argsort(z, axis=1)

        # apply the sorting to x and z
        x = x[np.arange(x.shape[0])[:, None], sorted_indices]
        z = z[np.arange(z.shape[0])[:, None], sorted_indices]

        # get the differences in x and z coordinates for the curve segment length
        dx = np.diff(x, axis=1) 
        dz = np.diff(z, axis=1)

        # mask where dz is greater than 0.1 -- this is where the two halves of the ring are visible at one vr
        mask = dz > 0.1  

        # set masked values to NaN
        dx[mask], dz[mask] = np.nan, np.nan  

        
        if foreshortening:

            # if geometric foreshortening is applied use the x-midpoint for the foreshortening factor
            x = ((x[:,1:] + x[:,:-1]) / 2)
            return np.nansum(np.sqrt(dx**2 + dz**2) * np.abs(x), axis=1)
        
        else:
           
            # otherwise, just return the length of the curve, ignoring the nans
            return np.nansum(np.sqrt(dx**2 + dz**2), axis=1)
    


