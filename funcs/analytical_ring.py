"""
UTF-8 
Python 3.10

Ekaterina Ilin 2025 -- MIT Licencse

Functions for the analytical ring model.
"""

import numpy as np


def get_equivr_lines(vrs, colat_min, colat_max, obliquity, rot, N=40):
    """Get the x and z for the ring at a given radial velocty.
    
    Parameters:
    -----------
    vrs : array-like
        Radial velocities for which to compute the ring lines.
    colat_min : float
        Minimum co-latitude of the ring in radians.
    colat_max : float
        Maximum co-latitude of the ring in radians.
    obliquity : float
        Magnetic obliquity in radians.
    rot : float
        Rotation phase in radians.
    N : int, optional
        Number of points to compute along the ring, default is 40. Sums to 4*N points in total.

    Returns:
    --------
    xs : numpy array
        x coordinates of the ring lines.
    zs : numpy array
        z coordinates of the ring lines.
    masks : numpy array
        Boolean mask indicating valid points on the ring.
    
    """
    crot, srot = np.cos(rot), np.sin(rot)
    cobl, sobl = np.cos(obliquity), np.sin(obliquity)

    colats = np.linspace(colat_min, colat_max, N)
    ccolat = np.tile(np.cos(colats), 4)  # (4N,)
    scolat = np.tile(np.sin(colats), 4)  # (4N,)
    colats = np.tile(colats, 4)

    pluspi = np.concatenate([
        np.zeros(N),
        np.ones(N) * np.pi,
        np.zeros(N),
        np.ones(N) * np.pi
    ])

    sigs = np.concatenate([
        np.ones(2 * N),
        -np.ones(2 * N)
    ])

    F = scolat * crot         # (4N,)
    F2 = F**2
    G = scolat * srot * cobl
    D = ccolat * sobl * srot

    vrs = np.asarray(vrs)     # (M,)
    M = len(vrs)
    xs = np.zeros((M, 4 * N), dtype=float)
    zs = np.zeros((M, 4 * N), dtype=float)
    masks = np.zeros((M, 4 * N), dtype=bool)

    # Mask for valid vrs
    valid_mask = np.abs(vrs) < 1.+1e-8
    if not np.any(valid_mask):
        return xs, zs, masks  # early exit if all are invalid

    vrs_valid = vrs[valid_mask]             # (M_valid,)
    vr2_valid = vrs_valid**2                # (M_valid,)
    D2 = (-vrs_valid[:, None] - D[None, :])**2  # (M_valid, 4N)

    sqrt_term = np.emath.sqrt(-D2**2 + D2 * F2 + D2 * G**2)
    tphi = (sigs[None, :] * sqrt_term + F[None, :] * G[None, :]) / (D2 - F2)

    phi = np.arctan(tphi).real + pluspi[None, :]
    cphi = np.cos(phi).real
    sphi = np.sin(phi).real

    scolat_cphi = scolat[None, :] * cphi
    x_phi = crot * (cobl * scolat_cphi - sobl * ccolat[None, :]) - srot * scolat[None, :] * sphi
    z_phi = sobl * scolat_cphi + cobl * ccolat[None, :]

    mask = ((np.abs(x_phi**2 + vr2_valid[:, None] + z_phi**2 - 1) < 1e-4) &
            (x_phi > 0))

    # Assign results only to valid rows
    xs[valid_mask, :] = x_phi
    zs[valid_mask, :] = z_phi
    masks[valid_mask, :] = mask

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
        return np.zeros(x.shape[0])
    
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
    


