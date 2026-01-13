import numpy as np
import xarray 
from xhistogram.xarray import histogram 
import matplotlib.pyplot as plt

from .geometry import rotate_around_arb_axis, calculate_surface_element_velocities, rotate_around_arb_axis_x_only


def numerical_spectral_line(alpha, x, y, z, z_rot, omega, Rstar, bins, amplitude,
                            dalpha=1e-8 * np.pi/180, normalize=True, foreshortening=False):
    """Calculate the broadened spectral line of the ring defined
    by x, y, z.

    Parameters
    ----------
    alpha : float
        The rotational phase of the star in rad.
    x : array
        The x positions of the ring in stellar radii.
    y : array
        The y positions of the ring in stellar radii.
    z : array
        The z positions of the ring in stellar radii.
    z_rot : float
        The rotational axis of the star in rad.
    omega : float
        The rotation rate of the star in rad / day.
    Rstar : float
        The radius of the star in solar radii.
    bins : array
        The velocity bins to use for the spectral line.
    dalpha : float
        The step size in alpha to use for velocity calculation.
    foreshortening : bool
        Whether to include geometric (Lambertian) foreshortening in the calculation.

    Returns
    -------
    flux : array
        The flux of the spectral line.
    """


    # rotate the ring
    xr = rotate_around_arb_axis_x_only(alpha, np.array([x, y, z]), z_rot)

    # print(xr)

    # calculate the surface element velocities
    dxr = calculate_surface_element_velocities(alpha, dalpha, x, y, z, z_rot, omega, Rstar, xr)


    # define the visible part of the ring
    q = xr > 0

    # print("q", q.shape)
    # print("amplitude", amplitude.shape)
    a = np.copy(amplitude)
    a[~q] = 0
    # amplitude[amplitude<0] = 0
    
    if foreshortening == False:
        weights = xarray.DataArray(np.ones_like(dxr) * a, dims=['phase', 'velocity'], name="weights")
    else:

        weights = xarray.DataArray(xr * a, dims=['phase', 'velocity'], name="weights")
   

    da = xarray.DataArray(dxr, dims=['phase', 'velocity'],
                  name='griddata')
    
    # print(da.values)
    
    # print(len(weights))
    # print(len(da))

    if weights.shape[1] > 0:
        flux = histogram(da, bins=[bins], dim=["velocity"], weights=weights)
        # binmids = (bins[1:] + bins[:-1]) / 2
        # off = 0
        # for f in flux:
        #     plt.plot(binmids, f+off)
        #     off += 1.5

    else:
        # print((weights.shape[0], len(bins)-1))
        flux = np.zeros((weights.shape[0], len(bins)-1))

    # print(flux.shape)
    # flux, _ = np.histogram(dxr, bins=bins, weights=weights)

    # print(flux.shape)

    # normalize the flux
    # throw error until i fix this
    if normalize:
        return ValueError("Normalization not yet implemented")
        # if max(flux) != 0:
            # flux = flux / np.max(flux)

    return flux, amplitude, q, xr, dxr