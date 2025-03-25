
from .analytical import get_analytical_spectral_line
from .numerical import numerical_spectral_line
from .geometry import create_spherical_grid, set_up_oblique_auroral_ring, rotate_around_arb_axis
from .geometry_spot import get_two_spots, get_one_spot, get_point_sources
import numpy as np
import matplotlib.pyplot as plt


THETA, PHI = create_spherical_grid(int(10000))

class AuroralRing:
    """A class to represent an auroral ring on a star.

    Attributes
    ----------
    i_rot : float
        The rotation inclination in rad.
    i_mag : float
        The magnetic obliquity in rad.
    latitude : float
        The mid-latitude of the ring in rad.
    width : float
        The width of the ring in rad.
    Rstar : float
        The radius of the star in solar radii.
    phi : array
        The phase angles of the ring in rad. From 0 to 2 pi of size N.
    omega : float
        The rotation rate of the star in rad / day.
    v_bins : array
        The velocity bins to use for the spectral line.
    lat_min : float
        The minimum latitude of the ring in rad.
    lat_max : float
        The maximum latitude of the ring in rad.
    v_mids : array
        The midpoints of the velocity bins.
    """

    # init function takes the parameters of the ring and sets up the phi array
    def __init__(self, i_rot, v_bins, v_mids, omega, vmax, phi=None, 
                 i_mag=None, latitude=None, width=None, longitude=None,
                 latitude2 = None, longitude2 = None, width2 = None, Rstar=None, amps=None):
        """Initialize the AuroralRing class.

        Parameters
        ----------
        i_rot : float
            The rotation inclination in rad.
        i_mag : float
            The magnetic obliquity in rad.
        latitude : float
            The mid-latitude of the ring in rad.
        width : float
            The width of the ring in rad.

        N : int 
            The number of phase angles to use for the ring.
            The same is used for the velocity bins.
        gridsize : int
            The number of grid points to use for the numerical
            calculation of the ring.
        v_bins : array
            The velocity bins to use for the spectral line.
        v_mids : array
            The midpoints of the velocity bins.
        phi : array
            The phase angles of the ring in rad. From 0 to 2 pi.
        omega : float
            The rotation rate of the star in rad / day.
        convert_to_kms : float  
            The conversion factor to convert from stellar radii / s to km / s.
        """
        self.i_rot = i_rot

        self.omega = omega
        self.vmax = vmax 
        self.phi = phi
        self.v_bins = v_bins
        self.v_mids = v_mids



        # calculate max and min latitude of the ring using width
        self.i_mag = i_mag
        self.latitude = latitude
        self.longitude = longitude  
        self.width = width
        self.lat_min = latitude - width/2
        self.lat_max = latitude + width/2

        self.latitude2 = latitude2
        self.longitude2 = longitude2
        self.width2 = width2
        self.amps = amps


        self.THETA, self.PHI = THETA, PHI

        self.Rstar = Rstar
        



    # define a method to get the flux of the ring
    def get_flux_analytically(self, alpha, foreshortening=False, normalize=True):
        """Calculate the flux of the ring at a given rotational phase.

        Parameters
        ----------
        alpha : float
            The rotational phase of the star in rad.
        forehortening : bool
            Whether to include geometric (Lambertian) foreshortening in the calculation.

        Returns
        -------
        flux : array
            The flux of the ring at the given rotational phase.
        """
        return get_analytical_spectral_line(self.phi, self.i_rot, self.i_mag, self.latitude, 
                                            alpha, self.v_bins, self.vmax, foreshortening=foreshortening,
                                            normalize=normalize)
    
    # define a method to get the flux of the ring numerically
    def get_flux_numerically(self, alpha, amp, offset, width, normalize=True, foreshortening=False):
        """Calculate the flux of the ring at a given rotational phase.

        Parameters
        ----------
        alpha : array
            The rotational phase of the star in rad.
        normalize : bool
            Whether to normalize the flux.
        foreshortening : bool
            Whether to include geometric (Lambertian) foreshortening in the calculation.

        Returns
        -------
        flux : array
            The flux of the ring at the given rotational phase.
        """
        # get the x, y, z positions of the ring
        (self.x, self.y, self.z), self.z_rot, self.z_rot_mag, self.amplitude = set_up_oblique_auroral_ring(self.THETA, self.PHI, 
                                                                            self.lat_max, self.lat_min, 
                                                                            self.i_rot, self.i_mag, amp, offset, width)
        

        
        # reshape output to the size of alpha array as the second dimension]

        # print(self.amplitude)
        self.amplitude = np.copy(np.broadcast_to(self.amplitude, (len(alpha),len(self.amplitude))))
        # print(self.amplitude)

        # calculate the flux
        flux, weights, q, self.xr, self.dxr = numerical_spectral_line(alpha, self.x, self.y, self.z, self.z_rot,
                                       self.omega, self.Rstar, self.v_bins, self.amplitude, normalize=normalize,
                                       foreshortening=foreshortening)
        
        
        # self.amplitude = weights
        self.q = q
        return flux
    
    def get_spot_flux_numerically(self, alpha, normalize=True, foreshortening=False, nspots=1):
        """Calculate the flux of the ring at a given rotational phase.

        Parameters
        ----------
        alpha : float
            The rotational phase of the star in rad.
        normalize : bool
            Whether to normalize the flux.
        foreshortening : bool
            Whether to include geometric (Lambertian) foreshortening in the calculation.

        Returns
        -------
        flux : array
            The flux of the ring at the given rotational phase.
        """
        # get the x, y, z positions of the ring
        if nspots == 2:
            (self.x, self.y, self.z), self.z_rot, self.z_rot_mag, self.amplitude = get_two_spots(self.THETA, self.PHI, 
                                                                            self.latitude, self.longitude, self.width,
                                                                            self.latitude2, self.longitude2, self.width2, 
                                                                            self.i_rot)
        elif nspots == 1:
            (self.x, self.y, self.z), self.z_rot, self.z_rot_mag, self.amplitude = get_one_spot(self.THETA, self.PHI, 
                                                                            self.latitude, self.longitude, self.width,
                                                                            self.i_rot) 
        elif nspots == 999:
            (self.x, self.y, self.z), self.z_rot, self.z_rot_mag, self.amplitude = get_point_sources(self.latitude, self.longitude, self.amps) 
            

        self.amplitude = np.copy(np.broadcast_to(self.amplitude, (len(alpha),len(self.amplitude))))
       
        # calculate the flux
        flux, weights, q, self.xr, self.dxr = numerical_spectral_line(alpha, self.x, self.y, self.z, self.z_rot,
                                       self.omega, self.Rstar, self.v_bins, self.amplitude, normalize=normalize,
                                       foreshortening=foreshortening)
        
        # self.amplitude = weights
        self.q = q
        return flux
    

    

    def get_phase_integrated_numerical_line(self, alpha):
        """Calculate the full flux of the ring integrated over several
        rotational phases alpha.

        Parameters
        ----------
        alpha : array
            The covered rotational phases of the star in rad.

        Returns
        -------
        full_flux_numerical : array
            The full flux of the ring integrated over the rotational phases.
        """

        # initialize the full flux array with the same shape as the velocity bins
        full_flux_numerical = np.zeros_like(self.v_mids)
        
        # loop over all rotational phases and add up the flux
        for a in alpha:
            full_flux_numerical += self.get_flux_numerically(a, normalize=False)

        # normalize the flux and return
        return full_flux_numerical / np.max(full_flux_numerical)
        

    def plot_sphere_with_auroral_ring(self, ax, alpha, c_ring="cyan",
                                      c_sphere="grey", c_irot="red",
                                      c_imag="yellow", sphere_alpha=0.1,
                                      ring_alpha=0.5):

        ax.scatter(np.sin(self.THETA)*np.cos(self.PHI),
              np.sin(self.THETA)*np.sin(self.PHI),
              np.cos(self.THETA), c="#00204C", alpha=sphere_alpha)

        # plot the x axis as a dashed line
        ax.plot([-1, 1], [0, 0], [0, 0], c='k', ls='--')

        z_mag_alpha = rotate_around_arb_axis(alpha, self.z_rot_mag, self.z_rot)

        xr, yr, zr = rotate_around_arb_axis(alpha, np.array([self.x, self.y, self.z]), self.z_rot)

        # plot z_rot
        ax.plot([0, 1.5 *self.z_rot[0]], [0, 1.5 *self.z_rot[1]], [0,1.5 * self.z_rot[2]], c=c_irot)

        # THE RING ----------
        

        # plot the rotated blue points
        ax.scatter(xr, yr, zr, 
                   cmap="cividis", c=self.amplitude, norm="linear", alpha=ring_alpha)


        # plot z_rot_mag
        ax.plot([0, z_mag_alpha[0]], [0, z_mag_alpha[1]], [0, z_mag_alpha[2]], c=c_imag)

    def plot_layout_sphere(self, ax, view="observer front"):
        # set figure limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-.95, .95)

        # label axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # rotate the figure such that x-axis point towards me
        if view == "observer front":
            ax.view_init(0, 0)
        elif view == "observer left":
            ax.view_init(0, 90)
        elif (type(view) is float) or (type(view) is int):
            ax.view_init(0, view)

        # let axes disappear
        ax.set_axis_off()


    def plot_setup_sphere(self):

        fig = plt.figure(figsize=(10, 5))
        spec = fig.add_gridspec(ncols=1, nrows=1)

        ax = fig.add_subplot(spec[0, 0], projection='3d')
        ax.set_axis_off()

        return fig, ax