
from .analytical import get_analytical_spectral_line
from .numerical import numerical_spectral_line
from .geometry import create_spherical_grid, set_up_oblique_auroral_ring, rotate_around_arb_axis
from .geometry_spot import get_one_spot
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
                 latitude2 = None, longitude2 = None, width2 = None, Rstar=None, 
                 amps=None, typ="ring"):
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

        self.typ = typ
        



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
    def get_flux_numerically(self, alpha, normalize=True, foreshortening=False,
                            ):
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
                                                                            self.i_rot, self.i_mag)
        
        self.amplitude = np.copy(np.broadcast_to(self.amplitude, (len(alpha),len(self.amplitude))))


        # calculate the flux
        flux, weights, q, self.xr, self.dxr = numerical_spectral_line(alpha, self.x, self.y, self.z, self.z_rot,
                                       self.omega, self.Rstar, self.v_bins, self.amplitude, normalize=normalize,
                                       foreshortening=foreshortening)
        
        self.q = q
        return flux
    
    def get_spot_flux_numerically(self, alpha, normalize=True, foreshortening=False,):
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
        (self.x, self.y, self.z), self.z_rot, self.z_rot_mag, self.amplitude = get_one_spot(self.THETA, self.PHI, 
                                                                            self.latitude, self.longitude, self.width,
                                                                            self.i_rot) 
     
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
        ax.set_zlim(-1.2, 1.2)

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

        fig = plt.figure(figsize=(8, 8))
        spec = fig.add_gridspec(ncols=1, nrows=1)

        ax = fig.add_subplot(spec[0, 0], projection='3d')
        ax.set_axis_off()

        ax.set_box_aspect([1,1,1])  # aspect ratio is 1:1:1

        return fig, ax
    

    def plot_sphere_only(self, ax, sphere_alpha=0.05, c="#00204C", s=1,
                       ):

        ax.scatter(np.sin(self.THETA)*np.cos(self.PHI),
            np.sin(self.THETA)*np.sin(self.PHI),
            np.cos(self.THETA), c=c, alpha=sphere_alpha, s=s)
        
        
    def plot_spot(self, ax, alpha, c="navy", ring_alpha=0.5, s=1):

        # crot = np.cos(self.i_rot)
        # srot = np.sin(self.i_rot)
        # Rrot = np.array([[crot, 0, srot],
        #                 [0, 1, 0],
        #                 [-srot, 0, crot]])
        
        # #rotate xr, yr, zr with Rrot
        # xr, yr, zr = rotate_around_arb_axis(self.i_rot, np.array([self.x, self.y, self.z]), [1,0,0])

        # rotation_axis = np.array([0,0,1])
        # rotation_axis = Rrot @ rotation_axis
        print(self.z_rot)
        
        
        xr, yr, zr = rotate_around_arb_axis(alpha, np.array([self.x, self.y, self.z]), self.z_rot)
        # xr, yr, zr =  self.x, self.y, self.z
        

        # THE SPOT ----------
        # use self.amplitude for color with viridis colormap
        ax.scatter(xr, yr, zr,  cmap="viridis", c=c, 
                                      norm="linear", alpha=ring_alpha, s=s)



    def plot_ring(self, ax, alpha, c="navy", c_irot="red", c_imag="yellow", ring_alpha=0.5, s=1):

        z_mag_alpha = rotate_around_arb_axis(alpha, self.z_rot_mag, self.z_rot)

        xr, yr, zr = rotate_around_arb_axis(alpha, np.array([self.x, self.y, self.z]), self.z_rot)

        # plot z_rot
        ax.plot([0, 1.5 *self.z_rot[0]], [0, 1.5 *self.z_rot[1]], [0,1.5 * self.z_rot[2]], c=c_irot)

        # THE RING ----------
        

        # plot the rotated blue points
        ax.scatter(xr, yr, zr, 
                   cmap="cividis", c=self.amplitude[0,:], norm="linear", alpha=ring_alpha, s=s)


        # plot z_rot_mag
        ax.plot([0, z_mag_alpha[0]], [0, z_mag_alpha[1]], [0, z_mag_alpha[2]], c=c_imag)


    def plot_lat_lon_grid(self, ax=None, num_lat=6, num_lon=12, show_surface=False):
        """
        Plot a latitude-longitude grid on a unit sphere (1-sphere).
        
        Parameters:
        -----------
        ax : matplotlib 3D axis, optional
            Existing 3D axis to plot on. If None, creates a new figure and axis.
        num_lat : int, default=9
            Number of latitude lines (including equator, excluding poles)
        num_lon : int, default=12
            Number of longitude lines (meridians)
        show_surface : bool, default=False
            Whether to show the sphere surface with transparency
        
        Returns:
        --------
        ax : matplotlib 3D axis object
        """
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
        
        # Draw latitude lines (circles of constant latitude)
        # Latitude ranges from -90° to +90° (or -π/2 to π/2 in radians)
        latitudes = np.linspace(-np.pi/2, np.pi/2, num_lat + 2)[1:-1]  # Exclude poles
        
        theta = np.linspace(0, 2*np.pi, 100)  # Azimuthal angle
        
        for lat in latitudes:
            # For a given latitude, radius of the circle is cos(lat)
            r = np.cos(lat)
            z = np.sin(lat)  # Height
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z_line = np.full_like(theta, z)
            
            ax.plot(x, y, z_line, 'k-', alpha=0.6, linewidth=0.8)
        
        # Draw equator (special case, thicker line)
        x_eq = np.cos(theta)
        y_eq = np.sin(theta)
        z_eq = np.zeros_like(theta)
        ax.plot(x_eq, y_eq, z_eq, 'k-', alpha=0.8, linewidth=1.)
        
        # Draw longitude lines (meridians from pole to pole)
        longitudes = np.linspace(0, 2*np.pi, num_lon, endpoint=False)
        
        phi = np.linspace(-np.pi/2, np.pi/2, 100)  # Polar angle from south to north pole
        
        for lon in longitudes:
            x = np.cos(phi) * np.cos(lon)
            y = np.cos(phi) * np.sin(lon)
            z = np.sin(phi)
            
            ax.plot(x, y, z, 'k-', alpha=0.6, linewidth=0.8)
        
        # Optionally draw the sphere surface
        if show_surface:
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_surf = np.outer(np.cos(u), np.sin(v))
            y_surf = np.outer(np.sin(u), np.sin(v))
            z_surf = np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x_surf, y_surf, z_surf, color='lightblue', 
                        alpha=0.2, edgecolor='none')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set axis limits
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        
        return ax