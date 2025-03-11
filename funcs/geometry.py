import numpy as np

import matplotlib.pyplot as plt



def set_up_oblique_auroral_ring(THETA, PHI, PHI_max, PHI_min,
                                i_rot, i_mag, amp, offset):
     """Set up an oblique auroral ring around the magnetic axis.

     Parameters
     ----------
     THETA : array
           1D array of theta values in radians.
     PHI : array
           1D array of phi values in radians.
     PHI_max : float
           Upper latitude of ring around magnetic axis in radians.
     PHI_min : float
           Lower latitude of ring around magnetic axis in radians.
     i_rot : float
           Inclination of rotation axis in radians with the right convention.
     i_mag : float
           Inclination of magnetic axis in radians relative to rotation axis.

     Returns
     -------
     (x, y, z) : tuple
           Tuple of arrays of x, y, z coordinates of points on the sphere at phase 0.
     z_rot : array
           rotation axis after rotation around y axis with the i_rot angle.
     z_rot_mag : array
           magnetic axis after rotation around y axis with the i_rot + i_mag angle.
     """
     

     # select the points on a sphere that are within the ring
     # around the magnetic axis
     q = ((THETA > (np.pi/2 - PHI_max)) &
          (THETA < (np.pi/2 - PHI_min)))

     # 3D rotation matrix for rotation around y axis with the i_rot + i_mag angle
     crotmag = np.cos(i_rot + i_mag)
     srotmag = np.sin(i_rot + i_mag)
     Rrotmag = np.array([[crotmag, 0, srotmag],
                    [0, 1, 0],
                    [-srotmag, 0, crotmag]])

     # 3D rotation matrix for rotation around y axis with the i_rot angle
     crot = np.cos(i_rot)
     srot = np.sin(i_rot)
     Rrot = np.array([[crot, 0, srot],
                    [0, 1, 0],
                    [-srot, 0, crot]])


     # rotate the points on the sphere
     stq = np.sin(THETA[q])
     x, y, z = np.dot(Rrotmag, np.array([stq * np.cos(PHI[q]),
                                   stq * np.sin(PHI[q]),
                                   np.cos(THETA[q])]))
     
     # rotate the z axis with Rrot
     z_rot = np.array([0, 0, 1])
     z_rot = np.dot(Rrot, z_rot)

     # rotate the z axis by i_rot + i_mag
     z_rot_mag = 1.5 * np.array([0, 0, 1])
     z_rot_mag = np.dot(Rrotmag, z_rot_mag)

     amplitude = amp * np.cos(PHI[q] + offset) + 1.
#      amplitude[PHI[q] > np.pi] = 0
#      plt.figure()
#      plt.scatter(x,amplitude, c='r')
#      qx = x>0
#      plt.scatter(x[qx],amplitude[qx], c='b')
#      plt.scatter(y,amplitude)
#      plt.scatter(z,amplitude)

     return (x, y, z), z_rot, z_rot_mag, amplitude

def create_spherical_grid(num_pts):
    """Method see:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    answered by CR Drost

    Coversion to cartesian coordinates:
    x = np.cos(PHI) * np.sin(THETA)
    y = np.sin(PHI) * np.sin(THETA)
    z = np.cos(THETA);

    Parameters:
    -----------
    num_pts : int
        number of grid points on the full sphere

    Return:
    --------
    THETA, PHI - numpy arrays of latitude, longitude
    """

    # This is CR Drost's solution to the sunflower spiral:
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    THETA = np.arccos(1 - 2 * indices/num_pts) #latitude
    PHI = np.pi * (1 + 5**0.5) * indices #longitude

    return THETA, PHI%(2*np.pi)


def rotate_around_arb_axis(a, pos, axis):
      """Rotate pos around axis with angle.
      
      Parameters:
      -----------
      a : float
            angle in radians
      pos : array of the form: [x, y, z] or (3, N)
            position(s) to be rotated
      axis : array of the form: [x, y, z]
            axis around which pos is rotated

      Return:
      --------
      pos_rot : array of the form: [x, y, z] or (3, N)
            rotated position(s)
      """
      # some shortcts
      ca = np.cos(a)
      cosa = 1 - ca
      sa = np.sin(a)
      ux, uy, uz = axis

      A = uz * ux * cosa
      B = uy * sa
      C = ux * sa
      D = uz * sa
      E = ux * uy * cosa
      F = uz * uy * cosa

      Rrotstar = np.array([[ca + ux**2 * cosa, E - D, A + B],
                              [E + D, ca + uy**2 * cosa, F - C],
                              [A - B, F + C, ca + uz**2 * cosa]])

      return np.dot(Rrotstar, pos)

def rotate_around_arb_axis_x_only(a, pos, axis):
      """Rotate pos around axis with angle.
      
      Parameters:
      -----------
      a : arrays
            angles in radians
      pos : array of the form: [x, y, z] or (3, N)
            position(s) to be rotated
      axis : array of the form: [x, y, z]
            axis around which pos is rotated

      Return:
      --------
      pos_rot : array of the form: [x, y, z] or (3, N)
            rotated position(s)
      """
      # some shortcts
      ca = np.cos(a) #1
      cosa = 1 - ca # 0
      sa = np.sin(a)
      ux, uy, uz = axis

      rot_x = np.array([[ca + ux**2 * cosa], [ux * uy * cosa - uz * sa], [uz * ux * cosa + uy * sa]])

      rot_x = rot_x.reshape(3, len(a))

      

      val = rot_x[0][:, np.newaxis] * pos[0] + rot_x[1][:, np.newaxis] * pos[1] + rot_x[2][:, np.newaxis] * pos[2]

      # print(val.shape)
      

      
      # print(rot_x)
      # print(pos.T[0])

      # res =   pos.T[0] * rot_x

      # print(res)
     
      # print(res.shape)
      return val



def calculate_surface_element_velocities(alpha, dalpha, x, y, z, z_rot, omega, Rstar, xalpha):
    """At a given phase angle alpha of the rotating star,
    calculate the radial velocity of the surface element from the
    x-component of the derivative
    
    ((x,y,z)(alpha + dalpha) - (x,y,z)(alpha)) / dalpha * omega

    where dalpha is a small angle in radians, x, y, z are the
    coordinates of the surface element in cartesian coordinates, and
    omega is the angular velocity of the star in radians per day.

    Convert the x-component of the derivative to km/s using Rstar and
    return their values for surface elements that are visible to the observer.

    Parameters
    ----------
    alpha : float
        phase angle in radians
    dalpha : float
        small angle in radians
    x : float
        x coordinate of surface element in cartesian coordinates
    y : float
        y coordinate of surface element in cartesian coordinates
    z : float
        z coordinate of surface element in cartesian coordinates
    z_rot : array
        rotation axis of the star in cartesian coordinates
    omega : float
        angular velocity of the star in radians per day
    Rstar : float
        stellar radius in solar radii

    Returns
    -------
    dxr_visible : array
        x-component of the derivative in km/s for surface elements
        that are visible to the observer.
    """

    # rotate the surface element around the rotation axis
    xr1 = xalpha

    # rotate the surface element around the rotation axis with an extra small angle
    xr2 = rotate_around_arb_axis_x_only(alpha + dalpha, np.array([x, y, z]), z_rot)

    # calculate the derivative of xr
    dxr = (xr2 - xr1) / dalpha * omega 

    # convert to stellar radii / s
    dxr = dxr / 86400.
     
    # convert to km/s
    dxr = dxr * Rstar * 695700.

    return dxr