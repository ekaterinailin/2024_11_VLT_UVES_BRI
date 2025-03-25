import numpy as np  

def create_spot(THETA, PHI, lat, lon, r):
  
    # select pole-on using just radial extent
    q = (THETA - r < 0)

    # rotation matrix around y-axis
    crotl = np.cos(lat)
    srotl = np.sin(lat)
    Rrot_lat = np.array([[crotl, 0, srotl],
                [0, 1, 0],
                [-srotl, 0, crotl]])

    # rotation matrix around z-axis
    crotp = np.cos(lon)
    srotp = np.sin(lon)
    Rrot_lon = np.array([[crotp, -srotp, 0],
                [srotp, crotp,0],
                [0, 0, 1]])

    # first around y, then around z-axis
    Rrot = Rrot_lon @ Rrot_lat

    # get x,y,z coordinates from phi and theta
    xyz = np.array([np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI), np.cos(THETA)])

    # get rotated coordinates
    xyz_rot = Rrot @ xyz

    # select those on the spot only
    return xyz_rot[0][q], xyz_rot[1][q], xyz_rot[2][q]


def rotate_inclination(xyz, inclination):

    # rotation matrix around the y axis again
    crot = np.cos(inclination)
    srot = np.sin(inclination)
    Rrot_y = np.array([[crot, 0, srot],
                [0, 1, 0],
                [-srot, 0, crot]])
    
    # rotate the stellar rotation axis
    z_rot = Rrot_y @ np.array([0,0,1])

    # rotate the spot to be at the right latitude for 
    xyz_rot = Rrot_y @ xyz

    return xyz_rot[0], xyz_rot[1], xyz_rot[2], z_rot   


def get_two_spots(THETA, PHI, lat1, lon1, r1, lat2, lon2, r2, inc):
 
    # define the two spots
    spot1 = create_spot(THETA, PHI, lat1, lon1, r1)
    spot2 = create_spot(THETA, PHI, lat2, lon2, r2)

    # combine the coordinates
    spots = np.concatenate((spot1, spot2), axis=1)

    # rotate around the y-axis by the inclination angle
    x, y, z, z_rot = rotate_inclination(spots, inc)

    # amplitude uniform for now
    amplitude = np.ones_like(x)

    # return z_rot twice for compatibility 
    return (x, y, z), z_rot, z_rot, amplitude


def get_one_spot(THETA, PHI, lat1, lon1, r1, inc):
 
    # define the two spots
    spot = create_spot(THETA, PHI, lat1, lon1, r1)

    # rotate around the y-axis by the inclination angle
    x, y, z, z_rot = rotate_inclination(spot, inc)

    # amplitude uniform for now
    amplitude = np.ones_like(x)

    # return z_rot twice for compatibility 
    return (x, y, z), z_rot, z_rot, amplitude

def get_point_sources(lats, lons, amps):

    x = np.sin(lats) * np.cos(lons)
    y = np.sin(lats) * np.sin(lons)
    z = np.cos(lats)
    z_rot = np.array([0,0,1])

    return (x, y, z), z_rot, z_rot, amps

