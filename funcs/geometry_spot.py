import numpy as np  
from .background import add_background

def create_spot(THETA, PHI, lat, lon, r, background):

    q = (THETA - r < 0)

    q, PHI, THETA, amplitude = add_background(q, THETA, PHI, background)


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
    return xyz_rot[0][q], xyz_rot[1][q], xyz_rot[2][q], amplitude


def create_croissant(THETA, PHI, lat, lon, r):
  
    # q = (THETA - r < 0)
    # q1 = (np.cos(PHI)*np.tan(THETA) < r) & (THETA < (np.pi/2)) 
    # q2 =( PHI < np.pi/2) | (PHI > 3*np.pi/2)
    # q = q1 & q2

    q1 = THETA > lat
    q2 = THETA < (lat + r ) * np.power(np.abs(PHI - np.pi), 0.7) / np.pi
    q3 = THETA < (lat + r/7)
    qa = q1 & (q2 | q3)


    # amplitude should go quadratically with PHI-np.pi
    amplitude = np.ones_like(PHI)
    amplitude[qa] = 1 + 0.5*np.abs((PHI[qa] - np.pi))
    amplitude[qa] = amplitude[qa] * (1 - ((THETA[qa] - lat -r/3 )/np.pi)**2)
    amplitude[amplitude < 1] =1.

    q = np.ones_like(PHI).astype(bool)
    

    
    # amplitude = np.ones_like(THETA) 

    # smoot function from -np.pi to np.pi in PHI
    

    # rotation matrix around y-axis
    # crotl = np.cos(lat)
    # srotl = np.sin(lat)
    # Rrot_lat = np.array([[crotl, 0, srotl],
    #             [0, 1, 0],
    #             [-srotl, 0, crotl]])

    # rotation matrix around z-axis
    crotp = np.cos(lon)
    srotp = np.sin(lon)
    Rrot_lon = np.array([[crotp, -srotp, 0],
                [srotp, crotp,0],
                [0, 0, 1]])

    # first around y, then around z-axis
    Rrot = Rrot_lon #@ Rrot_lat

    # get x,y,z coordinates from phi and theta
    xyz = np.array([np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI), np.cos(THETA)])

    # get rotated coordinates
    xyz_rot = Rrot @ xyz

    # select those on the spot only
    return (xyz_rot[0][q], xyz_rot[1][q], xyz_rot[2][q]), amplitude[q]


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


def get_one_spot(THETA, PHI, lat1, lon1, r1, inc, background, croissant=False):
 
    # define the two spots
    # spot = create_spot(THETA, PHI, lat1, lon1, r1)
    if croissant:
        spot, amplitude = create_croissant(THETA, PHI, lat1, lon1, r1)
    else:
        _ = create_spot(THETA, PHI, lat1, lon1, r1, background)
        spot = _[:3]
        amplitude = _[3]

    # rotate around the y-axis by the inclination angle
    x, y, z, z_rot = rotate_inclination(spot, inc)

    # amplitude uniform for now
    # amplitude = np.ones_like(x)

    # return z_rot twice for compatibility 
    return (x, y, z), z_rot, z_rot, amplitude

def get_point_sources(lats, lons, amps):

    x = np.sin(lats) * np.cos(lons)
    y = np.sin(lats) * np.sin(lons)
    z = np.cos(lats)
    z_rot = np.array([0,0,1])

    return (x, y, z), z_rot, z_rot, amps

