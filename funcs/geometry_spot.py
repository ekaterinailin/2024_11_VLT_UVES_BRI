import numpy as np  

def create_spot(THETA, PHI, lat, lon, r):

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

    amplitude = np.ones_like(THETA[q])

    # select those on the spot only
    return xyz_rot[0][q], xyz_rot[1][q], xyz_rot[2][q], amplitude



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




def get_one_spot(THETA, PHI, lat1, lon1, r1, inc):
 
    _ = create_spot(THETA, PHI, lat1, lon1, r1)
    spot = _[:3]
    amplitude = _[3]

    # rotate around the y-axis by the inclination angle
    x, y, z, z_rot = rotate_inclination(spot, inc)


    # return z_rot twice for compatibility 
    return (x, y, z), z_rot, z_rot, amplitude

