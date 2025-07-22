"""
UTF-8 
Python 3.10

Ekaterina Ilin 2025 -- MIT Licencse

Unit and integration tests for the analytical ring model.
"""

import numpy as np

from funcs.analytical_ring import get_equivr_lines, compute_curve_length


def test_get_equivr_lines():
    """Unit test for the get_equivr_lines function."""
    vrs = np.array([0, 1, 2])
    colat_min = 0
    colat_max = np.pi / 2
    obliquity = np.pi / 4
    rot = np.pi / 6
    xs, zs, masks = get_equivr_lines(vrs, colat_min, colat_max, obliquity, rot)
    
    assert xs.shape == (len(vrs), 160)
    assert zs.shape == (len(vrs), 160)
    assert masks.shape == (len(vrs), 160)

    # test that a high latitude 90 deg inclined ring at 0 rotation gives all empty xs and zs
    vrs_high_lat = np.array([0, 1, 2])
    colat_min_high = np.pi / 2
    colat_max_high = np.pi / 2 - 0.1
    obliquity_high = np.pi / 2
    rot_high = 0
    _, _, masks_high = get_equivr_lines(vrs_high_lat, colat_min_high, colat_max_high, obliquity_high, rot_high)
    assert np.all(masks_high == False)  # All points should be masked out

    # test that an equator-on ring will have values for vr=-1 and vr=1
    vrs_equator = np.array([-1, 0, 1])
    colat_min_equator = np.pi/4
    colat_max_equator = np.pi/4*3
    obliquity_equator = 0
    rot_equator = 0
    xs_equator, _, _ = get_equivr_lines(vrs_equator, colat_min_equator, colat_max_equator, obliquity_equator, rot_equator)
    assert np.any(xs_equator[1, :] != 0)  # At least one point should be non-zero for vr=0
    assert np.any(xs_equator[0, :] != 0)  # At least one point should be non-zero for vr=-1
    assert np.any(xs_equator[2, :] != 0)  # At least one point should be non-zero for vr=1

def test_compute_curve_length():
    """Unit test for the compute_curve_length function."""
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    z = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    masks = np.array([[True, True, True], [True, True, True]])
    
    length = compute_curve_length(x, z, masks)
    
    assert length.shape == (2,)
    assert np.all(length == 0)  # Length should be  zero for this simple case because the gaps are > 0.1

    # test with foreshortening
    length_foreshortened = compute_curve_length(x, z, masks, foreshortening=True)
    assert length_foreshortened.shape == (2,)   

    # create a case with some False values in masks
    masks_with_false = np.array([[True, False, True], [True, True, False]])
    length_with_false = compute_curve_length(x, z, masks_with_false)        
    assert length_with_false.shape == (2,)
    assert np.all(length == 0)  # Length should be  zero for this simple case because the gaps are > 0.1

    # now test with xs and zs with diffs smaller than 0.1
    x_small_diff = np.array([[1, 1.05, 1.15], [4, 4.05, 4.1]], dtype=float)
    z_small_diff = np.array([[1, 1.05, 1.15], [4, 4.05, 4.1]], dtype=float)
    masks_small_diff = np.array([[True, True, True], [True, True, True]])
    length_small_diff = compute_curve_length(x_small_diff, z_small_diff, masks_small_diff)  
    assert length_small_diff.shape == (2,)
    assert np.all(length_small_diff > 0)  # Length should be greater than zero

    # what should the exact value be
    expected_length = [np.sqrt((0.05**2)*2) + np.sqrt((0.1**2)*2), np.sqrt(0.05**2 *2)*2]
    assert np.allclose(length_small_diff, expected_length)  # Length should be equal to the expected value 

