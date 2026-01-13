import numpy as np

def add_background(q_shape, THETA, PHI, background):
    """Add uniform background to a given shape mask on the sphere."""

    if background == 0:
        amplitude = np.ones_like(THETA[q_shape])

        return q_shape, PHI, THETA, amplitude
    
    else:
        amplitude_background = np.ones_like(THETA) * background

        amplitude_shape = np.zeros_like(THETA)
        amplitude_shape[q_shape] = 1.0

        amplitude = amplitude_shape + amplitude_background

        q = np.ones_like(THETA).astype(bool)

        return q, PHI, THETA, amplitude