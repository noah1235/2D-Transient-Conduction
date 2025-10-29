import numpy as np
from scipy.ndimage import gaussian_filter

def generate_random_temp_BC(Nx: int, Ny: int,
                            T_mean: float = 100.0,
                            amplitude: float = 5.0,
                            smooth_sigma: float = 5.0,
                            seed: int | None = None) -> np.ndarray:
    """
    Generate a smooth 2D random temperature boundary condition (non-periodic).

    Args:
        Nx, Ny : int
            Grid resolution in x and y directions.
        T_mean : float
            Mean/base temperature value.
        amplitude : float
            Magnitude of random perturbations (°C).
        smooth_sigma : float
            Standard deviation for Gaussian smoothing (controls smoothness).
        seed : int or None
            Random seed for reproducibility.

    Returns:
        T_field : np.ndarray
            (Nx, Ny) array of smooth random temperatures.
    """
    if seed is not None:
        np.random.seed(seed)

    # Start from random noise
    noise = np.random.randn(Nx, Ny)

    # Smooth it using a Gaussian filter
    smooth_noise = gaussian_filter(noise, sigma=smooth_sigma)

    # Normalize to unit std
    smooth_noise -= smooth_noise.mean()
    smooth_noise /= np.std(smooth_noise)

    # Add to base temperature
    T_field = T_mean + amplitude * smooth_noise

    return T_field