from abc import ABC, abstractmethod
import numpy as np

class Q(ABC):
    @abstractmethod
    def __call__(self, T, t):
        pass

class Q_gen(Q):
    def __init__(self, Q):
        self.Q = Q
    
    def __call__(self, T, t):
        return self.Q
    

class Q_gen_cent_square:
    def __init__(self, Q_val: float, Nx: int, Ny: int, px: float, py: float):
        """
        Generate a centered square heat source with given fraction sizes.

        Args:
            Q_val: Value to fill in the central region.
            Nx, Ny: Grid dimensions.
            px, py: Fractional widths (0 < p ≤ 1) of the square along x and y.
        """
        self.Q = np.zeros((Nx, Ny))

        # Compute half-width indices for the central square
        half_x = int((1 - px) * Nx / 2)
        half_y = int((1 - py) * Ny / 2)

        # Ensure indices stay valid
        x_start, x_end = half_x, Nx - half_x
        y_start, y_end = half_y, Ny - half_y

        self.Q[x_start:x_end, y_start:y_end] = Q_val

    def __call__(self, T: np.ndarray, t: float) -> np.ndarray:
        """Return the source field (independent of T, t)."""
        return self.Q