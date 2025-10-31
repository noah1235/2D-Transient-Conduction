import numpy as np
from abc import ABC, abstractmethod


class Temp_BC(ABC):
    """
    Abstract base class for temperature (Dirichlet) boundary-condition providers.

    Implementations must support item access via:
        bc[i, t] -> T_b
    where
        i : int   (span-wise boundary index or position index)
        t : float (time)
        T_b : float (prescribed boundary temperature)
    """

    @abstractmethod
    def __getitem__(self, key):
        """
        Retrieve boundary temperature at (i, t).

        Parameters
        ----------
        key : tuple
            (i, t) → span-wise index i and time t.

        Returns
        -------
        float
            Boundary temperature T_b.
        """
        pass


class Const_Temp_BC:
    """
    Constant-in-time Dirichlet BC that is either:
      - spatially constant (scalar T), or
      - span-wise varying via an indexable array T[i].

    Note: Does not inherit Temp_BC explicitly, but implements the same interface.
    """

    def __init__(self, T):
        # If T is scalar → same temperature everywhere; else indexable by i.
        self.const_temp = isinstance(T, (int, float))
        self.T = T

    def __getitem__(self, key):
        """
        Return boundary temperature at (i, t). 't' is accepted for interface
        compatibility but ignored since this BC is time-invariant.
        """
        i, t = key  # t unused
        if self.const_temp:
            return self.T
        return self.T[i]


class Sin_Temp_BC:
    """
    Time-varying sinusoidal Dirichlet BC (no spatial phase variation by index).

    T_b(t) = mean + A * sin( (2π/P) * t + φ )

    'mean' and 'A' can be scalars or arrays indexed by i; P and φ are scalars.
    """

    def __init__(self, mean, A, P, phi):
        self.const_mean = isinstance(mean, (int, float))
        self.const_A = isinstance(A, (int, float))
        self.mean = mean  # scalar or array-like
        self.A = A        # scalar or array-like
        self.P = P        # period (time units)
        self.phi = phi    # phase (radians)

    def __getitem__(self, key):
        """
        Evaluate the sinusoidal BC at span-wise index i and time t.
        """
        i, t = key
        # Select per-index or scalar amplitude/mean
        A = self.A if self.const_A else self.A[i]
        mean = self.mean if self.const_mean else self.mean[i]
        # Sinusoid in time with fixed phase
        return A * np.sin((2 * np.pi) / self.P * t + self.phi) + mean


class T_Wave_BC:
    """
    Traveling wave Dirichlet BC in space-time.

    T_b(x, t) = mean + A * sin( (2π/P) * x + vel * t )

    Here:
      - x is derived from the index i via x = i * Δx_j
      - 'mean' and 'A' are scalars (as in the original implementation)
      - P is a spatial period (in x-units), vel is temporal angular speed.
    """

    def __init__(self, mean, A, P, vel, Delta_xj):
        self.mean = mean        # scalar mean temperature
        self.A = A              # scalar amplitude
        self.vel = vel          # temporal angular velocity term multiplying t
        self.P = P              # spatial period in same units as x
        self.Delta_xj = Delta_xj  # grid spacing along boundary-normal/parallel axis

    def __getitem__(self, key):
        """
        Evaluate the traveling wave at span-wise index i and time t.
        """
        i, t = key
        x = i * self.Delta_xj
        # Note: the formula uses spatial phase (2π/P)*x and temporal term vel*t
        return self.A * np.sin((2 * np.pi) / self.P * x + self.vel * t) + self.mean
