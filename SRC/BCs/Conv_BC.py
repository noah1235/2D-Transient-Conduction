import numpy as np
from abc import ABC, abstractmethod


class Conv_BC(ABC):
    """
    Abstract base class for convection (Robin) boundary-condition providers.

    Implementations must support item access via:
        bc[i, t] -> (h, T_inf)
    where
        i : int (span-wise index or position index along the boundary)
        t : float (time)
        h : float (convection coefficient at that location/time)
        T_inf : float (ambient/fluid temperature at that location/time)
    """

    @abstractmethod
    def __getitem__(self, key):
        """
        Retrieve (h, T_inf) for the given (index/time) key.

        Parameters
        ----------
        key : tuple
            A 2-tuple (i, t) where i is a span-wise index and t is time.

        Returns
        -------
        (h, T_inf) : tuple of floats
        """
        pass


class Const_Conv_BC(Conv_BC):
    """
    Convection BC with constant or span-wise-varying h and T_inf.

    Supports:
      - scalar h and/or T_inf (same across the boundary)
      - array-like h and/or T_inf indexed by i (span-wise)
    """

    def __init__(self, h, T_inf):
        # Flags indicate whether inputs are scalars (constant) or indexable arrays
        self.const_h = isinstance(h, (int, float))
        self.const_t_inf = isinstance(T_inf, (int, float))
        self.h = h
        self.T_inf = T_inf

    def __getitem__(self, key):
        """
        Return (h, T_inf) at span-wise index i and time t (t unused here but
        kept for interface compatibility).
        """
        i, t = key  # t is accepted but not used for constant-in-time data

        # Choose scalar or per-index value for h
        if self.const_h:
            h = self.h
        else:
            h = self.h[i]

        # Choose scalar or per-index value for T_inf
        if self.const_t_inf:
            T_inf = self.T_inf
        else:
            T_inf = self.T_inf[i]

        return h, T_inf


class Time_Space_Conv_BC(Conv_BC):
    """
    Convection BC varying in both space and time.

    Model:
        h(x, t) = h0 * [1 + amp_h * sin( 2π f_h t + k_h x )]
        T_inf(x, t) = T0   (constant here, but could be extended similarly)

    Parameters
    ----------
    h0 : float
        Baseline convection coefficient.
    T0 : float
        Ambient/fluid temperature (constant in this model).
    Delta_xj : float
        Spatial grid spacing along the boundary (used to convert index -> x).
    amp_h : float, optional
        Amplitude of sinusoidal variation (fraction of h0).
    f_h : float, optional
        Temporal frequency of the sinusoid [Hz].
    k_h : float, optional
        Spatial wavenumber of the sinusoid [rad/m].
    """

    def __init__(self, h0, T0, Delta_xj, amp_h=0.2,
                 f_h=0.2, k_h=2*np.pi):
        self.h0 = h0
        self.T0 = T0
        self.amp_h = amp_h
        self.f_h = f_h
        self.k_h = k_h
        self.Delta_xj = Delta_xj

    def __call__(self, x, t):
        """
        Evaluate BC at physical coordinate x (scalar or array) and time t.

        Returns
        -------
        (h, T_inf) : tuple
            h can be scalar or ndarray if x is array-like; T_inf is T0.
        """
        # Time- and space-dependent convection coefficient
        h = self.h0 * (1.0 + self.amp_h * np.sin(2*np.pi*self.f_h*t + self.k_h*x))
        return h, self.T0

    def __getitem__(self, key):
        """
        Backward-compatible indexing interface:
            bc[i, t] -> (h, T_inf)
        Internally maps index i to position x = i * Delta_xj.
        """
        i, t = key
        x = i * self.Delta_xj
        return self.__call__(x, t)
