from abc import ABC, abstractmethod


class Heat_Flux_BC(ABC):
    """
    Abstract base class for heat-flux (Neumann) boundary-condition providers.

    Implementations should support:
        bc[i, t] -> q
    where
        i : int  (span-wise boundary index)
        t : float (time)
        q : float (prescribed outward heat flux, sign per your convention)
    """

    @abstractmethod
    def __getitem__(self, key):
        """
        Retrieve the heat flux q at (i, t).

        Parameters
        ----------
        key : tuple
            (i, t) where i is span-wise index and t is time.

        Returns
        -------
        q : float
            Heat flux value at that location/time.
        """
        pass


class Const_Heat_Flux_BC:
    """
    Heat-flux BC that is constant in time and either:
      - constant along the boundary (scalar q), or
      - varies span-wise via an indexable array q[i].

    Notes
    -----
    - This class does not inherit Heat_Flux_BC in the current design, but it
      implements the same `__getitem__` interface expected by callers.
    """

    def __init__(self, q):
        # If q is a scalar → same flux everywhere; otherwise q is indexable by i.
        self.const_q = isinstance(q, (int, float))
        self.q = q

    def __getitem__(self, key):
        """
        Return the heat flux at span-wise index i and time t (t is accepted for
        interface compatibility but not used since this BC is time-constant).

        Parameters
        ----------
        key : tuple
            (i, t) with i the span-wise index; t is ignored.

        Returns
        -------
        float
            q if scalar, else q[i].
        """
        i, t = key  # t unused; kept to match expected bc[i, t] signature
        if self.const_q:
            return self.q
        return self.q[i]
