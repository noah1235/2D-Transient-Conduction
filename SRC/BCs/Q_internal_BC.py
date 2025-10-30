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
    


from abc import ABC, abstractmethod
import numpy as np

class Q(ABC):
    @abstractmethod
    def __call__(self, T, t):
        pass


class QSpatioTemporalArt(Q):
    """
    A 'really cool' spatio-temporal Q distribution:

        Q(x,y,t) = env(t) * [  Σ_i A_i * G_i(x,y,t)  +  A_wave * sin(2π(kx x + ky y - c t)) ]
                    + A_noise * N_filtered(x,y,t)  +  α * (T_target - T)   # optional feedback

    - Moving anisotropic Gaussians follow Lissajous-like trajectories with independent phases.
    - Traveling-wave 'carrier' adds coherent spatial structure.
    - Spatially filtered noise provides texture, slowly evolving in time.
    - Smooth raised-cosine envelope modulates overall intensity over time.
    - If desired, a mild temperature feedback term (α) can pull T toward T_target.

    Parameters
    ----------
    Nx, Ny : int
        Reference grid size (used to prebuild coordinate arrays). If T at call time
        has a different shape, the grid is rebuilt automatically to match T.shape.
    seed : int | None
        Random seed for reproducible blob positions, amplitudes, phases.
    n_blobs : int
        Number of moving Gaussian blobs.
    A_blob : float
        Peak amplitude scale for blobs (each blob gets a slight random scale).
    sigma_x, sigma_y : float
        Gaussian widths as fractions of domain size in x and y (0..1).
    lissajous_fx, lissajous_fy : float
        Base Lissajous frequencies for blob motion (perturbed per-blob).
    lissajous_amp : float
        Amplitude of blob path excursions as fraction of domain (0..0.5 recommended).
    drift : tuple[float, float]
        Constant drift velocity (vx, vy) in domain units per second (wrapped periodic).
    A_wave : float
        Amplitude of the traveling-wave component.
    kx, ky : float
        Spatial wavenumbers (cycles per unit domain) for the traveling wave.
    c : float
        Phase speed (units of domain-lengths per second) for the traveling wave.
    A_noise : float
        Amplitude of spatial noise after Gaussian filtering.
    noise_corr : float
        Spatial correlation length (fraction of domain) for noise Gaussian filter.
    noise_dt : float
        Temporal correlation time for noise (seconds); larger = slower-evolving noise.
    env_period : float
        Period (seconds) of the global envelope.
    env_exp : float
        Exponent for the raised-cosine envelope (≥1 sharpens peaks).
    alpha_feedback : float
        If >0, adds α*(T_target - T) to Q (simple proportional feedback).
    T_target : float
        Target temperature for feedback term.

    Notes
    -----
    - Domain is normalized to x∈[0,1), y∈[0,1).
    - All components are periodic in space; blob centers wrap around edges.
    """
    def __init__(self,
                 Nx: int,
                 Ny: int,
                 *,
                 seed: int | None = 0,
                 n_blobs: int = 3,
                 A_blob: float = 1.0,
                 sigma_x: float = 0.08,
                 sigma_y: float = 0.08,
                 lissajous_fx: float = 1.0,
                 lissajous_fy: float = 1.5,
                 lissajous_amp: float = 0.25,
                 drift: tuple[float, float] = (0.03, -0.02),
                 A_wave: float = 0.4,
                 kx: float = 4.0,
                 ky: float = 2.0,
                 c: float = 0.2,
                 A_noise: float = 0.2,
                 noise_corr: float = 0.06,
                 noise_dt: float = 1.2,
                 env_period: float = 6.0,
                 env_exp: float = 1.6,
                 alpha_feedback: float = 0.0,
                 T_target: float = 0.0):
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.Nx = Nx
        self.Ny = Ny

        # Blob parameters (each blob gets light randomness)
        self.n_blobs = n_blobs
        self.A_blob = A_blob
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.lx = lissajous_fx
        self.ly = lissajous_fy
        self.lamp = lissajous_amp
        self.vx, self.vy = drift

        self.blob_amp_jitter = 0.7 + 0.6 * self._rng.random(n_blobs)
        self.blob_phi_x = 2*np.pi * self._rng.random(n_blobs)
        self.blob_phi_y = 2*np.pi * self._rng.random(n_blobs)
        self.blob_fx = self.lx * (0.9 + 0.2 * self._rng.random(n_blobs))
        self.blob_fy = self.ly * (0.9 + 0.2 * self._rng.random(n_blobs))

        # Traveling wave
        self.A_wave = A_wave
        self.kx = kx
        self.ky = ky
        self.c = c

        # Noise (Ornstein-Uhlenbeck in time, Gaussian-filtered in space)
        self.A_noise = A_noise
        self.noise_corr = noise_corr
        self.noise_dt = max(1e-6, noise_dt)
        self._last_noise_t = None
        self._noise_field = None  # stored in real space

        # Envelope
        self.env_period = env_period
        self.env_exp = env_exp

        # Feedback
        self.alpha_feedback = alpha_feedback
        self.T_target = T_target

        # Build initial grid
        self._build_grid(Nx, Ny)

    # ---------------------- public API ----------------------

    def __call__(self, T: np.ndarray, t: float) -> np.ndarray:
        # Rebuild grid if T shape changed
        if T.shape != (self.Nx, self.Ny):
            self._build_grid(*T.shape)

        # Envelope
        env = self._envelope(t)

        # Moving Gaussian blobs
        blobs = self._moving_blobs(t)

        # Traveling wave
        wave = self.A_wave * np.sin(2*np.pi*(self.kx*self.X + self.ky*self.Y) - 2*np.pi*self.c*t)

        # Spatially filtered, temporally correlated noise
        noise = self._noise(t)

        Q = env * (blobs + wave) + noise

        # Optional temperature feedback
        if self.alpha_feedback != 0.0:
            Q = Q + self.alpha_feedback * (self.T_target - T)

        return Q

    # ---------------------- internals ----------------------

    def _build_grid(self, Nx: int, Ny: int):
        self.Nx, self.Ny = Nx, Ny
        x = np.linspace(0.0, 1.0, Nx, endpoint=False)
        y = np.linspace(0.0, 1.0, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        # Precompute Fourier k-space grids for noise filtering
        kx = np.fft.fftfreq(Nx, d=1.0/Nx) / Nx  # cycles per unit domain
        ky = np.fft.fftfreq(Ny, d=1.0/Ny) / Ny
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        K2 = self.KX**2 + self.KY**2
        # Gaussian filter in k-space with variance ~ 1/(2π*noise_corr)^2
        sigma_k = 1.0 / (2*np.pi*max(1e-6, self.noise_corr))
        self._noise_filter = np.exp(-0.5 * (K2 / (sigma_k**2)))

        # Reset noise state when grid changes
        self._noise_field = np.zeros((Nx, Ny))
        self._last_noise_t = None

    def _envelope(self, t: float) -> float:
        # Raised-cosine envelope on [0, env_period)
        phase = (t / max(1e-6, self.env_period)) % 1.0
        rc = 0.5 * (1.0 - np.cos(2*np.pi*phase))  # 0..1..0
        return rc ** max(1.0, self.env_exp)

    def _moving_blobs(self, t: float) -> np.ndarray:
        # Centers follow Lissajous with drift, all periodic
        # cx_i(t) = 0.5 + lamp * sin(2π fx_i t + φx_i) + vx t
        # cy_i(t) = 0.5 + lamp * sin(2π fy_i t + φy_i) + vy t
        cx = (0.5
              + self.lamp * np.sin(2*np.pi*self.blob_fx*t + self.blob_phi_x)
              + self.vx * t) % 1.0
        cy = (0.5
              + self.lamp * np.sin(2*np.pi*self.blob_fy*t + self.blob_phi_y)
              + self.vy * t) % 1.0

        # Blob widths (convert fractional sigmas to absolute units in [0,1])
        sx2 = (self.sigma_x**2)
        sy2 = (self.sigma_y**2)

        field = np.zeros((self.Nx, self.Ny))
        for i in range(self.n_blobs):
            # periodic distance (torus metric)
            dx = self._periodic_delta(self.X - cx[i])
            dy = self._periodic_delta(self.Y - cy[i])
            g = np.exp(-0.5 * (dx*dx / sx2 + dy*dy / sy2))
            field += (self.A_blob * self.blob_amp_jitter[i]) * g
        return field

    @staticmethod
    def _periodic_delta(d):
        # Map distances to [-0.5, 0.5) for periodic minimal image
        return (d + 0.5) % 1.0 - 0.5

    def _noise(self, t: float) -> np.ndarray:
        """
        Ornstein-Uhlenbeck in time (Euler step), Gaussian filtered in space.
        Evolves a latent white field in k-space, inverse-FFTs to real-space.
        """
        if self._last_noise_t is None:
            self._last_noise_t = t
            # start from a filtered random field
            eta = self._rng.standard_normal((self.Nx, self.Ny))
            nf = np.fft.fftn(eta)
            nf *= self._noise_filter
            real = np.fft.ifftn(nf).real
            self._noise_field = self.A_noise * real
            return self._noise_field

        dt = max(0.0, t - self._last_noise_t)
        self._last_noise_t = t

        # OU update: dX = -(1/τ) X dt + σ sqrt(2/τ) dW
        tau = self.noise_dt
        decay = np.exp(-dt / max(1e-6, tau))
        # innovation strength chosen so stationary var ~ A_noise^2
        sigma = self.A_noise * np.sqrt(1.0 - decay**2)

        eta = self._rng.standard_normal((self.Nx, self.Ny))
        nf = np.fft.fftn(eta)
        nf *= self._noise_filter
        real = np.fft.ifftn(nf).real

        self._noise_field = decay * self._noise_field + sigma * real
        return self._noise_field
