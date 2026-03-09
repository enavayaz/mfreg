import numpy as np
import numpy.linalg as lg
import jax
from util_pred import eval_poly_dc
from jax import random as rnd
import jax.numpy as jnp
from functools import partial
from morphomatics.manifold import manifold, Sphere
from morphomatics.stats import ExponentialBarycenter as Mean
from util import bez_sph
eps = 1e-8

def map2D3D(x, y, uniform=True):
    #Z = np.zeros((n_points, 3))
    if uniform:
        Z = np.sqrt(1-y**2)*np.cos(x), np.sqrt(1-y**2)*np.sin(x), y
    else:
        Z = np.cos(y)*np.cos(x), np.cos(y)*np.sin(x), np.sin(y)  # central distribution
    return Z
#map2D3D = jax.vmap(map2D3D)


def generate_polynomial_series(M,
                               n_points: int = 30,
                               deg: int = 3,
                               noise_level: float = 0.1,
                               key=None,
                               ) -> np.ndarray:
    """
    Generate manifold time series with polynomial (Bézier) trend

    Creates smooth, predictable evolution that polynomial regression
    should capture well.

    Parameters
    ----------
    M : Manifold
        The manifold (e.g., SPD, Sphere)
    n_points : int
        Number of time points
    deg : int
        Polynomial degree (1=linear, 2=quadratic, 3=cubic)
    noise_level : float
        Noise level (0=deterministic, 0.5=very noisy)
        key : jax.random.PRNGKey, optional
        Random key for reproducibility. If None, generates a random one.

    Returns
    -------
    np.ndarray, shape (n_points, *M.point_shape)
        Manifold time series
    """
    # Initialize random keys
    if key is None:  # ADD THIS CHECK
        key = rnd.PRNGKey(np.random.randint(0, 2 ** 32))

    master_key = key  # CHANGE THIS LINE (was: random.PRNGKey(0))
    init_key, noise_key = rnd.split(key)

    # Generate time parameter
    t = np.linspace(0, 1, n_points)

    # Generate deg+1 random control points on the manifold
    ctl_pts = np.empty((deg + 1,) + M.point_shape)
    init_keys_array = rnd.split(init_key, deg + 1)

    for i in range(deg + 1):
        ctl_pts[i] = M.rand(init_keys_array[i])

    # Evaluate Bézier polynomial curve
    Y = eval_poly_dc(M, ctl_pts, t)

    # Add noise if requested
    if noise_level > 0:
        Y_noisy = add_correlated_noise_TS(M, Y, noise_key, noise_level, correlation=.8)
        #Y_noisy = add_gauss_noise(M, Y, noise_key, noise_level)
    else:
        Y_noisy = Y

    # Convert JAX array to NumPy array
    return np.asarray(Y_noisy)

# ============================================================================
# Add Noise to Trajectories
# ============================================================================

#@partial(jax.jit, static_argnums=(3,))
def add_correlated_noise_TS(
    M,
    Y: np.ndarray,
    key,
    noise_level: float,
    correlation: float = 0.8
) -> np.ndarray:
    """
    Add temporally correlated noise to time series on manifold.

    Creates realistic observation noise with temporal correlation,
    mimicking real-world measurement processes where consecutive
    observations have correlated errors.

    If Y is not a time series, just use add_gauss_noise

    Parameters
    ----------
    M : Manifold
        The manifold
    Y : np.ndarray
        Clean trajectory
    key : JAX random key
        Random seed
    noise_level : float
        Base noise magnitude
    correlation : float in [0, 1]
        Temporal correlation strength
        - 0 = independent noise (like add_noise_TS)
        - 1 = fully correlated (random walk noise)
        - 0.8 = realistic (recommended)

    Returns
    -------
    Y_noisy : np.ndarray
        Trajectory with correlated noise
    """
    n = len(Y)
    Y_noisy = np.empty_like(Y)
    Y_noisy[0] = Y[0]

    # Initialize noise direction
    key, subkey = rnd.split(key)
    noise_direction = M.randvec(Y[0], subkey)
    noise_direction = noise_direction / (np.linalg.norm(noise_direction) + eps)

    for i in range(1, n):
        key, subkey = rnd.split(key)

        # Update noise direction with correlation
        new_random = M.randvec(Y[i], subkey)
        new_random = new_random / (np.linalg.norm(new_random) + eps)

        # Blend: correlation * old + (1-correlation) * new
        noise_direction = correlation * M.metric.transp(Y[i-1], Y[i], noise_direction) + (1 - correlation) * new_random
        noise_direction = noise_direction / (np.linalg.norm(noise_direction) + eps)

        # Apply correlated noise
        noise = noise_level * noise_direction
        Y_noisy[i] = M.metric.exp(Y[i], noise)

    return Y_noisy


#@partial(jax.jit, static_argnums=(3,))
def add_gauss_noise(
        M,
        Y: np.ndarray,
        key,
        noise_level: float
) -> np.ndarray:
    """
    Add independent Riemannian Gaussian noise to each sample in Y on manifold.

    Creates i.i.d. observation noise where each error is independent
    of the previous one. This corresponds to the case where correlation = 0.

    Parameters
    ----------
    M : Manifold
        The manifold
    Y : np.ndarray
        Clean trajectory
    key : JAX random key
        Random seed
    noise_level : float
        Standard deviation of the noise magnitude

    Returns
    -------
    Y_noisy : np.ndarray
        Trajectory with independent Gaussian noise
    """
    n = len(Y)
    Y_noisy = np.empty_like(Y)

    for i in range(n):
        key, subkey = rnd.split(key)

        # Sample a random tangent vector (innovation)
        # Note: M.randvec usually samples from a standard normal in the tangent space
        noise_direction = M.randvec(Y[i], subkey)

        # Normalize to ensure isotropic direction if desired,
        # or leave as is for true Gaussian scaling.
        # Here we follow your convention of fixed-magnitude directional noise:
        noise_direction = noise_direction / (np.linalg.norm(noise_direction) + eps)

        # Apply independent noise
        noise = noise_level * noise_direction
        Y_noisy[i] = M.metric.exp(Y[i], noise)

    return Y_noisy

# ============================================================================
# Generate HIGHLY CORRELATED trajectories
# ============================================================================

def sph_correlated_trjs(lon_max, lat_max, n_trj=30, n_points=50, noise_std=0.03, mean_curve='Else',
                        between_std=2.0, temporal_correlation=0.95):
    """
    Generate trajectories with both within- and between-trajectory correlation.

    Within-trajectory correlation: noise is temporally smooth along each track
    (AR(1) process with temporal_correlation coefficient).

    Between-trajectory correlation: all trajectories share a common random
    perturbation drawn once per trajectory group, with individual noise added
    on top. This induces non-diagonal covariance in the control points across
    trajectories — the structure that ridge regression exploits.

    Parameters
    ----------
    lon_max, lat_max : float
        Spatial extent of the template
    n_trj : int
        Number of trajectories to generate
    n_points : int
        Number of points per trajectory
    noise_std : float
        Individual (within-trajectory) noise std.
        Lower = higher between-trajectory correlation.
    mean_curve : str
        Template shape: 'Geo', 'Poly', or 'Else'
    between_std : float
        Std of the shared perturbation (between-trajectory component).
        The between-trajectory correlation coefficient is:
            rho = between_std² / (between_std² + 1)
        E.g. between_std=2.0 → rho=0.80 (strong, recommended)
             between_std=3.0 → rho=0.90 (very strong)
             between_std=0.7 → rho=0.33 (weak)
    temporal_correlation : float in [0,1]
        AR(1) coefficient for within-trajectory noise smoothness.
        0 = independent, 0.95 = very smooth (default).

    Returns
    -------
    Y : list of arrays, each shape (n_points, 3)
    template : array, shape (n_points, 3)
    """
    from morphomatics.manifold import Sphere
    M = Sphere()

    # ── template ──────────────────────────────────────────────────────────────
    if mean_curve == 'Geo':
        start_point = np.array([1.0, 0.0, 0.0])
        end_point   = np.array([0.0, 1.0, 0.0])
        template    = np.array([M.metric.geopoint(start_point, end_point, t)
                                for t in np.linspace(0, 1, n_points)])
    elif mean_curve == 'Poly':
        template = bez_sph(n_points)
    else:  # 'Else': perturbed sine curve
        x_template = np.linspace(0, lon_max, n_points)
        y_max      = np.sin(lat_max)
        y_template = 0.5 * y_max * np.sin(np.pi * x_template / lon_max)
        z_template = map2D3D(x_template, y_template, uniform=True)
        template   = np.array([z_template[0], z_template[1], z_template[2]]).T

    # ── shared perturbation (drawn once for all trajectories) ─────────────────
    # This is the between-trajectory component: all tracks are displaced
    # in the same direction by a common random amount, inducing non-diagonal
    # covariance in their control points.
    shared_u = np.random.normal(0, between_std)
    shared_v = np.random.normal(0, between_std)

    # ── generate trajectories ─────────────────────────────────────────────────
    Y = []
    for i in range(n_trj):
        noisy_trj          = np.zeros((n_points, 3))
        prev_noise_u       = 0.0
        prev_noise_v       = 0.0

        for j in range(n_points):
            p = template[j]

            # local orthonormal frame in tangent space at p
            u = np.array([-p[1], p[0], 0.0])
            norm_u = lg.norm(u)
            if norm_u < eps:
                u = np.array([0.0, -p[2], p[1]])
                norm_u = lg.norm(u)
            u = u / norm_u
            v = np.cross(p, u)

            # individual noise (AR(1) for temporal smoothness)
            indiv_u = np.random.normal(0, noise_std)
            indiv_v = np.random.normal(0, noise_std)
            if j > 0:
                indiv_u = (temporal_correlation * prev_noise_u +
                           np.sqrt(1 - temporal_correlation**2) * indiv_u)
                indiv_v = (temporal_correlation * prev_noise_v +
                           np.sqrt(1 - temporal_correlation**2) * indiv_v)

            prev_noise_u, prev_noise_v = indiv_u, indiv_v

            # total noise = shared (between-trj) + individual (within-trj)
            total_u = shared_u * noise_std + indiv_u
            total_v = shared_v * noise_std + indiv_v

            noisy_trj[j] = M.metric.exp(p, total_u * u + total_v * v)

        Y.append(noisy_trj)

    return Y, template

# Generate list of random trajectories
def sph_rand_trjs(lon_max, lat_max, n_trj=30, n_points=30, uniform=True):
    Y = []
    for i in range(n_trj):
        n_points = np.random.randint(30, 70)
        # Generate x coordinates: random but sorted (increasing)
        x = np.sort(np.random.uniform(0, lon_max, n_points))
        #x =np.linspace(0,lon_max,n_points)

        # Generate y coordinates: random
        if uniform:
            y_max = np.sin(lat_max)
            y = np.random.uniform(-y_max, y_max, n_points)
        else:
            y = np.random.uniform(-lat_max, lat_max, n_points)
        y = np.sort(y)
        # map to sphere
        z = map2D3D(x, y, uniform=uniform)
        Y.append(np.array([z[0], z[1], z[2]]).T)
    return Y

def save_sph(B, Y, strTemp='Else'):
    path = '../datasets/sph' + strTemp + '.npz'
    np.savez(path, B=B, Y=np.array(Y, dtype=object))

def load_sph(strTemp='Else'):
    path = '../datasets/sph' + strTemp + '.npz'
    data = np.load(path, allow_pickle=True)
    B, Y = data['B'], data['Y'].tolist()
    return B, [np.array(y) for y in Y]

#==========================================
# SPD Random
#==========================================

# Random SPD Trajectories
def randA_spd(dim=2):
    A = np.random.randn(dim, dim)
    return np.dot(A, A.T)

def rand_spd_exp(dim=2):
    from scipy.linalg import expm
    A = np.random.randn(dim, dim)
    A_sym = (A + A.T) / 2  # make symmetric
    return expm(A_sym)

def rand_spd(dim=2):
    D = np.eye(dim)
    D[0, 0] = np.random.uniform(.5, 1.5)  # or exponential, etc.
    D[1, 1] = np.random.uniform(2.5, 3.5)
    #Q, _ = np.linalg.qr(np.random.randn(dim, dim))  # random orthogonal
    phi = np.random.uniform(-np.pi/20, np.pi/20)
    Q = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    return Q @ D @ Q.T

def generate_rand_trj(n_subj=10, n_points=50, dim=2):
    Y =[]
    for i in range(n_subj):
        spd = np.zeros((n_points, dim, dim))
        for j in range(n_points):
            spd[j] = rand_spd(dim)
        Y += [spd]
    return Y

#==========================================
# PGA
#==========================================

class PrincipalGeodesicAnalysis(object):
    """
    Principal Geodesic Analysis (PGA) as introduced by
    Fletcher et al. (2003): Statistics of manifold via principal geodesic analysis on Lie groups.
    """

    def __init__(self, mfd, data, mu=None):
        """
        Setup PGA.

        :arg mfd: underlying data space (Assumes that mfd#inner(...) supports list of vectors)
        :arg data: list of data points
        :arg mu: intrinsic mean of data
        """
        assert mfd.connec and mfd.metric
        self.mfd = mfd
        N = len(data)

        # assure mean
        if mu is None:
            mu = Mean.compute(mfd, data)
        self._mean = mu

        ################################
        # inexact PGA, aka tangent PCA
        ################################

        # map data to tangent space at mean
        v = jax.vmap(jax.jit(mfd.connec.log), (None, 0))(mu, data)

        # setup covariance operator
        v_vec = v.reshape(N, -1)
        C = 1/N * v_vec.T @ v_vec

        variances, modes, coeffs = self.compute_cov(C, v_vec)
        self.cov = C

        self._variances = variances
        self._modes = modes
        self._coeffs = coeffs

    def compute_cov(self, C, v):
        d = self.mfd.dim
        # decompose
        vals, vecs = jnp.linalg.eigh(C)

        # set variance and modes
        n = jnp.sum(vals > 1e-6)
        e = d - n - 1 if n<d else -d-1
        variances = vals[:e:-1]
        modes = vecs[:, :e:-1].T.reshape((n,) + self.mfd.point_shape)

        coeffs = v @ vecs[:,:e:-1]

        return variances, modes, coeffs